"""
Adjoint Transport Monte Carlo reverse sampler for MERFISH.

Key difference from the 2D torus case: we cannot use Bessel functions
for the 649-dim importance ratio. Instead we use the Gaussian CLT
approximation of the Skellam pseudo-marginal:

    log P(x_t | x_0) ≈ -||x_t - x_0||^2 / (4*mu) + const

This makes the importance ratio a pure squared-distance computation,
eliminating all scipy calls and enabling full PyTorch/MPS compatibility.

The spatial kernel K(p_i, p_j) is precomputed once from cell coordinates
and is fixed throughout the reverse pass.
"""

import numpy as np
import torch


class MerfishAdjointReverse:
    """
    Adjoint Transport Monte Carlo reverse sampler for the MERFISH Moran model.

    Parameters
    ----------
    gamma_mut : float  Per-gene mutation rate.
    kappa : float      Moran resampling rate.
    h_kernel : float   Spatial kernel bandwidth (microns).
    G : int            Number of genes.
    n_substeps : int   Substeps per denoising step.
    guidance_weight : float  omega in the adjoint rate formula.
    """

    def __init__(
        self,
        gamma_mut: float = 1.0,
        kappa: float = 0.5,
        h_kernel: float = 20.0,
        G: int = 649,
        n_substeps: int = 10,
        guidance_weight: float = 2.0,
    ):
        self.gamma_mut = gamma_mut
        self.kappa = kappa
        self.h_kernel = h_kernel
        self.G = G
        self.n_substeps = n_substeps
        self.guidance_weight = guidance_weight

    def _time_scale(self, t: float) -> float:
        return 1.0 / max(1.0 - t, 0.001) ** 2

    def _integrated_rate(self, t: float) -> float:
        """Integrated mutation rate S(t) = 1/(1-t) - 1."""
        return 1.0 / max(1.0 - t, 0.001) - 1.0

    def _spatial_kernel(self, coords: np.ndarray) -> np.ndarray:
        """
        Gaussian kernel on cell spatial coordinates.
        coords: [N, 2] in microns → [N, N] (diagonal 0)
        """
        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        K = np.exp(-dist_sq / (2.0 * self.h_kernel ** 2))
        np.fill_diagonal(K, 0.0)
        return K.astype(np.float32)

    def _log_pseudo_marginal(
        self,
        x: np.ndarray,
        x_hat: np.ndarray,
        mu: float,
    ) -> np.ndarray:
        """
        Gaussian CLT approximation of log P(x_t | x_0_hat).

        For each gene j, x_t^j - x_0^j ~ Skellam(mu, mu) ≈ N(0, 2*mu),
        so the joint log-likelihood (ignoring constants) is:

            log P(x_t | x_0) ≈ -||x_t - x_0||^2 / (4*mu)

        x:     [N, G] current state
        x_hat: [N, G] model's prediction of clean state
        mu:    float  integrated mutation parameter

        Returns: [N] log probabilities
        """
        diff_sq = np.sum((x.astype(np.float32) - x_hat.astype(np.float32)) ** 2, axis=-1)
        return -diff_sq / (4.0 * max(mu, 1e-4))

    def _reverse_mutation_step(
        self,
        x: np.ndarray,
        x_hat: np.ndarray,
        rate_mult: float,
        mu: float,
        dt: float,
    ) -> np.ndarray:
        """
        Guided reverse mutation (importance-weighted immigration-death).

        For gene j in cell i, the reverse rate toward x_hat is proportional
        to the ratio P(x_t - 1_j | x_hat) / P(x_t | x_hat).

        In the Gaussian approximation:
          log ratio = [(x_j - x_hat_j)^2 - (x_j ± 1 - x_hat_j)^2] / (4*mu)
                    = [±2*(x_j - x_hat_j) - 1] / (4*mu)

        We implement this as guided Poisson birth-death with the ratio
        as an acceptance weight on each proposed move.
        """
        N, G = x.shape
        x_new = x.copy().astype(np.int32)

        base_rate = self.gamma_mut * rate_mult * dt

        for i in range(N):
            for j in range(G):
                xij = float(x[i, j])
                xhat_ij = float(x_hat[i, j])
                diff = xij - xhat_ij

                # Guided birth: x_j → x_j + 1
                # ratio_birth = exp((-2*diff - 1) / (4*mu))
                log_ratio_birth = (-2.0 * diff - 1.0) / (4.0 * max(mu, 1e-4))
                r_birth = base_rate * np.exp(np.clip(log_ratio_birth * self.guidance_weight, -5, 5))

                # Guided death: x_j → x_j - 1 (only if x_j > 0)
                # ratio_death = exp((2*diff - 1) / (4*mu))
                if xij > 0:
                    log_ratio_death = (2.0 * diff - 1.0) / (4.0 * max(mu, 1e-4))
                    r_death = base_rate * np.exp(np.clip(log_ratio_death * self.guidance_weight, -5, 5))
                else:
                    r_death = 0.0

                if np.random.rand() < min(r_birth, 0.5):
                    x_new[i, j] = min(x_new[i, j] + 1, 500)
                if np.random.rand() < min(r_death, 0.5):
                    x_new[i, j] = max(x_new[i, j] - 1, 0)

        return x_new

    def _adjoint_resampling_step(
        self,
        x: np.ndarray,
        x_hat: np.ndarray,
        K: np.ndarray,
        log_p: np.ndarray,
        rate_mult: float,
        dt: float,
        N: int,
    ) -> np.ndarray:
        """
        Adjoint coalescent transport step.

        The adjoint rate for cell i to coalesce to cell j's profile:
          R(i → j) = (kappa_eff / N) * K(p_i, p_j) * exp(omega * (log_p[j] - log_p[i]))

        x:     [N, G]
        x_hat: [N, G]
        K:     [N, N] spatial kernel (diagonal 0)
        log_p: [N]    log pseudo-marginal for each cell
        """
        x_new = x.copy()

        # NxN importance ratio matrix
        log_ratio = log_p[None, :] - log_p[:, None]  # [N, N]: log p_j - log p_i
        importance = np.exp(np.clip(self.guidance_weight * log_ratio, -10, 10))

        # Full adjoint rate matrix
        rate_matrix = (self.kappa * rate_mult / N) * K * importance  # [N, N]

        # Per-cell total rate → jump probability
        total_rate = rate_matrix.sum(axis=1)  # [N]
        jump_prob = np.minimum(total_rate * dt, 0.9)
        jumped = np.random.rand(N) < jump_prob

        if jumped.any():
            # Categorical selection of target cell j
            denom = rate_matrix.sum(axis=1, keepdims=True)
            denom = np.maximum(denom, 1e-10)
            prob_matrix = rate_matrix / denom

            idx_i = np.where(jumped)[0]
            for i in idx_i:
                j = np.random.choice(N, p=prob_matrix[i])
                x_new[i] = x[j].copy()

        return x_new

    def reverse_step(
        self,
        x_t: np.ndarray,
        x_hat: np.ndarray,
        coords: np.ndarray,
        t_curr: float,
        dt: float,
    ) -> np.ndarray:
        """
        One reverse denoising step from t_curr → t_curr - dt.

        x_t:   [N, G] current noisy expression
        x_hat: [N, G] model prediction of clean expression
        coords: [N, 2] cell spatial positions (microns)
        t_curr: float current time
        dt:     float step size
        """
        N = x_t.shape[0]
        x_out = x_t.copy()

        total_steps = max(1, int(self.n_substeps * dt))
        dt_sub = dt / total_steps

        # Precompute spatial kernel for this spot (fixed)
        K = self._spatial_kernel(coords)

        for step in range(total_steps):
            t_eval = t_curr - (step + 0.5) * dt_sub
            if t_eval <= 0.001:
                continue

            rate_mult = self._time_scale(t_eval)
            mu = max(self.gamma_mut * self._integrated_rate(t_eval), 1e-4)

            # Importance weights (log pseudo-marginal for each cell)
            log_p = self._log_pseudo_marginal(x_out, x_hat, mu)

            # 1. Reverse mutation (guided by x_hat)
            x_out = self._reverse_mutation_step(x_out, x_hat, rate_mult, mu, dt_sub)

            # 2. Adjoint resampling
            if self.kappa > 0 and N > 1:
                # Recompute log_p after mutation
                log_p = self._log_pseudo_marginal(x_out, x_hat, mu)
                x_out = self._adjoint_resampling_step(
                    x_out, x_hat, K, log_p, rate_mult, dt_sub, N
                )

        return x_out


class MerfishFastAdjointReverse:
    """
    Vectorised (PyTorch/MPS-compatible) adjoint reverse sampler.

    Processes a batch of spots without Python loops over genes.
    The mutation step is approximated as a Gaussian displacement in
    the guidance direction, which is accurate for large G and is
    fully GPU-compatible.

    This is the recommended implementation for MPS acceleration.
    """

    def __init__(
        self,
        gamma_mut: float = 1.0,
        kappa: float = 0.5,
        h_kernel: float = 20.0,
        n_substeps: int = 10,
        guidance_weight: float = 2.0,
        device: str = "mps",
    ):
        self.gamma_mut = gamma_mut
        self.kappa = kappa
        self.h_kernel = h_kernel
        self.n_substeps = n_substeps
        self.guidance_weight = guidance_weight
        # MPS doesn't support float64; use cpu for numpy ops, mps for torch
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")

    def _time_scale(self, t: float) -> float:
        return 1.0 / max(1.0 - t, 0.001) ** 2

    def _integrated_rate(self, t: float) -> float:
        return 1.0 / max(1.0 - t, 0.001) - 1.0

    @torch.no_grad()
    def reverse_step(
        self,
        x_t: torch.Tensor,
        x_hat: torch.Tensor,
        coords: torch.Tensor,
        t_curr: float,
        dt: float,
    ) -> torch.Tensor:
        """
        Vectorised reverse step.

        x_t:    [N, G] float  current noisy expression
        x_hat:  [N, G] float  model prediction
        coords: [N, 2] float  cell positions (microns)
        """
        N, G = x_t.shape
        x_out = x_t.clone().float()
        x_hat = x_hat.float()

        total_steps = max(1, int(self.n_substeps * dt))
        dt_sub = dt / total_steps

        # Precompute spatial kernel [N, N] — no cdist needed on MPS
        # ||p_i - p_j||^2  via (a-b)^2 expansion
        diff = coords.float().unsqueeze(0) - coords.float().unsqueeze(1)   # [N, N, 2]
        dist_sq = (diff ** 2).sum(-1)                       # [N, N]
        K = torch.exp(-dist_sq / (2.0 * self.h_kernel ** 2))
        K.fill_diagonal_(0.0)

        for step in range(total_steps):
            t_eval = t_curr - (step + 0.5) * dt_sub
            if t_eval <= 0.001:
                break

            rate_mult = self._time_scale(t_eval)
            mu = max(self.gamma_mut * self._integrated_rate(t_eval), 1e-4)

            # ------------------------------------------------------------------
            # 1. Vectorised guided mutation
            # ------------------------------------------------------------------
            # displacement toward x_hat via Poisson birth/death
            diff_to_hat = x_hat - x_out           # [N, G]
            base_r = self.gamma_mut * rate_mult * dt_sub

            # Birth: move +1 toward x_hat where diff > 0
            log_r_birth = (-2.0 * diff_to_hat - 1.0) / (4.0 * mu)
            r_birth = base_r * torch.exp(
                torch.clamp(self.guidance_weight * log_r_birth, -5, 5)
            )
            born = torch.rand_like(x_out) < r_birth.clamp(0, 0.5)
            x_out = x_out + born.float()

            # Death: move -1 toward x_hat where diff < 0
            log_r_death = (2.0 * diff_to_hat - 1.0) / (4.0 * mu)
            r_death = base_r * torch.exp(
                torch.clamp(self.guidance_weight * log_r_death, -5, 5)
            )
            can_die = (x_out > 0)
            died = (torch.rand_like(x_out) < r_death.clamp(0, 0.5)) & can_die
            x_out = x_out - died.float()

            # ------------------------------------------------------------------
            # 2. Vectorised adjoint resampling
            # ------------------------------------------------------------------
            if self.kappa > 0 and N > 1:
                # Log pseudo-marginal: -||x - x_hat||^2 / (4*mu)  [N]
                log_p = -(x_out - x_hat).pow(2).sum(-1) / (4.0 * mu)

                # Importance ratio: log p_j - log p_i  [N, N]
                log_ratio = log_p.unsqueeze(0) - log_p.unsqueeze(1)
                importance = torch.exp(
                    torch.clamp(self.guidance_weight * log_ratio, -10, 10)
                )

                rate_matrix = (self.kappa * rate_mult / N) * K * importance  # [N, N]
                total_rate = rate_matrix.sum(-1)                               # [N]
                jump_prob = (total_rate * dt_sub).clamp(0, 0.9)

                jumped_mask = torch.rand(N, device=x_out.device) < jump_prob  # [N]

                if jumped_mask.any():
                    denom = rate_matrix.sum(-1, keepdim=True).clamp(min=1e-10)
                    prob_matrix = rate_matrix / denom                           # [N, N]

                    # Categorical sample for each jumping cell
                    x_snapshot = x_out.clone()
                    target_j = torch.multinomial(prob_matrix, 1).squeeze(-1)   # [N]
                    jumping_cells = jumped_mask.nonzero(as_tuple=True)[0]
                    x_out[jumping_cells] = x_snapshot[target_j[jumping_cells]]

            x_out = x_out.clamp(0, 500)

        return x_out
