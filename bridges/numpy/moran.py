"""
Moran Bridge: Joint mutation + resampling CTMC, simulated via tau-leaping.

Unlike the Skellam bridge which has exact bridge conditionals, the Moran
model must be SIMULATED step-by-step because resampling couples particles
— a resampling event at time s < t changes the state, affecting all
subsequent mutation. The generator acts simultaneously:

    Af = sum_i A_mut^(i) f  +  (kappa/N) sum_{i!=j} K(x_i, x_j) [f(x^{i<-j}) - f(x)]

Both terms fire continuously. We approximate via tau-leaping: subdivide
[0, t] into substeps, at each substep apply mutation (Poisson increments)
then resampling (kernel-weighted copying).

The reverse sampler starts from the Poisson-Dirichlet / Dirichlet-Multinomial
de Finetti equilibrium — the stationary distribution of the forward process.
"""

import numpy as np
import torch
from typing import Callable, Optional
from .utils import dlpack_backend
from .scheduling import make_weight_schedule


class MoranBridge:
    """
    Moran bridge with interleaved mutation and resampling.

    Forward: tau-leaping simulation of joint CTMC from x_0.
    Reverse: iterative denoising from Poisson-Dirichlet prior.

    Parameters:
        n_steps: number of bridge time steps (for weight schedule + reverse)
        slack_sampler: Bessel slack sampler (for reverse step compatibility)
        kappa: resampling rate
        bandwidth: RBF kernel bandwidth
        theta: mutation-drift ratio 2*lambda/kappa (for de Finetti prior)
        n_substeps: tau-leaping substeps per unit time in forward simulation
        lambda_rate: base Poisson birth-death rate per substep
    """

    def __init__(
        self,
        n_steps: int,
        slack_sampler: Callable,
        kappa: float = 2.0,
        bandwidth: float = 5.0,
        theta: float = 1.0,
        n_substeps: int = 20,
        lambda_rate: float = 1.0,
        delta: bool = False,
        schedule_type: str = "linear",
        homogeneous_time: bool = False,
        backend: str = "torch",
        device: int = 0,
        **schedule_kwargs,
    ):
        self.n_steps = n_steps
        self.slack_sampler = slack_sampler
        self.kappa = kappa
        self.bandwidth = bandwidth
        self.theta = theta
        self.n_substeps = n_substeps
        self.lambda_rate = lambda_rate
        self.delta = delta
        self.schedule_type = schedule_type
        self.homogeneous_time = homogeneous_time
        self.backend = backend
        self.device = device

        self.time_points = np.linspace(0, 1, n_steps + 1)
        self.weights = make_weight_schedule(n_steps, schedule_type, **schedule_kwargs)

    # ------------------------------------------------------------------
    # Spatial kernel
    # ------------------------------------------------------------------

    def _kernel(self, x: np.ndarray) -> np.ndarray:
        """
        Pairwise RBF kernel K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2h^2)).
        x: [B, D] integer array
        returns: [B, B] kernel matrix
        """
        x_f = x.astype(np.float64)
        diff = x_f[:, None, :] - x_f[None, :, :]  # [B, B, D]
        dist_sq = np.sum(diff ** 2, axis=-1)        # [B, B]
        return np.exp(-dist_sq / (2 * self.bandwidth ** 2))

    # ------------------------------------------------------------------
    # Forward: tau-leaping simulation of joint Moran CTMC
    # ------------------------------------------------------------------

    def _simulate_forward(self, x_0: np.ndarray, t_target: float) -> np.ndarray:
        """
        Simulate the Moran CTMC from x_0 at time 0 to time t_target.

        At each substep dt:
          1. Mutation: Poisson birth-death per coordinate
             x_i += Poi(lambda * dt) - Poi(lambda * dt), clamped >= 0
          2. Resampling: for each particle i, with probability
             kappa * dt * mean_j(K(x_i, x_j)), copy from a random
             kernel-weighted neighbour j.

        x_0: [B, D] integer array
        t_target: float in (0, 1]
        returns: [B, D] integer array at time t_target
        """
        B, D = x_0.shape
        x = x_0.copy()

        total_substeps = max(1, int(self.n_substeps * t_target))
        dt = t_target / total_substeps

        for _ in range(total_substeps):
            # --- Mutation: Poisson birth-death ---
            births = np.random.poisson(self.lambda_rate * dt, size=(B, D))
            deaths = np.random.poisson(self.lambda_rate * dt, size=(B, D))
            x = np.maximum(x + births - deaths, 0).astype(np.int32)

            # --- Resampling: Moran kernel-weighted copying ---
            if self.kappa > 0 and B > 1:
                K = self._kernel(x)  # [B, B]
                np.fill_diagonal(K, 0)  # no self-copying

                for i in range(B):
                    # Rate of resampling for particle i
                    rate_i = self.kappa * dt * K[i].mean()
                    if np.random.rand() < rate_i:
                        # Choose partner j proportional to K(x_i, x_j)
                        weights = K[i]
                        w_sum = weights.sum()
                        if w_sum > 0:
                            probs = weights / w_sum
                            j = np.random.choice(B, p=probs)
                            x[i] = x[j].copy()

        return x

    def __call__(self, x_0, x_1, t_target=None):
        """
        Forward process: simulate Moran CTMC from x_0 to produce x_t.

        Unlike Skellam which uses x_1 and exact bridge conditionals,
        the Moran bridge ignores x_1 and simulates the forward CTMC
        from x_0 directly. The noise comes from the interleaved
        mutation + resampling, not from a bridge to a source distribution.

        Returns dict matching the numpy Skellam interface:
            {"inputs": {"x_t": tensor, "t": tensor}, "output": tensor}
        """
        x_0_np, x_1_np = dlpack_backend(x_0, x_1, backend='numpy', dtype="int32")
        B, D = x_0_np.shape

        # Sample time
        if t_target is not None:
            time_diffs = np.abs(self.time_points - t_target)
            k = np.argmin(time_diffs)
            k = np.broadcast_to(k, (B,))
        elif self.homogeneous_time:
            k = np.random.randint(1, self.n_steps + 1)
            k = np.broadcast_to(k, (B,))
        else:
            k = np.random.randint(1, self.n_steps + 1, (B,))

        t = self.time_points[k].reshape(-1, 1)

        # Simulate forward for each sample at its own time
        # For efficiency, use the max time and subsample
        t_max = float(np.max(t))
        x_t = self._simulate_forward(x_0_np, t_max)

        # Convert to output format
        target = x_0_np  # denoiser predicts x_0 from x_t
        x_t, t, target = dlpack_backend(
            x_t, t, target, backend=self.backend, dtype="float32"
        )

        return {
            "inputs": {"x_t": x_t, "t": t},
            "output": target,
        }

    # ------------------------------------------------------------------
    # De Finetti prior: Poisson-Dirichlet / Dirichlet-Multinomial
    # ------------------------------------------------------------------

    def sample_prior(self, n_samples: int, data_dim: int, value_range: int) -> np.ndarray:
        """
        Sample from the de Finetti equilibrium of the Moran model.

        For finite type space E = {0, ..., value_range-1}^D:
          1. Sample p ~ Dir(theta/G, ..., theta/G) where G = value_range^D
          2. Sample N points: each coordinate sampled from Categorical(p_marginal)

        For the 2D discrete moons case, we sample each coordinate independently
        from a Dirichlet-Multinomial with concentration theta/value_range.

        n_samples: number of points to generate
        data_dim: dimension (2 for moons)
        value_range: max integer value per coordinate
        returns: [n_samples, data_dim] integer array
        """
        alpha = max(self.theta / value_range, 1e-6)
        alphas = np.full(value_range, alpha)

        result = np.zeros((n_samples, data_dim), dtype=np.int32)
        for d in range(data_dim):
            # Sample probability vector from Dirichlet
            p = np.random.dirichlet(alphas)
            # Sample coordinates from Categorical
            result[:, d] = np.random.choice(value_range, size=n_samples, p=p)

        return result

    # ------------------------------------------------------------------
    # Reverse sampler
    # ------------------------------------------------------------------

    def sampler(
        self,
        x_1: np.ndarray,
        z: dict,
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        return_M: bool = False,
        guidance_x_0: np.ndarray = None,
        guidance_schedule: np.ndarray = None,
        value_range: int = 128,
    ):
        """
        Reverse sampling from the Poisson-Dirichlet prior.

        Instead of starting from x_1 (source distribution), we start from
        the de Finetti equilibrium — the stationary distribution of the
        Moran forward process.

        Then iteratively denoise using the trained model, applying
        Skellam-style bridge steps for the mutation component.
        """
        B, D = x_1.shape

        # --- Sample from de Finetti prior instead of using x_1 ---
        x_t = self.sample_prior(B, D, value_range).astype(np.int32)

        if guidance_x_0 is not None:
            guidance_x_0 = guidance_x_0.round().astype(np.int32)

        traj, xhat_traj = [x_t.copy()], []

        for k in range(self.n_steps, 0, -1):
            t = np.broadcast_to(self.time_points[k], (B, 1))

            # Get model prediction of x_0
            x_t_dl, t_dl = dlpack_backend(
                x_t, t, backend=self.backend, dtype="float32"
            )
            model_out = model.sample(x_t=x_t_dl, t=t_dl, **z)
            x0_hat = dlpack_backend(model_out, backend='numpy', dtype="float32")
            x0_hat = x0_hat.round().astype(np.int32)

            if guidance_x_0 is not None:
                gs = guidance_schedule[k] if guidance_schedule is not None else 0
                x0_hat = (gs * guidance_x_0 + (1 - gs) * x0_hat).round().astype(np.int32)

            # --- Reverse step: simulate backward Moran ---
            # Use Skellam bridge step for the mutation component
            diff = x_t - x0_hat
            M_t = self.slack_sampler(diff)

            N_t = np.abs(diff) + 2 * M_t
            B_t = (N_t + diff) // 2

            # Weight ratio for reverse step
            rho = self.weights[k - 1] / self.weights[k] if self.weights[k] > 0 else 0

            non_zero = N_t > 0
            N_s = np.zeros_like(N_t)
            N_s[non_zero] = np.random.binomial(N_t[non_zero], rho)

            non_zero_s = N_s > 0
            B_s = np.zeros_like(B_t)
            B_s[non_zero_s] = np.random.hypergeometric(
                ngood=B_t[non_zero_s],
                nbad=N_t[non_zero_s] - B_t[non_zero_s],
                nsample=N_s[non_zero_s]
            )

            x_s = x_t - 2 * (B_t - B_s) + (N_t - N_s)
            x_t = x_s

            if return_trajectory:
                traj.append(x_t.copy())
            if return_x_hat:
                xhat_traj.append(x0_hat.copy())

        outs = [x_t]
        if return_trajectory:
            outs.append(np.stack(traj))
        if return_x_hat:
            outs.append(np.stack(xhat_traj))

        return dlpack_backend(
            *outs, backend=self.backend, dtype="float32"
        )
