"""
Moran Bridge — Regime A: Particles hopping on a 2D integer grid.

Forward CTMC with two interleaved components:
  1. Mutation: random walk on the grid (hop to a neighbouring cell)
  2. Resampling: particle i copies particle j's position, weighted by
     spatial kernel K(x_i, x_j)

The type space is E = {0, ..., g_max}^2 (joint 2D grid cells).
Mutation to uniform neighbours gives a proper stationary distribution
(uniform over grid), so the de Finetti theorem applies:

  Empirical measure -> DP(theta, Uniform(E))

For finite |E| = G = (g_max+1)^2, this is:
  cell frequencies ~ Dir(theta/G, ..., theta/G)
  particles iid from Categorical(frequencies)

The reverse process uses pure iterative denoising:
  1. Start from de Finetti prior
  2. Model predicts x_0 from (x_t, t)
  3. Re-noise x_0 to t_{k-1} via the Moran forward (tau-leaping)
  4. At t=0 output prediction directly

No bridge kernels anywhere.
"""

import numpy as np
import torch
from typing import Callable, Optional
from .utils import dlpack_backend
from .scheduling import make_weight_schedule


class MoranBridge:
    """
    Regime A Moran model: N particles on a 2D integer grid.

    Parameters:
        n_steps: number of discrete time steps
        slack_sampler: kept for interface compatibility (unused in forward/reverse)
        kappa: resampling rate
        bandwidth: RBF kernel bandwidth for spatial similarity
        theta: mutation-drift ratio 2*gamma_mut/kappa (controls de Finetti prior)
        n_substeps: tau-leaping substeps per unit time
        gamma_mut: mutation rate (hops per unit time)
        grid_max: maximum grid coordinate (type space is {0,...,grid_max}^2)
    """

    def __init__(
        self,
        n_steps: int,
        slack_sampler: Callable,
        kappa: float = 2.0,
        bandwidth: float = 5.0,
        theta: float = 1.0,
        n_substeps: int = 20,
        gamma_mut: float = 1.0,
        grid_max: int = 127,
        delta: bool = False,
        schedule_type: str = "linear",
        homogeneous_time: bool = False,
        backend: str = "torch",
        device: int = 0,
        **schedule_kwargs,
    ):
        self.n_steps = n_steps
        self.slack_sampler = slack_sampler  # interface compat
        self.kappa = kappa
        self.bandwidth = bandwidth
        self.theta = theta
        self.n_substeps = n_substeps
        self.gamma_mut = gamma_mut
        self.grid_max = grid_max
        self.delta = delta
        self.backend = backend
        self.device = device

        self.time_points = np.linspace(0, 1, n_steps + 1)
        self.weights = make_weight_schedule(n_steps, schedule_type, **schedule_kwargs)

    # ------------------------------------------------------------------
    # Spatial kernel
    # ------------------------------------------------------------------

    def _kernel(self, x: np.ndarray) -> np.ndarray:
        """Pairwise RBF kernel. x: [B, 2] int -> [B, B] float."""
        x_f = x.astype(np.float64)
        diff = x_f[:, None, :] - x_f[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.exp(-dist_sq / (2 * self.bandwidth ** 2))

    # ------------------------------------------------------------------
    # Forward: tau-leaping of joint mutation + resampling CTMC
    # ------------------------------------------------------------------

    def _time_scale(self, t: float) -> float:
        """
        Time rescaling: s = 1/(1-t) so that physical time s -> inf as t -> 1.

        The rate multiplier at time t is ds/dt = 1/(1-t)^2.
        Both mutation and resampling rates scale by this factor,
        preserving theta = 2*gamma_mut/kappa at all times.

        At t=0: multiplier = 1 (base rates).
        At t=1: multiplier = inf (process at de Finetti equilibrium).
        """
        # Clamp to avoid division by zero
        return 1.0 / max(1.0 - t, 0.01) ** 2

    def _simulate_forward(self, x_0: np.ndarray, t_target: float) -> np.ndarray:
        """
        Simulate the Moran CTMC from x_0 to time t_target.

        Rates are time-scaled: gamma_mut(t) = gamma_mut / (1-t)^2,
        kappa(t) = kappa / (1-t)^2. This ensures the process reaches
        the de Finetti equilibrium at t=1 (total physical time = infinity).

        At each substep, interleaved:
          1. Mutation: random walk on {0,...,grid_max}^2 (hop to neighbour)
          2. Resampling: particle i copies particle j weighted by K(x_i, x_j)

        x_0: [B, 2] integer positions
        returns: [B, 2] integer positions at time t_target
        """
        B, D = x_0.shape
        x = x_0.copy()

        total_substeps = max(1, int(self.n_substeps * t_target))
        dt = t_target / total_substeps

        for step in range(total_substeps):
            t_current = (step + 0.5) * dt  # midpoint of substep
            rate_mult = self._time_scale(t_current)

            # --- Mutation: random walk on grid (time-scaled) ---
            # Expected hops this substep: gamma_mut * rate_mult * dt
            # Can be > 1, so use Poisson for number of hops per particle
            n_hops_per_particle = np.random.poisson(
                self.gamma_mut * rate_mult * dt, size=B
            )
            for i in range(B):
                for _ in range(n_hops_per_particle[i]):
                    direction = np.random.randint(0, 4)
                    delta = np.array([0, 0], dtype=np.int32)
                    if direction == 0: delta[1] = 1    # up
                    elif direction == 1: delta[1] = -1  # down
                    elif direction == 2: delta[0] = -1  # left
                    else: delta[0] = 1                  # right
                    x[i] = np.clip(x[i] + delta, 0, self.grid_max)

            # --- Resampling: Moran kernel-weighted (time-scaled) ---
            if self.kappa > 0 and B > 1:
                K = self._kernel(x)
                np.fill_diagonal(K, 0)

                for i in range(B):
                    rate_i = self.kappa * rate_mult / B * K[i].sum() * dt
                    if np.random.rand() < min(rate_i, 0.8):
                        w = K[i]
                        w_sum = w.sum()
                        if w_sum > 0:
                            j = np.random.choice(B, p=w / w_sum)
                            x[i] = x[j].copy()

        return x

    # ------------------------------------------------------------------
    # Forward interface (for training)
    # ------------------------------------------------------------------

    def __call__(self, x_0, x_1, t_target=None):
        """
        Produce training tuples (t, x_t, target) by simulating Moran forward.

        x_0: target data (two moons)
        x_1: source (ignored — Moran doesn't use a source distribution)
        """
        x_0_np, _ = dlpack_backend(x_0, x_1, backend='numpy', dtype="int32")
        B, D = x_0_np.shape

        # Sample time
        if t_target is not None:
            time_diffs = np.abs(self.time_points - t_target)
            k = np.argmin(time_diffs)
            k = np.broadcast_to(k, (B,))
        else:
            k = np.random.randint(1, self.n_steps + 1, (B,))

        t = self.time_points[k].reshape(-1, 1)

        # Simulate forward from x_0 to time t
        # Use max t for the batch (all particles share the same simulation)
        t_max = float(np.max(t))
        x_t = self._simulate_forward(x_0_np, t_max)

        target = x_0_np  # denoiser predicts x_0

        x_t, t, target = dlpack_backend(
            x_t, t, target, backend=self.backend, dtype="float32"
        )

        return {"inputs": {"x_t": x_t, "t": t}, "output": target}

    # ------------------------------------------------------------------
    # De Finetti prior: Dir(theta/G) over joint grid cells
    # ------------------------------------------------------------------

    def sample_prior(self, n_samples: int, data_dim: int = 2,
                     value_range: int = None) -> np.ndarray:
        """
        Sample from the de Finetti equilibrium of the Moran model.

        Type space: E = {0, ..., g_max}^2, |E| = G = (g_max+1)^2.
        Mutation to uniform neighbours -> stationary = Uniform(E).
        De Finetti: frequencies ~ Dir(theta/G, ..., theta/G),
                    then N particles iid from Categorical(frequencies).

        Returns: [n_samples, 2] integer array of grid positions.
        """
        g = self.grid_max + 1
        G = g * g  # total number of grid cells
        alpha = max(self.theta / G, 1e-8)

        # Sample cell frequencies from Dirichlet over the JOINT grid
        p = np.random.dirichlet(np.full(G, alpha))

        # Sample n_samples cell indices from Categorical(p)
        cell_indices = np.random.choice(G, size=n_samples, p=p)

        # Convert flat index to (x, y) grid coordinates
        result = np.zeros((n_samples, 2), dtype=np.int32)
        result[:, 0] = cell_indices // g
        result[:, 1] = cell_indices % g

        return result

    # ------------------------------------------------------------------
    # Reverse: iterative denoising via Moran forward re-noising
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
        value_range: int = None,
    ):
        """
        Reverse: de Finetti prior -> iterative denoising -> data.

        At each step:
          1. Model predicts x_0_hat from (x_t, t)
          2. Re-noise x_0_hat to t_{k-1} using the Moran forward (tau-leaping)
          3. At t=0 output x_0_hat directly

        No bridge kernels. Just forward simulation + learned denoiser.
        """
        B, D = x_1.shape

        # Start from de Finetti prior
        x_t = self.sample_prior(B, D).astype(np.int32)

        traj, xhat_traj = [x_t.copy()], []

        for k in range(self.n_steps, 0, -1):
            t_curr = self.time_points[k]
            t_next = self.time_points[k - 1]
            t = np.broadcast_to(t_curr, (B, 1))

            # Model predicts x_0
            x_t_dl, t_dl = dlpack_backend(x_t, t, backend=self.backend, dtype="float32")
            model_out = model.sample(x_t=x_t_dl, t=t_dl, **z)
            x0_hat = dlpack_backend(model_out, backend='numpy', dtype="float32")
            x0_hat = np.clip(x0_hat.round().astype(np.int32), 0, self.grid_max)

            if guidance_x_0 is not None:
                gs = guidance_schedule[k] if guidance_schedule is not None else 0
                x0_hat = np.clip(
                    (gs * guidance_x_0 + (1 - gs) * x0_hat).round().astype(np.int32),
                    0, self.grid_max
                )

            # Re-noise via Moran forward to t_next
            if t_next > 1e-6:
                x_t = self._simulate_forward(x0_hat, t_next)
            else:
                x_t = x0_hat

            if return_trajectory:
                traj.append(x_t.copy())
            if return_x_hat:
                xhat_traj.append(x0_hat.copy())

        outs = [x_t]
        if return_trajectory:
            outs.append(np.stack(traj))
        if return_x_hat:
            outs.append(np.stack(xhat_traj))

        return dlpack_backend(*outs, backend=self.backend, dtype="float32")
