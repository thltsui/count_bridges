"""
Regime B: Pure iterative denoising for grid-count Moran model.

No Skellam bridge steps. No Lie-Trotter splitting. The forward and
reverse are both based on the Moran CTMC directly:

Forward (training):
  Simulate Moran CTMC from c_0 to time t via tau-leaping:
    - Poisson birth-death per cell (mutation)
    - Cell i copies cell j's count (resampling)
  Both interleaved at each substep.

Reverse (generation):
  Start from de Finetti prior or source distribution.
  At each step:
    1. Model predicts c_0_hat from (c_t, t)
    2. Re-noise c_0_hat to t_{k-1} via the Moran forward
    3. At t=0, output c_0_hat directly

Same approach as Regime A — just different state space and mutation type.
"""

import numpy as np
import torch
from bridges.numpy.utils import dlpack_backend
from bridges.numpy.scheduling import make_weight_schedule
from .forward import GridMoranForward


class GridMoranIterative:
    """
    Regime B with pure iterative denoising — no bridge kernels.

    Parameters:
        grid_forward: GridMoranForward instance
        n_steps: number of reverse denoising steps
        lambda_rate: Poisson birth-death rate
        n_substeps: tau-leaping substeps per unit time
    """

    def __init__(
        self,
        grid_forward: GridMoranForward,
        n_steps: int = 20,
        lambda_rate: float = 2.0,
        n_substeps: int = 20,
        backend: str = "torch",
    ):
        self.grid = grid_forward
        self.n_steps = n_steps
        self.lambda_rate = lambda_rate
        self.n_substeps = n_substeps
        self.backend = backend
        self.time_points = np.linspace(0, 1, n_steps + 1)

    def _time_scale(self, t: float) -> float:
        """Rate multiplier 1/(1-t)^2 for convergence to stationarity."""
        return 1.0 / max(1.0 - t, 0.01) ** 2

    def _simulate_forward(self, c_0: np.ndarray, t_target: float) -> np.ndarray:
        """
        Simulate Moran CTMC on count vectors from t=0 to t=t_target.

        At each substep, interleaved:
          1. Poisson birth-death per cell (mutation)
          2. Moran resampling between cells (coupling)

        c_0: [B, G] integer count vectors (B batch, G cells)
        returns: [B, G] count vectors at time t_target
        """
        B, G = c_0.shape
        c = c_0.copy()

        total_steps = max(1, int(self.n_substeps * t_target))
        dt = t_target / total_steps

        for step in range(total_steps):
            t_mid = (step + 0.5) * dt
            rate_mult = self._time_scale(t_mid)

            # --- Mutation: Poisson birth-death per cell ---
            lam = self.lambda_rate * rate_mult * dt
            births = np.random.poisson(lam, size=(B, G))
            deaths = np.random.poisson(lam, size=(B, G))
            c = np.maximum(c + births - deaths, 0).astype(np.int32)

            # --- Resampling: cell i copies cell j's count ---
            if self.grid.kappa > 0:
                for b in range(B):
                    c[b] = self.grid.resample_counts(
                        c[b], rate=self.grid.kappa * rate_mult * dt
                    )

        return c

    def __call__(self, x_0, x_1, t_target=None):
        """
        Forward: simulate Moran CTMC from c_0 to produce c_t.

        x_0: [B, G] target count vectors
        x_1: [B, G] source count vectors (ignored — Moran doesn't bridge)
        returns: dict {"inputs": {"x_t", "t"}, "output": target}
        """
        c_0 = dlpack_backend(x_0, backend='numpy', dtype="int32")
        if isinstance(c_0, tuple):
            c_0 = c_0[0]
        B = c_0.shape[0]

        # Sample time
        if t_target is not None:
            k = np.full(B, int(round(t_target * self.n_steps)))
        else:
            k = np.random.randint(1, self.n_steps + 1, (B,))

        t = self.time_points[k].reshape(-1, 1)

        # Simulate forward
        t_max = float(np.max(t))
        c_t = self._simulate_forward(c_0, t_max)

        target = c_0
        c_t_out, t_out, target_out = dlpack_backend(
            c_t, t, target, backend=self.backend, dtype="float32"
        )

        return {
            "inputs": {"x_t": c_t_out, "t": t_out},
            "output": target_out,
        }

    def sampler(
        self,
        x_1: np.ndarray,
        z: dict,
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        **kwargs,
    ):
        """
        Reverse: pure iterative denoising.

        Start from source distribution (x_1), then:
          1. Model predicts c_0_hat from (c_t, t)
          2. Re-noise c_0_hat to t_{k-1} via Moran forward
          3. At t=0, output c_0_hat directly
        """
        c_t = dlpack_backend(x_1, backend='numpy', dtype="int32")
        if isinstance(c_t, tuple):
            c_t = c_t[0]
        c_t = c_t.round().astype(np.int32)
        B = c_t.shape[0]

        traj, xhat_traj = [c_t.copy()], []

        for k in range(self.n_steps, 0, -1):
            t_curr = self.time_points[k]
            t_next = self.time_points[k - 1]
            t = np.broadcast_to(t_curr, (B, 1))

            # Model predicts c_0
            c_t_dl, t_dl = dlpack_backend(
                c_t, t, backend=self.backend, dtype="float32"
            )
            model_out = model.sample(x_t=c_t_dl, t=t_dl, **z)
            c0_hat = dlpack_backend(model_out, backend='numpy', dtype="float32")
            if isinstance(c0_hat, tuple):
                c0_hat = c0_hat[0]
            c0_hat = np.maximum(c0_hat.round().astype(np.int32), 0)

            # Re-noise via Moran forward to t_next
            if t_next > 1e-6:
                c_t = self._simulate_forward(
                    c0_hat.reshape(B, -1), t_next
                )
            else:
                c_t = c0_hat

            if return_trajectory:
                traj.append(c_t.copy())
            if return_x_hat:
                xhat_traj.append(c0_hat.copy())

        outs = [c_t]
        if return_trajectory:
            outs.append(np.stack(traj))
        if return_x_hat:
            outs.append(np.stack(xhat_traj))

        return dlpack_backend(*outs, backend=self.backend, dtype="float32")
