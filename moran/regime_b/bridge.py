"""
Regime B Bridge: Skellam per cell + Moran resampling between cells.

Uses the Count Bridge Skellam machinery for per-cell birth-death (exact
conditionals), then applies Moran resampling to couple cells spatially.

Training forward:
  1. Skellam bridge per cell: sample c_t given (c_0, c_1, t)
  2. Moran resampling: cell i copies cell j's count with rate ~ K(i,j)

Reverse step:
  1. Model predicts c_0_hat from (c_t, t)
  2. Skellam bridge step per cell: sample c_{t-dt} from exact conditional
  3. Moran resampling: couple cells spatially

The Lie-Trotter splitting (bridge step then resampling) is first-order
accurate and preserves the exact integer dynamics of Count Bridges.
"""

import numpy as np
import torch
from typing import Callable
from bridges.numpy.skellam import SkellamBridge
from bridges.numpy.utils import dlpack_backend
from .forward import GridMoranForward


class GridMoranBridge:
    """
    Regime B bridge: Skellam per cell + Moran resampling.

    Wraps a SkellamBridge for per-cell dynamics and adds inter-cell
    Moran resampling as a separate layer.

    Parameters:
        skellam: SkellamBridge instance (handles per-cell birth-death)
        grid_forward: GridMoranForward instance (handles resampling)
        kappa: resampling rate (0 = pure Count Bridge)
    """

    def __init__(
        self,
        skellam: SkellamBridge,
        grid_forward: GridMoranForward,
        kappa: float = 2.0,
    ):
        self.skellam = skellam
        self.grid = grid_forward
        self.kappa = kappa
        # Expose interface attributes from skellam
        self.n_steps = skellam.n_steps
        self.time_points = skellam.time_points
        self.weights = skellam.weights
        self.delta = skellam.delta
        self.backend = skellam.backend
        self.slack_sampler = skellam.slack_sampler

    def __call__(self, x_0, x_1, t_target=None):
        """
        Training forward: Skellam bridge per cell, then Moran resampling.

        x_0: [B, G] target count vectors (one count per cell)
        x_1: [B, G] source count vectors
        returns: dict with x_t (noised counts) and target (clean counts)
        """
        # Step 1: Skellam bridge per cell (exact conditional sampling)
        out = self.skellam(x_0, x_1, t_target=t_target)

        if self.kappa <= 0:
            return out

        # Step 2: Moran resampling on the noised count vector
        x_t = dlpack_backend(out["inputs"]["x_t"], backend='numpy', dtype="float32")
        t = dlpack_backend(out["inputs"]["t"], backend='numpy', dtype="float32")
        t_mean = float(np.mean(t))

        # Apply resampling to each sample in the batch
        x_t_int = x_t.round().astype(np.int32)
        for b in range(x_t_int.shape[0]):
            x_t_int[b] = self.grid.resample_counts(
                x_t_int[b], rate=self.kappa * t_mean
            )

        # Convert back
        x_t_resampled = dlpack_backend(
            x_t_int, backend=self.backend, dtype="float32"
        )
        if isinstance(x_t_resampled, tuple):
            x_t_resampled = x_t_resampled[0]
        out["inputs"]["x_t"] = x_t_resampled

        return out

    def sampler(
        self,
        x_1: np.ndarray,
        z: dict,
        model,
        return_trajectory: bool = False,
        return_x_hat: bool = False,
        return_M: bool = False,
        **kwargs,
    ):
        """
        Reverse: Skellam bridge step per cell + Moran resampling.

        At each reverse step k:
          1. Model predicts c_0_hat from (c_t, t)
          2. Skellam bridge step: exact per-cell reverse (Binomial + Hypergeometric)
          3. Moran resampling: couple cells spatially

        This is Lie-Trotter splitting: the exact bridge handles mutation,
        the resampling handles coupling.
        """
        B, G = x_1.shape
        x_t = dlpack_backend(x_1.round(), backend='numpy', dtype="int32")

        traj, xhat_traj = [x_t.copy()], []

        for k in range(self.skellam.n_steps, 0, -1):
            t = np.broadcast_to(self.skellam.time_points[k], (B, 1))

            # Model prediction
            x_t_dl, t_dl = dlpack_backend(
                x_t, t, backend=self.backend, dtype="float32"
            )
            model_out = model.sample(x_t=x_t_dl, t=t_dl, **z)
            x0_hat = dlpack_backend(model_out, backend='numpy', dtype="float32")
            x0_hat = np.maximum(x0_hat.round().astype(np.int32), 0)

            # Step 1: Skellam bridge step per cell (exact reverse)
            diff = x_t - x0_hat
            M_t = self.slack_sampler(diff)
            N_t = np.abs(diff) + 2 * M_t
            B_t = (N_t + diff) // 2

            rho = self.weights[k - 1] / self.weights[k] if self.weights[k] > 0 else 0
            non_zero = N_t > 0
            N_s = np.zeros_like(N_t)
            N_s[non_zero] = np.random.binomial(N_t[non_zero], rho)

            non_zero_s = N_s > 0
            B_s = np.zeros_like(B_t)
            B_s[non_zero_s] = np.random.hypergeometric(
                ngood=B_t[non_zero_s],
                nbad=N_t[non_zero_s] - B_t[non_zero_s],
                nsample=N_s[non_zero_s],
            )
            x_s = x_t - 2 * (B_t - B_s) + (N_t - N_s)

            # Step 2: Moran resampling (spatial coupling)
            if self.kappa > 0:
                t_next = self.skellam.time_points[k - 1]
                for b in range(B):
                    x_s[b] = self.grid.resample_counts(
                        x_s[b], rate=self.kappa * t_next
                    )

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

        return dlpack_backend(*outs, backend=self.backend, dtype="float32")
