"""
Moran Bridge (CuPy/GPU): Skellam birth-death bridge + spatial resampling.

GPU-accelerated version. Extends SkellamBridge with Moran resampling.
"""

import cupy as cp
import numpy as np
import torch
from typing import Callable
from .skellam import SkellamBridge
from .utils import dlpack_backend


class MoranBridge(SkellamBridge):
    """
    GPU Skellam bridge with Moran spatial resampling.
    """

    def __init__(
        self,
        slack_sampler: Callable,
        kappa: float = 2.0,
        bandwidth: float = 5.0,
        resample_frac: float = 0.3,
        delta: bool = False,
        schedule_type: str = "linear",
        homogeneous_time: bool = False,
        backend: str = "torch",
        device: int = 0,
        **schedule_kwargs,
    ):
        super().__init__(
            slack_sampler=slack_sampler,
            delta=delta,
            schedule_type=schedule_type,
            homogeneous_time=homogeneous_time,
            backend=backend,
            device=device,
            **schedule_kwargs,
        )
        self.kappa = kappa
        self.bandwidth = bandwidth
        self.resample_frac = resample_frac

    def _spatial_resample_cp(self, x: cp.ndarray, t_val: float) -> cp.ndarray:
        """Moran resampling on GPU using CuPy."""
        if self.kappa <= 0:
            return x

        B = x.shape[0]
        x_out = x.copy()

        resample_prob = min(self.kappa * t_val, self.resample_frac)
        do_resample = cp.random.rand(B) < resample_prob
        n_resample = int(do_resample.sum())

        if n_resample == 0:
            return x_out

        partners = cp.random.randint(0, B, size=B)
        diff = x[do_resample].astype(cp.float32) - x[partners[do_resample]].astype(cp.float32)
        dist_sq = cp.sum(diff ** 2, axis=-1)
        K = cp.exp(-dist_sq / (2 * self.bandwidth ** 2))

        copy_mask = cp.random.rand(n_resample) < K
        if copy_mask.any():
            resample_indices = cp.where(do_resample)[0][copy_mask]
            partner_indices = partners[do_resample][copy_mask]
            x_out[resample_indices] = x[partner_indices]

        return x_out

    def __call__(self, x_0, x_1, t=None):
        """Forward: Skellam bridge + Moran resampling."""
        with cp.cuda.Device(self.device):
            # Run standard Skellam forward
            t_out, x_t, target = super().__call__(x_0, x_1, t=t)

            # Apply Moran resampling on the CuPy arrays
            x_t_cp = cp.from_dlpack(x_t) if not isinstance(x_t, cp.ndarray) else x_t
            t_cp = cp.from_dlpack(t_out) if not isinstance(t_out, cp.ndarray) else t_out
            t_mean = float(cp.mean(t_cp))

            x_t_resampled = self._spatial_resample_cp(
                x_t_cp.round().astype(cp.int32), t_mean
            )

            x_t_resampled = dlpack_backend(x_t_resampled, backend=self.backend, dtype="float32")

            if self.delta:
                target_cp = cp.from_dlpack(target) if not isinstance(target, cp.ndarray) else target
                x_0_reconstructed = target_cp + x_t_cp
                new_target = x_0_reconstructed - cp.from_dlpack(x_t_resampled)
                target = dlpack_backend(new_target, backend=self.backend, dtype="float32")

            return t_out, x_t_resampled, target
