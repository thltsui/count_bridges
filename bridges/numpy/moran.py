"""
Moran Bridge: Skellam birth-death bridge + spatial resampling.

Extends the Count Bridge with Moran resampling from population genetics.
At kappa=0, reduces to the standard SkellamBridge.
At finite kappa, nearby positions (by kernel K) copy each other's values,
creating spatial correlation absent from the plain Count Bridge.

The forward process has two components per step:
  1. Mutation: Poisson birth-death (inherited from SkellamBridge)
  2. Resampling: position i copies position j's value with rate
     kappa/N * K(pos_i, pos_j), where K is a spatial RBF kernel.
"""

import numpy as np
from typing import Callable
from .skellam import SkellamBridge
from .utils import dlpack_backend


class MoranBridge(SkellamBridge):
    """
    Skellam bridge with Moran spatial resampling.

    Additional parameters:
        kappa: resampling rate (0 = no resampling = plain SkellamBridge)
        bandwidth: RBF kernel bandwidth for spatial similarity
        resample_frac: max fraction of points resampled per step
    """

    def __init__(
        self,
        n_steps: int,
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
            n_steps=n_steps,
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

    def _spatial_resample(self, x: np.ndarray, t_val: float) -> np.ndarray:
        """
        Apply Moran resampling: point i copies point j's value
        with probability proportional to kappa * t * K(x_i, x_j) / N.

        x: [B, D] integer array — current positions
        t_val: float — current time (resampling strength increases with t)
        returns: [B, D] integer array — resampled positions
        """
        if self.kappa <= 0:
            return x

        B = x.shape[0]
        x_out = x.copy()

        # Probability of resampling for each point (increases with time)
        resample_prob = min(self.kappa * t_val, self.resample_frac)

        # Which points get resampled
        do_resample = np.random.rand(B) < resample_prob
        n_resample = do_resample.sum()

        if n_resample == 0:
            return x_out

        # For each resampled point, pick a random partner
        partners = np.random.randint(0, B, size=B)

        # Compute spatial kernel K between each point and its partner
        diff = x[do_resample].astype(np.float64) - x[partners[do_resample]].astype(np.float64)
        dist_sq = np.sum(diff ** 2, axis=-1)
        K = np.exp(-dist_sq / (2 * self.bandwidth ** 2))

        # Copy with probability K
        copy_mask = np.random.rand(n_resample) < K
        if copy_mask.any():
            resample_indices = np.where(do_resample)[0][copy_mask]
            partner_indices = partners[do_resample][copy_mask]
            x_out[resample_indices] = x[partner_indices]

        return x_out

    def __call__(self, x_0, x_1, t_target=None):
        """
        Forward process: Skellam birth-death + Moran resampling.

        Applies the standard Skellam bridge, then resamples spatially.
        """
        # Get the standard Skellam bridge output
        out_dict = super().__call__(x_0, x_1, t_target=t_target)

        # Extract x_t and t from the output for resampling
        x_t = dlpack_backend(out_dict["inputs"]["x_t"], backend='numpy', dtype="float32")
        t = dlpack_backend(out_dict["inputs"]["t"], backend='numpy', dtype="float32")

        # Apply Moran resampling (spatial coupling)
        t_mean = float(np.mean(t))
        x_t_resampled = self._spatial_resample(
            x_t.round().astype(np.int32), t_mean
        )

        # Convert back
        x_t_resampled = dlpack_backend(
            x_t_resampled, backend=self.backend, dtype="float32", device=self.device
        )
        out_dict["inputs"]["x_t"] = x_t_resampled

        # Recompute target if needed
        if self.delta:
            x_0_tensor = out_dict["output"] + out_dict["inputs"]["x_t"]  # undo delta
            out_dict["output"] = x_0_tensor - x_t_resampled
        # If not delta, output is still x_0 (unchanged)

        return out_dict

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
    ):
        """
        Reverse sampling: identical to SkellamBridge.sampler().
        The Moran resampling only affects the forward process (training).
        The reverse process learns to undo the spatial coupling.
        """
        return super().sampler(
            x_1=x_1, z=z, model=model,
            return_trajectory=return_trajectory,
            return_x_hat=return_x_hat,
            return_M=return_M,
            guidance_x_0=guidance_x_0,
            guidance_schedule=guidance_schedule,
        )
