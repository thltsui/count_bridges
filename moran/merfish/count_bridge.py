"""
Count Bridge (Independent Skellam) baseline for MERFISH.

This removes the Moran iterative resampling (no spatial kernel) and treats
each cell independently as N independent Skellam bridges.

Forward: x_t = x_1 - B_t + N_t where B_t, N_t are drawn from hypergeometric
and binomial distributions (as in original Count Bridges).
Reverse: Also independent hypergeometric jumps.

We implement PyTorch/numpy vectorised versions for MPS compatibility matching
the exact signature of the Moran equivalents.
"""

import numpy as np
import torch
from bridges.numpy.skellam import SkellamBridge
from bridges.numpy.slack_samplers import BesselM

class MerfishCountBridgeForward:
    """
    Independent Skellam forward process.
    """
    def __init__(self, n_substeps: int = 10, G: int = 649):
        # We wrap the original repo's SkellamBridge directly
        # BesselM evaluates the modified bessel functions needed for slack
        self.slack_sampler = BesselM(1.0, 1.0)
        self.bridge = SkellamBridge(
            n_steps=n_substeps,
            slack_sampler=self.slack_sampler,
            delta=False,
            device='cpu', # Keep it in numpy/cpu space for bridges
            backend='numpy'
        )
        self.G = G

    def simulate(self, x_0, coords, t_target: float) -> np.ndarray:
        """
        x_0: [N, G] int  clean cells
        coords: ignored for Skellam, kept for signature compatibility

        Direct Skellam forward: add Poisson(lambda*t) births, subtract
        Poisson(lambda*t) deaths, keep >= 0.
        """
        if hasattr(x_0, 'numpy'):
            x_0 = x_0.cpu().numpy()
        x_0 = np.maximum(x_0, 0).astype(np.int32)
        lam = np.maximum(x_0.astype(np.float32), 0.5) * t_target
        births = np.random.poisson(lam).astype(np.int32)
        deaths = np.random.poisson(lam).astype(np.int32)
        x_t = np.clip(x_0 + births - deaths, 0, None).astype(np.int32)
        return x_t
        
    def sample_prior(self, n_cells: int, ref_counts: np.ndarray) -> np.ndarray:
        """
        Sample x_1: prior distribution.
        The original Count Bridge uses an independent uniform/poisson prior.
        For an apples-to-apples generative comparison, we use the EXACT SAME
        mixed prior as Moran so that performance differences are due to the
        bridge and NOT the prior.
        """
        M, G = ref_counts.shape
        # Just random uniform mixture choice for each cell independently
        prototype_idx = np.random.choice(M, size=n_cells, replace=True)
        prototypes = ref_counts[prototype_idx].astype(np.float32)
        x_1 = np.random.poisson(np.maximum(prototypes, 0.5))
        return np.clip(x_1, 0, 500).astype(np.int32)

class MerfishCountBridgeReverse:
    """
    Independent Skellam reverse sample.
    """
    def __init__(self, n_substeps: int = 10):
        self.slack_sampler = BesselM(1.0, 1.0)
        self.bridge = SkellamBridge(
            n_steps=n_substeps,
            slack_sampler=self.slack_sampler,
            delta=False,
            device='cpu',
            backend='numpy'
        )

    def reverse_step(
        self,
        x_t,
        x_hat,
        coords,
        t_curr: float,
        dt: float,
    ):
        """
        Count bridge reverse step from t_curr.
        Since the original SkellamBridge.sampler is an entire trajectory sampler,
        we just extract one step of it, or re-implement the hypergeometric draw.
        """
        if hasattr(x_t, 'numpy'): x_t = x_t.cpu().numpy()
        if hasattr(x_hat, 'numpy'): x_hat = x_hat.cpu().numpy()
        
        N, G = x_t.shape
        
        # Determine the discrete step indices
        # k corresponds to t_curr, k-1 corresponds to t_curr - dt
        k = max(1, int(t_curr * self.bridge.n_steps))
        
        if k <= 1:
            return x_hat.copy() # jump to target
            
        x_t = x_t.astype(np.int32)
        x0_hat_t = x_hat.astype(np.int32)
        
        diff = x_t - x0_hat_t
        M_t = self.slack_sampler(diff)
        
        N_t = np.abs(diff) + 2 * M_t
        B_t = (N_t + diff) // 2
        
        rho = self.bridge.weights[k-1] / self.bridge.weights[k]
        rho = min(int(rho), 1.0) if isinstance(rho, (int, float)) else np.minimum(rho, 1.0)
        
        non_zero = N_t > 0
        N_s = np.zeros_like(N_t)
        N_s[non_zero] = np.random.binomial(N_t[non_zero], rho)
        
        non_zero = N_s > 0
        B_s = np.zeros_like(B_t)
        B_s[non_zero] = np.random.hypergeometric(
            ngood=B_t[non_zero],
            nbad=N_t[non_zero] - B_t[non_zero],
            nsample=N_s[non_zero]
        )
        
        x_s = x_t - 2 * (B_t - B_s) + (N_t - N_s)
        return np.clip(x_s, 0, 500)
