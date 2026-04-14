"""
Moran Bridge: N-particle collate function for training and generation.

This is the bridge-layer adapter that wraps the Moran CTMC into the same
interface as SkellamBridge, so it can be used as a drop-in in main_mps.py.

The CTMC simulation is delegated entirely to ParticleMoranForward (in
moran/two_moons/forward.py) — no duplication of kernel or tau-leaping logic.

    bridges/numpy/moran.py   ← this file: training collate + sampler interface
    moran/two_moons/forward.py ← CTMC implementation (authoritative source)
"""

import numpy as np
import torch
from typing import Callable, Optional
from .utils import dlpack_backend
from .scheduling import make_weight_schedule
from moran.two_moons.forward import ParticleMoranForward


class MoranBridge:
    """
    N-particle Moran bridge: collate function + reverse sampler.

    Wraps ParticleMoranForward to produce training tuples
    (t, x_t, x_0) and run the iterative denoising reverse pass.

    Parameters
    ----------
    n_steps : int
        Number of discrete time steps for training / generation.
    slack_sampler : Callable
        Kept for interface compatibility with SkellamBridge (unused).
    kappa : float
        Resampling rate.
    bandwidth : float
        RBF kernel bandwidth for spatial similarity.
    theta : float
        Mutation-drift ratio 2*gamma_mut/kappa (controls de Finetti prior).
    n_substeps : int
        Tau-leaping substeps per unit time (passed to ParticleMoranForward).
    gamma_mut : float
        Mutation rate (random-walk hops per unit time).
    grid_max : int
        Maximum grid coordinate; type space is {0, ..., grid_max}^2.
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
        self.slack_sampler = slack_sampler   # interface compat
        self.delta = delta
        self.backend = backend
        self.device = device

        self.time_points = np.linspace(0, 1, n_steps + 1)
        self.weights = make_weight_schedule(n_steps, schedule_type, **schedule_kwargs)

        # Delegate all CTMC logic to the authoritative implementation
        self._fwd = ParticleMoranForward(
            g_max=grid_max,
            gamma_mut=gamma_mut,
            kappa=kappa,
            bandwidth=bandwidth,
            n_substeps=n_substeps,
        )

    # ------------------------------------------------------------------
    # Convenience properties (mirror ParticleMoranForward attributes)
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> float:
        return self._fwd.kappa

    @property
    def bandwidth(self) -> float:
        return self._fwd.bandwidth

    @property
    def theta(self) -> float:
        return self._fwd.theta

    @property
    def gamma_mut(self) -> float:
        return self._fwd.gamma_mut

    @property
    def grid_max(self) -> int:
        return self._fwd.g_max

    # ------------------------------------------------------------------
    # Training collate: produce (t, x_t, target) by Moran forward
    # ------------------------------------------------------------------

    def __call__(self, x_0, x_1, t_target=None):
        """
        Produce training tuples (t, x_t, target) by simulating Moran forward.

        x_0 : target data (e.g. Two-Moons integer positions)
        x_1 : source — ignored (Moran does not use a source distribution)
        """
        x_0_np, _ = dlpack_backend(x_0, x_1, backend="numpy", dtype="int32")
        B, D = x_0_np.shape

        # Sample (or fix) a time step
        if t_target is not None:
            time_diffs = np.abs(self.time_points - t_target)
            k = np.argmin(time_diffs)
            k = np.broadcast_to(k, (B,))
        else:
            k = np.random.randint(1, self.n_steps + 1, (B,))

        t = self.time_points[k].reshape(-1, 1)

        # Simulate forward — all particles share the same t (max over batch)
        t_max = float(np.max(t))
        x_t = self._fwd.simulate(x_0_np, t_max)

        target = x_0_np  # denoiser predicts x_0

        x_t, t, target = dlpack_backend(
            x_t, t, target, backend=self.backend, dtype="float32"
        )
        return {"inputs": {"x_t": x_t, "t": t}, "output": target}

    # ------------------------------------------------------------------
    # De Finetti prior (thin wrapper around ParticleMoranForward)
    # ------------------------------------------------------------------

    def sample_prior(
        self,
        n_samples: int,
        data_dim: int = 2,
        value_range: int = None,
    ) -> np.ndarray:
        """Sample N particles from the de Finetti equilibrium DP(theta, Uniform(E))."""
        return self._fwd.sample_prior(n_samples)

    # ------------------------------------------------------------------
    # Reverse: iterative denoising — de Finetti prior → data
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
        Reverse: de Finetti prior → iterative denoising → data.

        At each step:
          1. Model predicts x_0_hat from (x_t, t)
          2. Re-noise x_0_hat to t_{k-1} using the Moran forward
          3. At t=0 output x_0_hat directly

        No bridge kernels. Just forward simulation + learned denoiser.
        """
        B, D = x_1.shape

        # Start from de Finetti prior
        x_t = self._fwd.sample_prior(B).astype(np.int32)

        traj, xhat_traj = [x_t.copy()], []

        for k in range(self.n_steps, 0, -1):
            t_curr = self.time_points[k]
            t_next = self.time_points[k - 1]
            t = np.broadcast_to(t_curr, (B, 1))

            # Model predicts x_0
            x_t_dl, t_dl = dlpack_backend(
                x_t, t, backend=self.backend, dtype="float32"
            )
            model_out = model.sample(x_t=x_t_dl, t=t_dl, **z)
            x0_hat = dlpack_backend(model_out, backend="numpy", dtype="float32")[0]
            x0_hat = np.clip(
                x0_hat.round().astype(np.int32), 0, self._fwd.g_max
            )

            if guidance_x_0 is not None:
                gs = guidance_schedule[k] if guidance_schedule is not None else 0
                x0_hat = np.clip(
                    (gs * guidance_x_0 + (1 - gs) * x0_hat).round().astype(np.int32),
                    0, self._fwd.g_max,
                )

            # Re-noise via Moran forward to t_next
            if t_next > 1e-6:
                x_t = self._fwd.simulate(x0_hat, t_next)
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
