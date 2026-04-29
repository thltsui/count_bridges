"""
Moran forward process for 649-dim gene expression (MERFISH).

Each particle is a cell; its type is a G-dim non-negative integer vector.
Two processes act simultaneously:
  1. Mutation: per-gene immigration-death CTMC  x_j → x_j ± 1
  2. Resampling: cell i copies cell j's entire G-dim profile at rate
                 (kappa/N) * K(p_i, p_j)
     where K is a spatial Gaussian kernel on (x_um, y_um) coordinates.

Time scaling 1/(1-t)^2 guarantees convergence to de Finetti equilibrium
DP(theta, nu) at t=1, where theta = 2*gamma_mut/kappa.
"""

import numpy as np
import torch


class MerfishMoranForward:
    """
    Forward process for N cells in one tissue spot.

    Parameters
    ----------
    gamma_mut : float
        Per-gene mutation rate (immigration/death).
    kappa : float
        Moran resampling rate.
    h_kernel : float
        Spatial kernel bandwidth in microns. Typically set to the median
        inter-cell distance within spots (~15–30 µm for mouse brain MERFISH).
    G : int
        Number of genes (649 for Vizgen MERFISH Mouse Brain).
    n_substeps : int
        Tau-leaping substeps per unit time interval.
    max_count : int
        Cap on individual gene counts to prevent runaway.
    """

    def __init__(
        self,
        gamma_mut: float = 1.0,
        kappa: float = 0.5,
        h_kernel: float = 20.0,
        G: int = 649,
        n_substeps: int = 20,
        max_count: int = 500,
    ):
        self.gamma_mut = gamma_mut
        self.kappa = kappa
        self.h_kernel = h_kernel
        self.G = G
        self.n_substeps = n_substeps
        self.max_count = max_count

    @property
    def theta(self) -> float:
        """Mutation-drift ratio controlling DP concentration."""
        if self.kappa == 0:
            return float("inf")
        return 2.0 * self.gamma_mut / self.kappa

    def _time_scale(self, t: float) -> float:
        """Rate multiplier 1/(1-t)^2."""
        return 1.0 / max(1.0 - t, 0.001) ** 2

    def _spatial_kernel(self, coords: np.ndarray) -> np.ndarray:
        """
        Gaussian kernel on spatial coords.

        coords: [N, 2]  cell centroids in microns
        returns: [N, N] kernel matrix (diagonal zeroed)
        """
        diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 2]
        dist_sq = np.sum(diff ** 2, axis=-1)             # [N, N]
        K = np.exp(-dist_sq / (2.0 * self.h_kernel ** 2))
        np.fill_diagonal(K, 0.0)
        return K.astype(np.float32)

    def _mutation_step(
        self,
        x: np.ndarray,
        rate_mult: float,
        dt: float,
    ) -> np.ndarray:
        """
        Per-gene immigration-death tau-leap.

        x: [N, G] int  gene expression counts
        Immigration rate = gamma_mut * rate_mult  (always)
        Death rate       = gamma_mut * rate_mult  per molecule
        => at stationarity: Poisson(1) per gene

        We implement this as:
          births ~ Poi(gamma_mut * rate_mult * dt)  per (cell, gene)
          deaths ~ Bin(x_j, 1 - exp(-gamma_mut * rate_mult * dt)) per (cell, gene)
        """
        r = self.gamma_mut * rate_mult * dt
        # deaths: Binomial thinning
        p_survive = np.exp(-r)
        deaths = x - np.random.binomial(x, p_survive)
        # births: independent Poisson
        births = np.random.poisson(r, size=x.shape)
        x_new = np.clip(x - deaths + births, 0, self.max_count)
        return x_new.astype(np.int32)

    def _resampling_step(
        self,
        x: np.ndarray,
        K: np.ndarray,
        rate_mult: float,
        dt: float,
        N: int,
    ) -> np.ndarray:
        """
        Independent per-gene Moran resampling step (Mocking Gene Transfer / Diffusion).

        x: [N, G] int
        K: [N, N] spatial kernel (diagonal 0)
        """
        x_new = x.copy()
        for i in range(N):
            total_rate = self.kappa * rate_mult / N * K[i].sum() * dt
            p_replace = min(total_rate, 0.9)
            
            if p_replace > 0:
                # 1. Independent mask of which genes are replaced in this dt time step
                replaced_mask = np.random.rand(self.G) < p_replace
                num_replaced = replaced_mask.sum()
                
                if num_replaced > 0:
                    w = K[i]
                    ws = w.sum()
                    if ws > 0:
                        # 2. For each gene to be replaced, sample a source neighbor independently
                        j_choices = np.random.choice(N, size=num_replaced, p=w / ws)
                        
                        # 3. Apply the replacements for the specific genes
                        genes = np.where(replaced_mask)[0]
                        x_new[i, genes] = x[j_choices, genes]
                        
        return x_new

    def simulate(
        self,
        x_0: np.ndarray,
        coords: np.ndarray,
        t_target: float,
    ) -> np.ndarray:
        """
        Simulate the Moran CTMC from t=0 to t=t_target.

        x_0:    [N, G] int  clean cell expression profiles
        coords: [N, 2] float cell centroids in microns
        t_target: float in (0, 1]

        Returns: [N, G] int  noisy expression at t_target
        """
        N = x_0.shape[0]
        x = x_0.copy().astype(np.int32)

        total_steps = max(1, int(self.n_substeps * t_target))
        dt = t_target / total_steps

        # Precompute spatial kernel (fixed for this spot)
        K = self._spatial_kernel(coords)

        for step in range(total_steps):
            t_mid = (step + 0.5) * dt
            rate_mult = self._time_scale(t_mid)

            # 1. Mutation
            x = self._mutation_step(x, rate_mult, dt)

            # 2. Resampling
            if self.kappa > 0 and N > 1:
                x = self._resampling_step(x, K, rate_mult, dt, N)

        return x

    def sample_prior(
        self,
        n_cells: int,
        ref_counts: np.ndarray,
    ) -> np.ndarray:
        """
        Sample from approximated de Finetti prior DP(theta, nu).

        We approximate this as:
          1. Sample a "prototype" expression profile as the mean over a
             random subset of reference cells.
          2. Add Poisson noise around the prototype.
          3. Assign all cells a mixture of prototypes (using DP stick-breaking).

        This is more biologically meaningful than the uniform prior used
        in the 2D case — the prior already has gene expression structure.

        n_cells: int  number of cells (N) in this spot
        ref_counts: [M, G] sample of real expression vectors to use as
                    mixture components. Passed at runtime from the training set.

        Returns: [N, G] int  sampled population
        """
        M, G = ref_counts.shape
        theta = max(self.theta, 0.1)

        # DP stick-breaking (truncated at K_trunc clusters)
        K_trunc = min(max(2, int(np.ceil(theta * np.log(n_cells + 1)))), n_cells)

        # Sample cluster prototypes from reference data
        prototype_idx = np.random.choice(M, size=K_trunc, replace=True)
        prototypes = ref_counts[prototype_idx].astype(np.float32)  # [K, G]

        # Stick-breaking weights
        betas = np.random.beta(1.0, theta, size=K_trunc)
        sticks = np.zeros(K_trunc)
        remaining = 1.0
        for k in range(K_trunc):
            sticks[k] = betas[k] * remaining
            remaining *= (1.0 - betas[k])
        sticks[-1] += remaining
        sticks = np.clip(sticks / sticks.sum(), 0, 1)

        # Assign cells to clusters
        cluster_assignments = np.random.choice(K_trunc, size=n_cells, p=sticks)

        # Sample around prototypes with Poisson noise
        x = np.zeros((n_cells, G), dtype=np.int32)
        for i in range(n_cells):
            k = cluster_assignments[i]
            proto = prototypes[k]
            # Poisson with rate = prototype count
            x[i] = np.random.poisson(np.maximum(proto, 0.5))

        return np.clip(x, 0, self.max_count).astype(np.int32)
