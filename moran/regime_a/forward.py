"""
Regime A Forward Process: N particles hopping on a 2D integer grid.

State: {x^(1), ..., x^(N)} where x^(i) ∈ {0, ..., g_max}^2.
The histogram c_{ij} = #{k : x^(k) = (i,j)} is DERIVED by counting.

Generator:
  Af(x) = sum_i gamma_mut * sum_{y in nbrs(x_i)} 1/|nbrs| [f(x^{i<-y}) - f(x)]
         + kappa/N * sum_{i!=j} K(x_i, x_j) [f(x^{i<-j}) - f(x)]

Mutation: symmetric random walk on the grid (hop to 4-connected neighbour).
  Stationary distribution: Uniform over grid cells.
Resampling: particle i copies particle j's position, weighted by K.

Time scaling: rates multiply by 1/(1-t)^2 so total physical time = infinity
at t=1, guaranteeing convergence to the de Finetti equilibrium.

De Finetti prior: p ~ Dir(theta/G), particles iid ~ Categorical(p),
where G = (g_max+1)^2 and theta = 2*gamma_mut/kappa.
"""

import numpy as np


class ParticleMoranForward:
    """
    Forward process for Regime A: N particles on a 2D grid.

    Parameters:
        g_max: max grid coordinate ({0, ..., g_max}^2)
        gamma_mut: base mutation rate (random walk hops per unit time)
        kappa: base resampling rate
        bandwidth: RBF kernel bandwidth for resampling
        n_substeps: tau-leaping substeps per unit time
    """

    def __init__(
        self,
        g_max: int = 127,
        gamma_mut: float = 5.0,
        kappa: float = 2.0,
        bandwidth: float = 5.0,
        n_substeps: int = 30,
    ):
        self.g_max = g_max
        self.gamma_mut = gamma_mut
        self.kappa = kappa
        self.bandwidth = bandwidth
        self.n_substeps = n_substeps

    @property
    def theta(self) -> float:
        """Mutation-drift ratio. Controls de Finetti prior diversity."""
        if self.kappa == 0:
            return float('inf')
        return 2 * self.gamma_mut / self.kappa

    @property
    def G(self) -> int:
        """Total number of grid cells."""
        return (self.g_max + 1) ** 2

    def _time_scale(self, t: float) -> float:
        """Rate multiplier: 1/(1-t)^2. Blows up at t->1 (stationarity)."""
        return 1.0 / max(1.0 - t, 0.01) ** 2

    def _kernel(self, x: np.ndarray) -> np.ndarray:
        """Pairwise RBF kernel. x: [N, 2] -> [N, N]."""
        xf = x.astype(np.float64)
        diff = xf[:, None, :] - xf[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.exp(-dist_sq / (2 * self.bandwidth ** 2))

    def simulate(self, x_0: np.ndarray, t_target: float) -> np.ndarray:
        """
        Tau-leaping simulation from x_0 at t=0 to t=t_target.

        x_0: [N, 2] integer particle positions (data)
        t_target: float in (0, 1]
        returns: [N, 2] integer positions at time t_target
        """
        N = x_0.shape[0]
        x = x_0.copy()

        total_steps = max(1, int(self.n_substeps * t_target))
        dt = t_target / total_steps

        for step in range(total_steps):
            t_mid = (step + 0.5) * dt
            rate_mult = self._time_scale(t_mid)

            # --- Mutation: random walk ---
            # Number of hops per particle ~ Poi(gamma * rate_mult * dt)
            n_hops = np.random.poisson(self.gamma_mut * rate_mult * dt, size=N)
            for i in range(N):
                for _ in range(min(n_hops[i], 50)):  # cap to prevent runaway
                    d = np.random.randint(4)
                    dx = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]][d])
                    x[i] = np.clip(x[i] + dx, 0, self.g_max)

            # --- Resampling: Moran kernel-weighted copying ---
            if self.kappa > 0 and N > 1:
                K = self._kernel(x)
                np.fill_diagonal(K, 0)
                for i in range(N):
                    rate_i = self.kappa * rate_mult / N * K[i].sum() * dt
                    if np.random.rand() < min(rate_i, 0.8):
                        w = K[i]
                        ws = w.sum()
                        if ws > 0:
                            j = np.random.choice(N, p=w / ws)
                            x[i] = x[j].copy()

        return x

    def sample_prior(self, n_particles: int) -> np.ndarray:
        """
        Sample from the de Finetti equilibrium.

        p ~ Dir(theta/G, ..., theta/G) over G joint grid cells,
        then N particles iid from Categorical(p).

        returns: [n_particles, 2] integer positions
        """
        g = self.g_max + 1
        alpha = max(self.theta / self.G, 1e-8)
        p = np.random.dirichlet(np.full(self.G, alpha))
        cell_idx = np.random.choice(self.G, size=n_particles, p=p)
        result = np.zeros((n_particles, 2), dtype=np.int32)
        result[:, 0] = cell_idx // g
        result[:, 1] = cell_idx % g
        return result

    def to_histogram(self, x: np.ndarray) -> np.ndarray:
        """Convert particle positions to count histogram on grid."""
        g = self.g_max + 1
        hist = np.zeros((g, g), dtype=np.int32)
        for xi, yi in x:
            if 0 <= xi < g and 0 <= yi < g:
                hist[xi, yi] += 1
        return hist
