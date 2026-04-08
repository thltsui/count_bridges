"""
Regime B Forward Process: counts on a fixed grid + Moran resampling.

State: c = (c_1, ..., c_G) ∈ Z_≥0^G, where G = number of grid cells.
The histogram IS the state — no particles, just counts per cell.

Generator:
  Af(c) = sum_i [lambda_b(f(c^{i+}) - f(c)) + lambda_d * 1_{c_i>0}(f(c^{i-}) - f(c))]
         + kappa/G * sum_{i!=j} K(pos_i, pos_j) [f(c^{i<-j}) - f(c)]

Mutation: Poisson birth-death per cell (Count Bridge mechanism).
  c^{i+}: count at cell i increases by 1
  c^{i-}: count at cell i decreases by 1
Resampling: cell i copies cell j's ENTIRE count.

This builds directly on top of Count Bridges:
  - The Skellam bridge handles per-cell birth-death with exact conditionals
  - Moran resampling couples cells spatially (the new component)

For training, we use the Skellam bridge to get (t, c_t, c_0) per cell,
then apply resampling between cells.

For generation, we use Skellam bridge steps per cell (exact), then
resampling between cells (Lie-Trotter splitting).
"""

import numpy as np
from typing import Optional


class GridMoranForward:
    """
    Regime B: Counts on a 2D grid with Moran resampling.

    The grid has g x g cells. Each cell has a count c_i >= 0.
    Per-cell dynamics are Poisson birth-death (handled by Skellam bridge).
    Inter-cell coupling is Moran resampling.

    Parameters:
        grid_size: number of cells per side (grid is grid_size x grid_size)
        kappa: resampling rate (0 = pure Count Bridge, no coupling)
        bandwidth: RBF kernel bandwidth (in cell units)
    """

    def __init__(
        self,
        grid_size: int = 32,
        kappa: float = 2.0,
        bandwidth: float = 3.0,
    ):
        self.grid_size = grid_size
        self.kappa = kappa
        self.bandwidth = bandwidth
        self.G = grid_size ** 2

        # Precompute cell positions and pairwise kernel
        pos = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
        self.cell_positions = pos  # [G, 2]
        self._precompute_kernel()

    def _precompute_kernel(self):
        """Precompute pairwise spatial kernel between all grid cells."""
        pos = self.cell_positions.astype(np.float64)
        diff = pos[:, None, :] - pos[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        self.K = np.exp(-dist_sq / (2 * self.bandwidth ** 2))
        np.fill_diagonal(self.K, 0)
        # Normalise rows for resampling probabilities
        row_sums = self.K.sum(axis=1, keepdims=True)
        self.K_norm = np.where(row_sums > 0, self.K / row_sums, 0)

    def resample_counts(self, c: np.ndarray, rate: float) -> np.ndarray:
        """
        Apply Moran resampling to a count vector.

        For each cell i, with probability rate * K_row_sum_i / G:
          cell i copies cell j's count, with j ~ K_norm[i].

        c: [G] integer count vector
        rate: effective resampling rate this step
        returns: [G] resampled count vector
        """
        c_out = c.copy()
        G = len(c)

        for i in range(G):
            resample_prob = rate * self.K[i].sum() / G
            if np.random.rand() < min(resample_prob, 0.5):
                j = np.random.choice(G, p=self.K_norm[i])
                c_out[i] = c[j]  # copy j's count to i

        return c_out

    def data_to_counts(self, X_int: np.ndarray, value_range: int) -> np.ndarray:
        """
        Convert discrete 2D point positions to a count vector on the grid.

        X_int: [N, 2] integer positions in {0, ..., value_range-1}^2
        returns: [G] count vector (G = grid_size^2)
        """
        # Map from value_range to grid_size
        scale = self.grid_size / value_range
        c = np.zeros(self.G, dtype=np.int32)
        for x, y in X_int:
            gi = int(np.clip(x * scale, 0, self.grid_size - 1))
            gj = int(np.clip(y * scale, 0, self.grid_size - 1))
            c[gi * self.grid_size + gj] += 1
        return c

    def counts_to_image(self, c: np.ndarray) -> np.ndarray:
        """Reshape count vector to 2D grid image. c: [G] -> [g, g]."""
        return c.reshape(self.grid_size, self.grid_size)
