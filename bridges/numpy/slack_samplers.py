import numpy as np
from scipy.special import iv
from .sampling.bessel import sample_bessel_devroye

class ConstantM:
    def __init__(self, m: int, markov: bool = True):
        self.m = m
        self.markov = markov

    def __call__(self, diff: np.ndarray):
        return np.full(diff.shape, self.m)

class PoissonM:
    def __init__(self, lam_p: float, lam_m: float, markov: bool = False):
        self.lam_p = np.asarray(lam_p)
        self.lam_m = np.asarray(lam_m)
        self.lam_star = 2.0 * np.sqrt(self.lam_p * self.lam_m)
        self.markov = markov
    
    def __call__(self, diff: np.ndarray):
        return np.random.poisson(self.lam_star, diff.shape)

class BesselM:
    def __init__(self, lam_p: float, lam_m: float, markov: bool = True):
        self.lam_p = np.asarray(lam_p)
        self.lam_m = np.asarray(lam_m)
        self.lam_star = 2.0 * np.sqrt(self.lam_p * self.lam_m)
        self.markov = markov
        
    def __call__(self, diff: np.ndarray):
        """Sample M element-wise from the Bessel posterior."""
        flat_d = np.abs(diff).flatten().astype(int)
        M_flat = np.zeros_like(flat_d)
        lam_p = float(self.lam_p) if np.ndim(self.lam_p) == 0 else self.lam_p
        lam_m = float(self.lam_m) if np.ndim(self.lam_m) == 0 else self.lam_m

        # Group by unique |d| values — bessel sampler needs scalar d
        for d_val in np.unique(flat_d):
            mask = flat_d == d_val
            n = int(mask.sum())
            samples = sample_bessel_devroye(
                float(lam_p), float(lam_m),
                int(d_val), n_samples=n
            )
            M_flat[mask] = samples

        return M_flat.reshape(diff.shape)