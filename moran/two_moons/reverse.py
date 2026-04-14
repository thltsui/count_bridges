import numpy as np
from scipy.special import ive

class MoranReverseCTMC:
    """
    Reverse-time generating Continuous-Time Markov Chain for Regime A.
    Uses Independent Pseudo-Marginal Skellam Approximations to bypass interacting intractability.
    """
    def __init__(
        self,
        g_max: int = 195,
        gamma_mut: float = 5.0,
        kappa: float = 2.0,
        bandwidth: float = 5.0,
        n_substeps: int = 15,
    ):
        self.g_max = g_max
        self.gamma_mut = gamma_mut
        self.kappa = kappa
        self.bandwidth = bandwidth
        self.n_substeps = n_substeps
        self.dx_opts = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)
        
    def _time_scale(self, t: float) -> float:
        return 1.0 / (max(1.0 - t, 0.001) ** 2)
        
    def get_integrated_rate(self, t_start: float, t_end: float) -> float:
        def F(t):
            return 1.0 / max(1.0 - t, 0.001) - 1.0
        return max(F(t_end) - F(t_start), 0.0)

    def _kernel(self, x: np.ndarray) -> np.ndarray:
        xf = x.astype(np.float64)
        diff = xf[:, None, :] - xf[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.exp(-dist_sq / (2 * self.bandwidth ** 2))
        
    def tau_leap(self, x_t_batch: np.ndarray, x_0_hat_batch: np.ndarray, t_curr: float, dt: float) -> np.ndarray:
        B, N, _ = x_t_batch.shape
        x_out = x_t_batch.copy()
        
        total_steps = max(1, int(self.n_substeps * dt))
        dt_sub = dt / total_steps
        
        for step in range(total_steps):
            t_eval = t_curr - (step + 0.5) * dt_sub
            if t_eval <= 0.001:
                continue
                
            rate_mult = self._time_scale(t_eval)
            S_t = self.get_integrated_rate(0, t_eval)
            mu = 0.25 * self.gamma_mut * S_t
            if mu < 1e-4:
                mu = 1e-4
                
            # 1. REVERSE MUTATION
            diff_curr = np.abs(x_out - x_0_hat_batch)
            p_curr = ive(diff_curr[..., 0], 2*mu) * ive(diff_curr[..., 1], 2*mu)
            p_curr = np.maximum(p_curr, 1e-30)
            
            # [B, N, 4, 2]
            cx = x_out[:, :, None, :] + self.dx_opts[None, None, :, :]
            valid = (cx[..., 0] >= 0) & (cx[..., 0] <= self.g_max) & \
                    (cx[..., 1] >= 0) & (cx[..., 1] <= self.g_max)
                    
            c_hat = x_0_hat_batch[:, :, None, :]
            c_diff = np.abs(cx - c_hat)
            p_cand = ive(c_diff[..., 0], 2*mu) * ive(c_diff[..., 1], 2*mu)
            
            r_mut = self.gamma_mut * rate_mult * 0.25 * (p_cand / p_curr[:, :, None])
            r_mut = np.where(valid, r_mut, 0.0)
            
            jumps = np.random.rand(B, N, 4) < (r_mut * dt_sub)
            jumped = jumps.any(axis=-1)
            jump_dir = jumps.argmax(axis=-1)
            
            # BxN grid indices
            b_idx, n_idx = np.where(jumped)
            x_out[b_idx, n_idx] = cx[b_idx, n_idx, jump_dir[b_idx, n_idx]]
            
            # 2. REVERSE RESAMPLING
            if self.kappa > 0 and N > 1:
                diff_curr = np.abs(x_out - x_0_hat_batch)
                p_curr = ive(diff_curr[..., 0], 2*mu) * ive(diff_curr[..., 1], 2*mu)
                p_target = ive(0, 2*mu) * ive(0, 2*mu)
                
                spatial_dist_sq = np.sum((x_out - x_0_hat_batch)**2, axis=-1)
                k_val = np.exp(-spatial_dist_sq / (2 * self.bandwidth ** 2))
                
                base_r = self.kappa * rate_mult / N
                r_jump = base_r * k_val * (p_target / np.maximum(p_curr, 1e-30))
                r_jump = np.minimum(r_jump, 0.9 / dt_sub)
                
                is_at_target = (diff_curr[..., 0] == 0) & (diff_curr[..., 1] == 0)
                r_jump[is_at_target] = 0.0
                
                jumped_resample = np.random.rand(B, N) < (r_jump * dt_sub)
                x_out[jumped_resample] = x_0_hat_batch[jumped_resample]
                
        return x_out
