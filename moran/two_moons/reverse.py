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
        guidance_weight: float = 2.0,
    ):
        self.g_max = g_max
        self.gamma_mut = gamma_mut
        self.kappa = kappa
        self.bandwidth = bandwidth
        self.n_substeps = n_substeps
        self.guidance_weight = guidance_weight
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
            diff_curr = np.minimum(diff_curr, (self.g_max + 1) - diff_curr)
            p_curr = ive(diff_curr[..., 0], 2*mu) * ive(diff_curr[..., 1], 2*mu)
            p_curr = np.maximum(p_curr, 1e-30)
            
            # [B, N, 4, 2]
            cx = (x_out[:, :, None, :] + self.dx_opts[None, None, :, :]) % (self.g_max + 1)
            valid = np.ones(cx.shape[:-1], dtype=bool)
                    
            c_hat = x_0_hat_batch[:, :, None, :]
            c_diff = np.abs(cx - c_hat)
            c_diff = np.minimum(c_diff, (self.g_max + 1) - c_diff)
            p_cand = ive(c_diff[..., 0], 2*mu) * ive(c_diff[..., 1], 2*mu)
            
            ratio = (p_cand / p_curr[:, :, None]) ** self.guidance_weight
            r_mut = self.gamma_mut * rate_mult * 0.25 * ratio
            r_mut = np.where(valid, r_mut, 0.0)
            
            jumps = np.random.rand(B, N, 4) < (r_mut * dt_sub)
            jumped = jumps.any(axis=-1)
            jump_dir = jumps.argmax(axis=-1)
            
            # BxN grid indices
            b_idx, n_idx = np.where(jumped)
            x_out[b_idx, n_idx] = cx[b_idx, n_idx, jump_dir[b_idx, n_idx]]
            
            # 2. ADJOINT REVERSE RESAMPLING (Coalescent Transport)
            if self.kappa > 0 and N > 1:
                # p_curr traces proximity/importance of each state relative to theoretical origin
                diff_curr = np.abs(x_out - x_0_hat_batch)
                diff_curr = np.minimum(diff_curr, (self.g_max + 1) - diff_curr)
                p_curr = ive(diff_curr[..., 0], 2*mu) * ive(diff_curr[..., 1], 2*mu)
                p_curr = np.maximum(p_curr, 1e-30)
                
                # Spatial interaction kernel K(x_i, x_j)
                diff_ij = np.abs(x_out[:, :, None, :] - x_out[:, None, :, :])
                diff_ij = np.minimum(diff_ij, (self.g_max + 1) - diff_ij)
                dist_sq_ij = np.sum(diff_ij**2, axis=-1)
                k_val_ij = np.exp(-dist_sq_ij / (2 * self.bandwidth ** 2))
                
                # Remove self-interactions
                eye_mask = np.broadcast_to(np.eye(N, dtype=bool)[None, :, :], (B, N, N)).copy()
                k_val_ij[eye_mask] = 0.0
                
                # Adjoint Importance Ratio: particle i is evaluated against jumping to particle j's coordinates
                ratio_ij = (p_curr[:, None, :] / p_curr[:, :, None]) ** self.guidance_weight
                
                kappa_eff = self.kappa * t_eval
                base_r = kappa_eff * rate_mult / N
                r_jump_matrix = base_r * k_val_ij * ratio_ij  # [B, N, N]
                
                total_r_jump = np.sum(r_jump_matrix, axis=-1)
                total_r_jump = np.minimum(total_r_jump, 0.9 / dt_sub)
                
                jump_prob = total_r_jump * dt_sub
                jumped = np.random.rand(B, N) < jump_prob
                
                # Normalize cross-rates properly for categorical choice
                denom = np.maximum(np.sum(r_jump_matrix, axis=-1, keepdims=True), 1e-10)
                prob_matrix = r_jump_matrix / denom
                
                cum_prob = np.cumsum(prob_matrix, axis=-1)
                denom_cum = np.maximum(cum_prob[:, :, -1:], 1e-10)
                cum_prob_norm = cum_prob / denom_cum
                
                rand_vals = np.random.rand(B, N, 1)
                target_j = np.argmax(rand_vals < cum_prob_norm, axis=-1)
                
                # Execute Adjoint Coalescence simultaneously utilizing original batch copies limits
                x_out_snapshot = x_out.copy()
                b_idx, i_idx = np.where(jumped)
                j_idx = target_j[b_idx, i_idx]
                x_out[b_idx, i_idx] = x_out_snapshot[b_idx, j_idx]
                
        return x_out
