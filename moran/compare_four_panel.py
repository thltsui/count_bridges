"""
Generate a 4-panel comparison figure:
  Ground Truth | Count Bridge | Moran Tau-Leap | Moran Adjoint

Uses the same pre-trained Moran checkpoint (N=100, θ=2.0) for both
Tau-Leap and Adjoint generation to isolate the effect of the reverse sampler.

Usage:
    .venv/bin/python moran/compare_four_panel.py
"""

import numpy as np
import torch
import logging
import math
from pathlib import Path
from scipy.special import ive

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ---- Import project modules ----
from datasets.discrete_moons import DiscreteMoonsDataset
from moran.regime_a.forward import ParticleMoranForward
from moran.experiment import MoranDenoiser, compute_metrics
from bridges.numpy.skellam import SkellamBridge
from bridges.numpy.slack_samplers import BesselM

# ============================================================
# Tau-Leap reverse (baseline — NO adjoint)
# ============================================================
class TauLeapReverse:
    def __init__(self, g_max=195, gamma_mut=5.0, kappa=2.0, bandwidth=5.0, n_substeps=15, guidance_weight=2.0):
        self.g_max, self.gamma_mut, self.kappa = g_max, gamma_mut, kappa
        self.bandwidth, self.n_substeps, self.guidance_weight = bandwidth, n_substeps, guidance_weight
        self.dx_opts = np.array([[0,1],[0,-1],[1,0],[-1,0]], dtype=np.int32)

    def _time_scale(self, t): return 1.0 / (max(1.0 - t, 0.001) ** 2)
    def get_integrated_rate(self, t_start, t_end):
        F = lambda t: 1.0 / max(1.0 - t, 0.001) - 1.0
        return max(F(t_end) - F(t_start), 0.0)

    def tau_leap(self, x_t_batch, x_0_hat_batch, t_curr, dt):
        B, N, _ = x_t_batch.shape
        x_out = x_t_batch.copy()
        total_steps = max(1, int(self.n_substeps * dt))
        dt_sub = dt / total_steps
        for step in range(total_steps):
            t_eval = t_curr - (step + 0.5) * dt_sub
            if t_eval <= 0.001: continue
            rate_mult = self._time_scale(t_eval)
            S_t = self.get_integrated_rate(0, t_eval)
            mu = max(0.25 * self.gamma_mut * S_t, 1e-4)
            # 1. REVERSE MUTATION
            diff_curr = np.abs(x_out - x_0_hat_batch)
            diff_curr = np.minimum(diff_curr, (self.g_max + 1) - diff_curr)
            p_curr = ive(diff_curr[...,0], 2*mu) * ive(diff_curr[...,1], 2*mu)
            p_curr = np.maximum(p_curr, 1e-30)
            cx = (x_out[:,:,None,:] + self.dx_opts[None,None,:,:]) % (self.g_max + 1)
            c_hat = x_0_hat_batch[:,:,None,:]
            c_diff = np.abs(cx - c_hat)
            c_diff = np.minimum(c_diff, (self.g_max + 1) - c_diff)
            p_cand = ive(c_diff[...,0], 2*mu) * ive(c_diff[...,1], 2*mu)
            ratio = (p_cand / p_curr[:,:,None]) ** self.guidance_weight
            r_mut = self.gamma_mut * rate_mult * 0.25 * ratio
            jumps = np.random.rand(B, N, 4) < (r_mut * dt_sub)
            jumped = jumps.any(axis=-1)
            jump_dir = jumps.argmax(axis=-1)
            b_idx, n_idx = np.where(jumped)
            x_out[b_idx, n_idx] = cx[b_idx, n_idx, jump_dir[b_idx, n_idx]]
            # 2. BASELINE REVERSE RESAMPLING (jump to target)
            if self.kappa > 0 and N > 1:
                diff_curr = np.abs(x_out - x_0_hat_batch)
                diff_curr = np.minimum(diff_curr, (self.g_max + 1) - diff_curr)
                p_curr = ive(diff_curr[...,0], 2*mu) * ive(diff_curr[...,1], 2*mu)
                p_target = ive(0, 2*mu) * ive(0, 2*mu)
                spatial_diff = np.abs(x_out - x_0_hat_batch)
                spatial_diff = np.minimum(spatial_diff, (self.g_max + 1) - spatial_diff)
                spatial_dist_sq = np.sum(spatial_diff**2, axis=-1)
                k_val = np.exp(-spatial_dist_sq / (2 * self.bandwidth ** 2))
                kappa_eff = self.kappa * t_eval
                base_r = kappa_eff * rate_mult / N
                ratio = (p_target / np.maximum(p_curr, 1e-30)) ** self.guidance_weight
                r_jump = base_r * k_val * ratio
                r_jump = np.minimum(r_jump, 0.9 / dt_sub)
                is_at_target = (diff_curr[...,0] == 0) & (diff_curr[...,1] == 0)
                r_jump[is_at_target] = 0.0
                jumped_resample = np.random.rand(B, N) < (r_jump * dt_sub)
                x_out[jumped_resample] = x_0_hat_batch[jumped_resample]
        return x_out


# ============================================================
# Adjoint Transport reverse
# ============================================================
class AdjointReverse:
    def __init__(self, g_max=195, gamma_mut=5.0, kappa=2.0, bandwidth=5.0, n_substeps=15, guidance_weight=2.0):
        self.g_max, self.gamma_mut, self.kappa = g_max, gamma_mut, kappa
        self.bandwidth, self.n_substeps, self.guidance_weight = bandwidth, n_substeps, guidance_weight
        self.dx_opts = np.array([[0,1],[0,-1],[1,0],[-1,0]], dtype=np.int32)

    def _time_scale(self, t): return 1.0 / (max(1.0 - t, 0.001) ** 2)
    def get_integrated_rate(self, t_start, t_end):
        F = lambda t: 1.0 / max(1.0 - t, 0.001) - 1.0
        return max(F(t_end) - F(t_start), 0.0)

    def tau_leap(self, x_t_batch, x_0_hat_batch, t_curr, dt):
        B, N, _ = x_t_batch.shape
        x_out = x_t_batch.copy()
        total_steps = max(1, int(self.n_substeps * dt))
        dt_sub = dt / total_steps
        for step in range(total_steps):
            t_eval = t_curr - (step + 0.5) * dt_sub
            if t_eval <= 0.001: continue
            rate_mult = self._time_scale(t_eval)
            S_t = self.get_integrated_rate(0, t_eval)
            mu = max(0.25 * self.gamma_mut * S_t, 1e-4)
            # 1. REVERSE MUTATION (same as baseline)
            diff_curr = np.abs(x_out - x_0_hat_batch)
            diff_curr = np.minimum(diff_curr, (self.g_max + 1) - diff_curr)
            p_curr = ive(diff_curr[...,0], 2*mu) * ive(diff_curr[...,1], 2*mu)
            p_curr = np.maximum(p_curr, 1e-30)
            cx = (x_out[:,:,None,:] + self.dx_opts[None,None,:,:]) % (self.g_max + 1)
            c_hat = x_0_hat_batch[:,:,None,:]
            c_diff = np.abs(cx - c_hat)
            c_diff = np.minimum(c_diff, (self.g_max + 1) - c_diff)
            p_cand = ive(c_diff[...,0], 2*mu) * ive(c_diff[...,1], 2*mu)
            ratio = (p_cand / p_curr[:,:,None]) ** self.guidance_weight
            r_mut = self.gamma_mut * rate_mult * 0.25 * ratio
            jumps = np.random.rand(B, N, 4) < (r_mut * dt_sub)
            jumped = jumps.any(axis=-1)
            jump_dir = jumps.argmax(axis=-1)
            b_idx, n_idx = np.where(jumped)
            x_out[b_idx, n_idx] = cx[b_idx, n_idx, jump_dir[b_idx, n_idx]]
            # 2. ADJOINT REVERSE RESAMPLING (N×N coalescence)
            if self.kappa > 0 and N > 1:
                diff_curr = np.abs(x_out - x_0_hat_batch)
                diff_curr = np.minimum(diff_curr, (self.g_max + 1) - diff_curr)
                p_curr = ive(diff_curr[...,0], 2*mu) * ive(diff_curr[...,1], 2*mu)
                p_curr = np.maximum(p_curr, 1e-30)
                # NxN spatial kernel
                diff_ij = np.abs(x_out[:,:,None,:] - x_out[:,None,:,:])
                diff_ij = np.minimum(diff_ij, (self.g_max + 1) - diff_ij)
                dist_sq_ij = np.sum(diff_ij**2, axis=-1)
                k_val_ij = np.exp(-dist_sq_ij / (2 * self.bandwidth ** 2))
                eye_mask = np.broadcast_to(np.eye(N, dtype=bool)[None,:,:], (B, N, N)).copy()
                k_val_ij[eye_mask] = 0.0
                ratio_ij = (p_curr[:,None,:] / p_curr[:,:,None]) ** self.guidance_weight
                kappa_eff = self.kappa * t_eval
                base_r = kappa_eff * rate_mult / N
                r_jump_matrix = base_r * k_val_ij * ratio_ij
                total_r_jump = np.sum(r_jump_matrix, axis=-1)
                total_r_jump = np.minimum(total_r_jump, 0.9 / dt_sub)
                jump_prob = total_r_jump * dt_sub
                jumped = np.random.rand(B, N) < jump_prob
                denom = np.maximum(np.sum(r_jump_matrix, axis=-1, keepdims=True), 1e-10)
                prob_matrix = r_jump_matrix / denom
                cum_prob = np.cumsum(prob_matrix, axis=-1)
                denom_cum = np.maximum(cum_prob[:,:,-1:], 1e-10)
                cum_prob_norm = cum_prob / denom_cum
                rand_vals = np.random.rand(B, N, 1)
                target_j = np.argmax(rand_vals < cum_prob_norm, axis=-1)
                x_out_snapshot = x_out.copy()
                b_idx, i_idx = np.where(jumped)
                j_idx = target_j[b_idx, i_idx]
                x_out[b_idx, i_idx] = x_out_snapshot[b_idx, j_idx]
        return x_out


# ============================================================
# Shared generation loop
# ============================================================
def generate_moran(model, fwd, rev_ctmc, n_particles, g_max, total_points, device_obj, n_denoise_steps=20):
    model.eval()
    n_gen = total_points // n_particles
    gen_batch = min(64, n_gen)
    all_gen = []
    with torch.no_grad():
        for start in range(0, n_gen, gen_batch):
            B = min(gen_batch, n_gen - start)
            x_t_batch = np.stack([fwd.sample_prior(n_particles) for _ in range(B)])
            for k in range(n_denoise_steps, 0, -1):
                t_val = k / n_denoise_steps
                x_t_tensor = torch.from_numpy(x_t_batch).float().to(device_obj)
                t_tensor = torch.full((B,), t_val, device=device_obj)
                x_0_hat = model.sample(x_t_tensor, t_tensor).cpu().numpy()
                x_0_hat = np.mod(np.round(x_0_hat), g_max + 1).astype(np.int32)
                t_next = (k - 1) / n_denoise_steps
                if t_next > 0.01:
                    x_t_batch = rev_ctmc.tau_leap(x_t_batch, x_0_hat, t_val, t_val - t_next)
                else:
                    x_t_batch = x_0_hat
            all_gen.append(x_t_batch.reshape(-1, 2))
    return np.concatenate(all_gen, axis=0)


# ============================================================
# Count Bridge generation (N=1 Skellam baseline)
# ============================================================
def generate_count_bridge(dataset, n_samples=5000, epochs=200, device="cpu"):
    """Train a quick Skellam count bridge and generate samples."""
    from main_mps import SimpleMLP, train, generate
    from torch.utils.data import random_split

    logging.info("  Training Count Bridge (Skellam)...")
    model = SimpleMLP(data_dim=2, hidden_dim=256, n_layers=4, noise_dim=16, m_samples=16)
    bessel = BesselM(lam_p=32.0, lam_m=32.0, markov=True)
    bridge = SkellamBridge(n_steps=10, slack_sampler=bessel, schedule_type="linear", backend="torch", device=0)

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_ds, eval_ds = random_split(dataset, [train_size, eval_size])

    dev = torch.device(device)
    _ = train(model, bridge, train_ds, epochs, batch_size=256, lr=1e-3, device=dev)

    logging.info("  Generating Count Bridge samples...")
    x_gen = generate(model, bridge, eval_ds, n_samples, n_steps=10, device=dev)
    return x_gen


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    g_max = 195
    value_range = g_max + 1
    N = 100
    theta = 2.0
    gamma_mut = 5.0
    bandwidth = 5.0
    kappa = 2 * gamma_mut / theta
    device_obj = torch.device("cpu")
    # Reduce sample count for speed
    n_gen_points = 10000

    # --- Dataset ---
    dataset = DiscreteMoonsDataset(size=50_000, value_range=value_range, scale=30.0, offset=80.0, noise=0.1)
    all_x0 = np.stack([dataset[i]['x_0'].numpy() for i in range(len(dataset))])
    logging.info(f"Dataset: {len(all_x0)} points")

    # --- 1. Ground Truth ---
    x_gt = all_x0[:n_gen_points]

    # --- 2. Count Bridge ---
    logging.info("Panel 2: Count Bridge (Skellam)")
    x_count = generate_count_bridge(dataset, n_samples=n_gen_points, epochs=200, device="cpu")

    # --- 3 & 4. Load Moran checkpoint ---
    ckpt_path = Path("outputs/moran_sweep_Nall/checkpoints/N100_theta2.0_final.pt")
    model = MoranDenoiser(hidden_dim=256, time_dim=32, noise_dim=16, n_layers=3, m_samples=8, g_max=g_max).to(device_obj)
    model.load_state_dict(torch.load(ckpt_path, map_location=device_obj, weights_only=True))
    logging.info(f"Loaded Moran checkpoint: {ckpt_path}")

    fwd = ParticleMoranForward(g_max=g_max, gamma_mut=gamma_mut, kappa=kappa, bandwidth=bandwidth, n_substeps=30)

    # --- 3. Moran Tau-Leap ---
    logging.info("Panel 3: Moran Tau-Leap")
    rev_tau = TauLeapReverse(g_max=g_max, gamma_mut=gamma_mut, kappa=kappa, bandwidth=bandwidth)
    x_tau = generate_moran(model, fwd, rev_tau, N, g_max, n_gen_points, device_obj)

    # --- 4. Moran Adjoint ---
    logging.info("Panel 4: Moran Adjoint Transport")
    rev_adj = AdjointReverse(g_max=g_max, gamma_mut=gamma_mut, kappa=kappa, bandwidth=bandwidth)
    x_adj = generate_moran(model, fwd, rev_adj, N, g_max, n_gen_points, device_obj)

    # --- Metrics ---
    m_count = compute_metrics(all_x0, x_count, value_range)
    m_tau = compute_metrics(all_x0, x_tau, value_range)
    m_adj = compute_metrics(all_x0, x_adj, value_range)

    logging.info(f"  Count Bridge:  MMD={m_count['mmd']:.4f}  W2={m_count['w2']:.2f}")
    logging.info(f"  Tau-Leap:      MMD={m_tau['mmd']:.4f}  W2={m_tau['w2']:.2f}")
    logging.info(f"  Adjoint:       MMD={m_adj['mmd']:.4f}  W2={m_adj['w2']:.2f}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    panels = [
        (x_gt,    "Ground Truth",                                    "steelblue"),
        (x_count, f"Count Bridge (Skellam)\nMMD={m_count['mmd']:.4f}, W₂={m_count['w2']:.2f}", "coral"),
        (x_tau,   f"Moran Tau-Leap\nMMD={m_tau['mmd']:.4f}, W₂={m_tau['w2']:.2f}",            "seagreen"),
        (x_adj,   f"Moran Adjoint\nMMD={m_adj['mmd']:.4f}, W₂={m_adj['w2']:.2f}",             "darkorchid"),
    ]

    for ax, (data, title, color) in zip(axes, panels):
        pts = data.astype(float)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.8, alpha=0.35, c=color, edgecolors='none')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-10, value_range + 10)
        ax.set_ylim(-10, value_range + 10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("y", fontsize=11)
    for ax in axes:
        ax.set_xlabel("x", fontsize=11)

    fig.suptitle("Discrete Two-Moons: Reverse Sampler Comparison (N=100, θ=2.0)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = Path("tex/figures/four_panel_comparison.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved figure to {save_path}")


if __name__ == "__main__":
    main()
