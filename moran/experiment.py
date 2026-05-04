"""
Moran particle model: N × θ sweep on discrete two moons.

Same data, same metrics as Count Bridges (Skellam) — the only difference
is the generative framework:

  Count Bridges: N=1 per sample, 50k samples, exact Skellam conditionals.
  Moran:         N>1 per sample, 50k/N samples, iterative denoising + coupling.

The 50,000 data points are partitioned into populations of size N.
Each population is corrupted via the Moran CTMC (random walk mutation +
kernel-weighted resampling) and the model learns to reverse this.

At generation time, sample from de Finetti prior DP(θ, Uniform(grid)),
denoise, aggregate all particles → should recover two moons.

Usage:
    python -m moran.experiment --n-particles 10 50 100 --thetas 1 2 5 10
    python -m moran.experiment --n-particles 50 --thetas 2 --epochs 500  # single run
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance

from datasets.discrete_moons import DiscreteMoonsDataset
from moran.two_moons.forward import ParticleMoranForward
from moran.two_moons.reverse import MoranReverseCTMC


# ---------------------------------------------------------------------------
# Model: DeepSets with energy score loss
# ---------------------------------------------------------------------------

class MoranDenoiser(nn.Module):
    """
    Permutation-equivariant denoiser for particle populations.

    DeepSets architecture:
      1. Per-particle encoder: (x_i / g_max, t_emb, noise_i) → h_i
      2. Global context: mean(h) → MLP → c
      3. Per-particle decoder: (h_i, c) → x̂_0_i

    Stochastic output via noise input, trained with energy score loss.
    """

    def __init__(self, hidden_dim=256, time_dim=32, noise_dim=16,
                 n_layers=3, m_samples=8, g_max=195):
        super().__init__()
        self.g_max = g_max
        self.noise_dim = noise_dim
        self.m_samples = m_samples
        self.time_dim = time_dim

        # Per-particle encoder
        enc_in = 2 + time_dim + noise_dim
        enc = []
        for i in range(n_layers):
            enc.extend([nn.Linear(enc_in if i == 0 else hidden_dim, hidden_dim), nn.SiLU()])
        self.encoder = nn.Sequential(*enc)

        # Global context (mean-pool → MLP)
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )

        # Per-particle decoder
        dec_in = hidden_dim * 2  # h_i concat context
        dec = []
        for i in range(n_layers):
            dec.extend([nn.Linear(dec_in if i == 0 else hidden_dim, hidden_dim), nn.SiLU()])
        dec.append(nn.Linear(hidden_dim, 2))
        self.decoder = nn.Sequential(*dec)

    def _time_embed(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float() * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x_t, t, noise=None):
        """
        x_t: [B, N, 2]  particle positions
        t:   [B]         time
        noise: [B, N, noise_dim] or None
        returns: [B, N, 2] predicted x_0
        """
        B, N, _ = x_t.shape
        if noise is None:
            noise = torch.randn(B, N, self.noise_dim, device=x_t.device)

        x_norm = x_t.float() / self.g_max
        t_emb = self._time_embed(t).unsqueeze(1).expand(B, N, -1)

        h = self.encoder(torch.cat([x_norm, t_emb, noise], dim=-1))   # [B, N, H]
        ctx = self.context_net(h.mean(dim=1))                          # [B, H]
        ctx = ctx.unsqueeze(1).expand(B, N, -1)                        # [B, N, H]
        out = self.decoder(torch.cat([h, ctx], dim=-1))                # [B, N, 2]
        return out * self.g_max

    def sample(self, x_t, t):
        """Deterministic inference generation yielding absolute coordinates."""
        return self.forward(x_t, t)

    def loss(self, target, x_t, t):
        """
        Pure Set-to-Set Topological Distance constraint via symmetric Chamfer matching.
        """
        preds = self.forward(x_t, t)  # [B, N, D]
        
        # --- Confinement: Permutation-Invariant Chamfer Distance ---
        dist_matrix = (preds.unsqueeze(2) - target.unsqueeze(1)).norm(dim=-1) # [B, N, N]
        min_prediction_to_target = dist_matrix.min(dim=2)[0] # [B, N]
        min_target_to_prediction = dist_matrix.min(dim=1)[0] # [B, N]
        
        # Bounded deterministic evaluation
        return (min_prediction_to_target.mean() + min_target_to_prediction.mean()) / 2.0


# ---------------------------------------------------------------------------
# Metrics (same as Count Bridges — main_mps.py)
# ---------------------------------------------------------------------------

def compute_metrics(x_real, x_gen, value_range):
    x_r = x_real.astype(np.float32)
    x_g = x_gen.astype(np.float32)

    # MMD with RBF kernel (bandwidth = value_range / 10)
    bw = value_range / 10
    Xr, Xg = torch.from_numpy(x_r), torch.from_numpy(x_g)
    rbf = lambda a, b: torch.exp(-torch.cdist(a, b).pow(2) / (2 * bw**2))
    mmd = (rbf(Xr, Xr).mean() + rbf(Xg, Xg).mean() - 2 * rbf(Xr, Xg).mean()).item()

    # Wasserstein (per dimension, averaged)
    w2 = np.mean([wasserstein_distance(x_r[:, d], x_g[:, d]) for d in range(2)])

    # Coverage and off-support
    real_set = set(map(tuple, x_real.astype(int)))
    gen_set = set(map(tuple, x_gen.astype(int)))
    off_support = 1.0 - len(gen_set & real_set) / max(len(gen_set), 1)
    coverage = len(real_set & gen_set) / max(len(real_set), 1)

    return {"mmd": mmd, "w2": w2, "off_support": off_support, "coverage": coverage}


# ---------------------------------------------------------------------------
# Training + generation for one (N, θ) configuration
# ---------------------------------------------------------------------------

def run_one(
    all_points: np.ndarray,
    n_particles: int,
    theta: float,
    g_max: int = 195,
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_denoise_steps: int = 20,
    gamma_mut: float = 5.0,
    bandwidth: float = 5.0,
    device: str = "cpu",
    ckpt_dir: Path = None,
    ckpt_prefix: str = "",
):
    """Train one Moran model and return generated points + metrics."""
    device_obj = torch.device(device)
    total_points = len(all_points)

    kappa = 2 * gamma_mut / theta if theta < 1000 else 0.0

    fwd = ParticleMoranForward(
        g_max=g_max, gamma_mut=gamma_mut, kappa=kappa,
        bandwidth=bandwidth, n_substeps=30,
    )

    model = MoranDenoiser(
        hidden_dim=256, time_dim=32, noise_dim=16,
        n_layers=3, m_samples=8, g_max=g_max,
    ).to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  N={n_particles}, θ={theta:.1f}, κ={kappa:.2f}, "
                 f"model={n_params:,} params, "
                 f"populations={total_points // n_particles}")

    # --- Training ---
    # Each step: sample B random populations of N points from the dataset
    steps_per_epoch = max(1, total_points // (n_particles * batch_size))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        for _ in range(steps_per_epoch):
            # Sample B populations
            x_0_batch = np.stack([
                all_points[np.random.choice(total_points, n_particles, replace=False)]
                for _ in range(batch_size)
            ])  # [B, N, 2]

            # Random time per population
            t_vals = np.random.uniform(0.01, 0.99, size=batch_size)

            # Forward Moran (per population — the forward is inherently serial)
            x_t_batch = np.stack([
                fwd.simulate(x_0_batch[b], float(t_vals[b]))
                for b in range(batch_size)
            ])  # [B, N, 2]

            x_0_t = torch.from_numpy(x_0_batch).float().to(device_obj)
            x_t_t = torch.from_numpy(x_t_batch).float().to(device_obj)
            t_t = torch.from_numpy(t_vals).float().to(device_obj)

            optimizer.zero_grad()
            loss = model.loss(x_0_t, x_t_t, t_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch

        if epoch % 100 == 0 or epoch == 1:
            logging.info(f"    Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}")
            if ckpt_dir is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"{ckpt_prefix}_epoch{epoch}.pt")

    if ckpt_dir is not None:
        torch.save(model.state_dict(), ckpt_dir / f"{ckpt_prefix}_final.pt")

    # --- Generation (batched) ---
    logging.info("  Generating...")
    model.eval()
    n_gen = total_points // n_particles
    gen_batch = min(64, n_gen)
    all_gen = []

    with torch.no_grad():
        for start in range(0, n_gen, gen_batch):
            B = min(gen_batch, n_gen - start)
            # Sample B populations from de Finetti prior
            x_t_batch = np.stack([fwd.sample_prior(n_particles) for _ in range(B)])

            rev_ctmc = MoranReverseCTMC(
                g_max=g_max, gamma_mut=gamma_mut, kappa=kappa,
                bandwidth=bandwidth, n_substeps=15
            )

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

    x_gen = np.concatenate(all_gen, axis=0)
    return x_gen, model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(x_real, results, value_range, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by N
    n_values = sorted(set(r["N"] for r in results.values()))
    theta_values = sorted(set(r["theta"] for r in results.values()))

    n_cols = len(theta_values) + 1  # +1 for ground truth column
    n_rows = len(n_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, N in enumerate(n_values):
        # Ground truth in first column
        ax = axes[row, 0]
        pts = x_real.astype(float)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, alpha=0.3, c='steelblue')
        ax.set_title("Ground Truth" if row == 0 else "")
        ax.set_ylabel(f"N={N}", fontsize=12)
        ax.set_xlim(-10, value_range + 10)
        ax.set_ylim(-10, value_range + 10)
        ax.set_aspect("equal")

        for col, theta in enumerate(theta_values):
            ax = axes[row, col + 1]
            key = f"N={N}_theta={theta}"
            if key in results:
                r = results[key]
                pts = r["x_gen"].astype(float)
                ax.scatter(pts[:, 0], pts[:, 1], s=0.5, alpha=0.3, c='coral')
                ax.set_title(f"θ={theta}\nMMD={r['mmd']:.4f}" if row == 0
                             else f"MMD={r['mmd']:.4f}")
            ax.set_xlim(-10, value_range + 10)
            ax.set_ylim(-10, value_range + 10)
            ax.set_aspect("equal")

    fig.suptitle("Moran N×θ Sweep: Discrete Two Moons", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved plot to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Moran N×θ sweep on discrete two moons")
    parser.add_argument("--n-particles", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--thetas", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma-mut", type=float, default=5.0)
    parser.add_argument("--bandwidth", type=float, default=5.0)
    parser.add_argument("--dataset-size", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/moran_sweep")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(output_path / "train.log"),
            logging.StreamHandler()
        ]
    )
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data — exact same config as Count Bridges paper
    dataset = DiscreteMoonsDataset(
        size=args.dataset_size,
        value_range=196, scale=30.0, offset=80.0, noise=0.1,
    )
    all_x0 = np.stack([dataset[i]['x_0'].numpy() for i in range(len(dataset))])
    value_range = dataset.value_range
    g_max = value_range - 1

    logging.info(f"Dataset: {len(all_x0)} points, value_range={value_range}")
    logging.info(f"Sweep: N={args.n_particles}, θ={args.thetas}")

    results = {}

    for N in args.n_particles:
        for theta in args.thetas:
            key = f"N={N}_theta={theta}"
            logging.info(f"\n{'='*60}")
            logging.info(f"  {key}")
            logging.info(f"{'='*60}")

            x_gen, _ = run_one(
                all_x0, N, theta,
                g_max=g_max,
                epochs=args.epochs,
                batch_size=min(args.batch_size, max(1, len(all_x0) // (N * 4))),
                lr=args.lr,
                gamma_mut=args.gamma_mut,
                bandwidth=args.bandwidth,
                device=args.device,
                ckpt_dir=output_path / "checkpoints",
                ckpt_prefix=f"N{N}_theta{theta}",
            )

            metrics = compute_metrics(all_x0, x_gen, value_range)
            results[key] = {**metrics, "N": N, "theta": theta, "x_gen": x_gen}

            logging.info(f"  MMD={metrics['mmd']:.4f}  W2={metrics['w2']:.2f}  "
                         f"off={metrics['off_support']:.3f}  cov={metrics['coverage']:.3f}")

            # Save plot incrementally
            plot_sweep(all_x0, results, value_range, output_path / "sweep.png")

            # Save metrics incrementally (without the large x_gen arrays)
            metrics_only = {
                k: {kk: vv for kk, vv in v.items() if kk != "x_gen"}
                for k, v in results.items()
            }
            with open(output_path / "sweep_results.json", "w") as f:
                json.dump(metrics_only, f, indent=2)

    # Summary table
    logging.info(f"\n{'='*60}")
    logging.info(f"  RESULTS SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"  {'Config':<25s}  {'MMD':>8s}  {'W2':>6s}  {'Off':>6s}  {'Cov':>6s}")
    logging.info(f"  {'-'*25}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
    for key in sorted(results.keys()):
        r = results[key]
        logging.info(f"  {key:<25s}  {r['mmd']:8.4f}  {r['w2']:6.2f}  "
                     f"{r['off_support']:6.3f}  {r['coverage']:6.3f}")

    logging.info(f"\nAll saved to {output_path}/")


if __name__ == "__main__":
    main()
