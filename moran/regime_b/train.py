"""
Regime B: Train and evaluate grid-count Moran model on discrete two moons.

Built on top of Count Bridges:
  - Uses DiscreteMoonsDataset (same data as their experiments)
  - Skellam bridge for per-cell birth-death (exact conditionals)
  - Moran resampling for inter-cell spatial coupling (new)
  - Compares: pure Skellam (kappa=0) vs Moran (kappa>0)

The state is a count vector c ∈ Z_≥0^G where G = grid_size^2.
Each data point's (x,y) position is binned into a grid cell.

Usage:
    python -m moran.regime_b.train [--epochs 500] [--grid-size 32]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.discrete_moons import DiscreteMoonsDataset
from bridges.numpy.skellam import SkellamBridge
from bridges.numpy.slack_samplers import BesselM
from bridges.numpy.utils import dlpack_backend

from .forward import GridMoranForward
from .bridge import GridMoranBridge


# ---------------------------------------------------------------------------
# Grid-level denoiser: MLP on flattened count vector
# ---------------------------------------------------------------------------

class GridDenoiser(nn.Module):
    """
    Denoiser for count vectors: (c_t, t) -> c_0_hat.

    Input: flattened count vector [B, G] + time embedding
    Output: predicted clean counts [B, G]
    """

    def __init__(self, grid_size: int = 32, hidden_dim: int = 512,
                 n_layers: int = 4, time_dim: int = 32, noise_dim: int = 16,
                 m_samples: int = 8):
        super().__init__()
        import math
        self.G = grid_size ** 2
        self.noise_dim = noise_dim
        self.m_samples = m_samples

        # Time embedding
        self.time_dim = time_dim

        layers = []
        in_dim = self.G + time_dim + noise_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.G))
        layers.append(nn.Softplus())  # non-negative counts
        self.net = nn.Sequential(*layers)

    def _time_embed(self, t):
        import math
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x_t, t, noise=None):
        if noise is None:
            noise = torch.randn(x_t.shape[0], self.noise_dim, device=x_t.device)
        t_emb = self._time_embed(t)
        inp = torch.cat([x_t, t_emb, noise], dim=-1)
        return self.net(inp)

    def sample(self, x_t, t, **kwargs):
        preds = []
        for _ in range(self.m_samples):
            noise = torch.randn(x_t.shape[0], self.noise_dim, device=x_t.device)
            preds.append(self.forward(x_t, t, noise))
        return torch.stack(preds, dim=1).mean(dim=1)

    def loss(self, target, inputs):
        x_t = inputs['x_t']
        t = inputs['t']
        B = x_t.shape[0]
        preds = []
        for _ in range(self.m_samples):
            noise = torch.randn(B, self.noise_dim, device=x_t.device)
            preds.append(self.forward(x_t, t, noise))
        preds = torch.stack(preds, dim=1)  # [B, m, G]

        target_exp = target.unsqueeze(1).expand_as(preds)
        conf = (preds - target_exp).norm(dim=-1).mean(dim=1)

        m = self.m_samples
        sq = preds.pow(2).sum(-1)
        inn = torch.bmm(preds, preds.transpose(1, 2))
        sqd = (sq.unsqueeze(2) + sq.unsqueeze(1) - 2 * inn).clamp(min=1e-6).sqrt()
        mask = 1.0 - torch.eye(m, device=preds.device)
        inter = (sqd * mask).sum(dim=(1, 2)) / (m * (m - 1))

        return (conf - 0.5 * inter).mean()


# ---------------------------------------------------------------------------
# Dataset adapter: convert per-point data to grid count vectors
# ---------------------------------------------------------------------------

def points_to_grid_dataset(
    point_dataset: DiscreteMoonsDataset,
    grid_size: int,
    n_samples_per_hist: int = 500,
    n_hists: int = 100,
):
    """
    Convert the per-point DiscreteMoonsDataset to grid count vectors.

    Each "sample" is a pair of count vectors:
      x_0: target histogram (two moons binned to grid)
      x_1: source histogram (8-Gaussians binned to grid)
    """
    vr = point_dataset.value_range
    scale = grid_size / vr
    G = grid_size ** 2

    x0_hists, x1_hists = [], []
    for _ in range(n_hists):
        # Sample N points and bin them
        c0 = np.zeros(G, dtype=np.int32)
        c1 = np.zeros(G, dtype=np.int32)
        for _ in range(n_samples_per_hist):
            idx = np.random.randint(len(point_dataset))
            sample = point_dataset[idx]
            # Target
            x0 = sample['x_0'].numpy()
            gi = int(np.clip(x0[0] * scale, 0, grid_size - 1))
            gj = int(np.clip(x0[1] * scale, 0, grid_size - 1))
            c0[gi * grid_size + gj] += 1
            # Source
            x1 = sample['x_1'].numpy()
            gi = int(np.clip(x1[0] * scale, 0, grid_size - 1))
            gj = int(np.clip(x1[1] * scale, 0, grid_size - 1))
            c1[gi * grid_size + gj] += 1

        x0_hists.append(torch.from_numpy(c0).int())
        x1_hists.append(torch.from_numpy(c1).int())

    return list(zip(x0_hists, x1_hists))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 500,
    grid_size: int = 32,
    kappa: float = 2.0,
    bandwidth: float = 3.0,
    n_hists: int = 100,
    n_points_per_hist: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
    output_dir: str = "outputs/regime_b",
):
    device_obj = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    point_dataset = DiscreteMoonsDataset(size=10000)
    hist_data = points_to_grid_dataset(
        point_dataset, grid_size, n_points_per_hist, n_hists
    )

    # Bridges
    bessel = BesselM(lam_p=32.0, lam_m=32.0, markov=True)
    skellam = SkellamBridge(
        n_steps=20, slack_sampler=bessel,
        schedule_type="linear", backend="torch", device=0,
    )
    grid_fwd = GridMoranForward(grid_size=grid_size, kappa=kappa, bandwidth=bandwidth)

    # Two bridges to compare
    bridges = {
        "Skellam (kappa=0)": SkellamBridge(
            n_steps=20, slack_sampler=bessel,
            schedule_type="linear", backend="torch", device=0,
        ),
        f"Moran (kappa={kappa})": GridMoranBridge(
            skellam=skellam, grid_forward=grid_fwd, kappa=kappa,
        ),
    }

    results = {}

    for bridge_name, bridge in bridges.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"  {bridge_name}")
        logging.info(f"{'='*60}")

        model = GridDenoiser(grid_size=grid_size, hidden_dim=512, n_layers=4)
        model = model.to(device_obj)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        logging.info(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

        # Train
        losses = []
        for epoch in range(1, epochs + 1):
            model.train()
            c0, c1 = hist_data[np.random.randint(len(hist_data))]
            c0 = c0.unsqueeze(0).to(device_obj)  # [1, G]
            c1 = c1.unsqueeze(0).to(device_obj)

            bridge_out = bridge(x_0=c0, x_1=c1)
            t = bridge_out["inputs"]["t"].to(device_obj)
            x_t = bridge_out["inputs"]["x_t"].to(device_obj)
            target = bridge_out["output"].to(device_obj)

            optimizer.zero_grad()
            loss = model.loss(target, {"x_t": x_t, "t": t})
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            if epoch % 100 == 0 or epoch == 1:
                logging.info(f"    Epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}")

        # Generate
        logging.info("  Generating...")
        model.eval()
        n_gen = 20
        gen_hists = []

        with torch.no_grad():
            for _ in range(n_gen):
                c1, _ = hist_data[np.random.randint(len(hist_data))]
                c1_np = c1.numpy().reshape(1, -1)

                result = bridge.sampler(x_1=c1_np, z={}, model=model)
                if isinstance(result, tuple):
                    result = result[0]
                if isinstance(result, torch.Tensor):
                    result = result.cpu().numpy()
                gen_hists.append(np.maximum(result.reshape(grid_size, grid_size).round(), 0))

        # Evaluate
        real_hists = [grid_fwd.counts_to_image(h[0].numpy())
                      for h in [hist_data[i] for i in range(min(n_gen, len(hist_data)))]]

        gen_avg = np.mean(gen_hists, axis=0)
        real_avg = np.mean(real_hists, axis=0)

        eps = 1e-10
        g_flat = gen_avg.flatten() + eps
        r_flat = real_avg.flatten() + eps
        g_flat /= g_flat.sum()
        r_flat /= r_flat.sum()
        kl = float(np.sum(r_flat * np.log(r_flat / g_flat)))

        results[bridge_name] = {"kl": kl, "final_loss": losses[-1]}
        logging.info(f"  KL={kl:.4f}  final_loss={losses[-1]:.4f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bridges = len(results)
    fig, axes = plt.subplots(1, 1 + n_bridges, figsize=(5 * (1 + n_bridges), 5))

    axes[0].imshow(real_avg.T, origin='lower', cmap='viridis')
    axes[0].set_title("Real (avg histogram)")

    for i, (name, metrics) in enumerate(results.items()):
        axes[i + 1].imshow(gen_avg.T if i == n_bridges - 1 else gen_avg.T,
                           origin='lower', cmap='viridis')
        axes[i + 1].set_title(f"{name}\nKL={metrics['kl']:.3f}")

    fig.suptitle(f"Regime B: Grid Counts ({grid_size}x{grid_size})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path / "regime_b_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(output_path / "metrics.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    logging.info(f"All saved to {output_path}/")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--bandwidth", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/regime_b")
    args = parser.parse_args()

    train(**vars(args))
