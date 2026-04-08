"""
MPS-compatible main script for Count Bridges and Moran Bridge.

Bypasses CuPy entirely — uses numpy bridges + PyTorch MPS for training.
Runs the discrete moons experiment comparing Skellam vs Moran.

Usage:
    python main_mps.py                              # Skellam baseline
    python main_mps.py --bridge moran --kappa 2.0   # Moran bridge
    python main_mps.py --bridge cfm                 # CFM baseline
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets.discrete_moons import DiscreteMoonsDataset
from bridges.numpy.skellam import SkellamBridge
from bridges.numpy.moran import MoranBridge
from bridges.numpy.slack_samplers import BesselM


# ---------------------------------------------------------------------------
# Simple MLP (self-contained, no Hydra dependency)
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """MLP denoiser: (x_t, t, noise) -> predicted x_0."""

    def __init__(self, data_dim=2, hidden_dim=256, n_layers=4, noise_dim=16,
                 m_samples=16, act_fn="silu"):
        super().__init__()
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.m_samples = m_samples

        in_dim = data_dim + 1 + noise_dim  # x_t + t + noise
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU() if act_fn == "silu" else nn.Softplus())
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, noise=None, **kwargs):
        if noise is None:
            noise = torch.randn(x_t.shape[0], self.noise_dim, device=x_t.device)
        inp = torch.cat([x_t, t, noise], dim=-1)
        return self.net(inp)

    def sample(self, x_t, t, **kwargs):
        """Generate m_samples predictions and return mean."""
        preds = []
        for _ in range(self.m_samples):
            noise = torch.randn(x_t.shape[0], self.noise_dim, device=x_t.device)
            preds.append(self.forward(x_t, t, noise))
        return torch.stack(preds, dim=1).mean(dim=1)

    def loss(self, target, inputs):
        """Energy score loss (simplified)."""
        x_t = inputs['x_t']
        t = inputs['t']
        B = x_t.shape[0]

        # Generate m predictions with different noise
        preds = []
        for _ in range(self.m_samples):
            noise = torch.randn(B, self.noise_dim, device=x_t.device)
            preds.append(self.forward(x_t, t, noise))
        preds = torch.stack(preds, dim=1)  # [B, m, d]

        # Confinement: distance to target
        target_exp = target.unsqueeze(1).expand_as(preds)
        conf = (preds - target_exp).norm(dim=-1).mean(dim=1)  # [B]

        # Interaction: pairwise distances between predictions
        m = self.m_samples
        sq = preds.pow(2).sum(-1)  # [B, m]
        inn = torch.bmm(preds, preds.transpose(1, 2))  # [B, m, m]
        sqd = sq.unsqueeze(2) + sq.unsqueeze(1) - 2 * inn
        d = sqd.clamp(min=1e-6).sqrt()
        mask = 1.0 - torch.eye(m, device=d.device)
        inter = (d * mask).sum(dim=(1, 2)) / (m * (m - 1))  # [B]

        return (conf - 0.5 * inter).mean()


# ---------------------------------------------------------------------------
# CFM bridge (for baseline comparison)
# ---------------------------------------------------------------------------

class CFMBridge:
    """Simple continuous flow matching bridge (numpy-based, MPS-compatible)."""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x_0, x_1, t_target=None):
        x_0 = x_0.numpy().astype(np.float32)
        x_1 = x_1.numpy().astype(np.float32)
        B = x_0.shape[0]

        t = np.random.rand(B, 1).astype(np.float32)
        x_t = (1 - t) * x_0 + t * x_1 + self.sigma * np.random.randn(*x_0.shape).astype(np.float32)
        target = x_1 - x_0  # velocity

        return (
            torch.from_numpy(t),
            torch.from_numpy(x_t),
            torch.from_numpy(target),
        )

    def sampler(self, x_1, z, model, n_steps=10, **kwargs):
        """Euler ODE solver: x_1 -> x_0."""
        x = x_1.clone()
        dt = 1.0 / n_steps
        for k in range(n_steps, 0, -1):
            t = torch.full((x.shape[0], 1), k / n_steps, device=x.device)
            v = model.sample(x_t=x, t=t)
            x = x - v * dt  # integrate backward
        return x.round()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, bridge, dataset, epochs, batch_size, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in dataloader:
            x_0 = batch['x_0'].to(device)
            x_1 = batch['x_1'].to(device)

            bridge_out = bridge(x_0=x_0, x_1=x_1)

            # Handle both tuple (CuPy/CFM) and dict (numpy) bridge outputs
            if isinstance(bridge_out, dict):
                t = bridge_out['inputs']['t'].to(device)
                x_t = bridge_out['inputs']['x_t'].to(device)
                target = bridge_out['output'].to(device)
            else:
                t, x_t, target = bridge_out
                t = t.to(device)
                x_t = x_t.to(device)
                target = target.to(device)

            optimizer.zero_grad()
            loss = model.loss(target, {'x_t': x_t, 't': t})
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % 50 == 0 or epoch == 1:
            logging.info(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Sampling & Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, bridge, dataset, n_samples, n_steps, device):
    """Generate samples using the bridge's reverse process."""
    model.eval()
    # Numpy bridges produce CPU tensors; model must be on CPU for sampling
    model = model.to("cpu")

    # Sample source distribution x_1
    indices = np.random.choice(len(dataset), n_samples)
    x_1_list = [dataset[i]['x_1'] for i in indices]
    x_1 = torch.stack(x_1_list).float().to(device)

    if isinstance(bridge, CFMBridge):
        return bridge.sampler(x_1, {}, model, n_steps=n_steps).cpu().numpy()

    # For Skellam/Moran: use numpy sampler
    x_1_np = x_1.cpu().numpy()
    # Pass value_range for Moran's de Finetti prior
    extra_kwargs = {}
    if hasattr(bridge, 'sample_prior'):
        extra_kwargs['value_range'] = dataset.dataset.value_range if hasattr(dataset, 'dataset') else 128
    result = bridge.sampler(
        x_1=x_1_np, z={}, model=model,
        return_trajectory=False, return_x_hat=False,
        **extra_kwargs,
    )
    # Unwrap tuple from dlpack_backend
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, torch.Tensor):
        result = result.cpu().numpy()
    else:
        result = np.asarray(result)
    if result.ndim == 1:
        result = result.reshape(-1, 2)
    return result.astype(np.float32)


def compute_metrics(x_real, x_gen, value_range):
    """Compute evaluation metrics."""
    from scipy.stats import wasserstein_distance

    x_r = x_real.astype(np.float32)
    x_g = x_gen.astype(np.float32)

    # MMD
    bw = value_range / 10
    Xr = torch.from_numpy(x_r)
    Xg = torch.from_numpy(x_g)
    def rbf(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * bw**2))
    mmd = (rbf(Xr, Xr).mean() + rbf(Xg, Xg).mean() - 2 * rbf(Xr, Xg).mean()).item()

    # Energy distance (W2 per dimension, averaged)
    w2 = np.mean([wasserstein_distance(x_r[:, d], x_g[:, d]) for d in range(x_r.shape[1])])

    # Off-support rate
    real_set = set(map(tuple, x_real.astype(int)))
    gen_set = set(map(tuple, x_gen.astype(int)))
    off_support = 1.0 - len(gen_set & real_set) / max(len(gen_set), 1)
    coverage = len(real_set & gen_set) / max(len(real_set), 1)

    return {"mmd": mmd, "w2": w2, "off_support": off_support, "coverage": coverage}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(x_real, results, value_range, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 1 + len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    def plot_one(ax_scatter, ax_hist, data, title, color):
        pts = data.astype(float)
        ax_scatter.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.4, c=color)
        ax_scatter.set_title(title, fontsize=10)
        ax_scatter.set_xlim(-10, value_range + 10)
        ax_scatter.set_ylim(-10, value_range + 10)
        ax_scatter.set_aspect("equal")
        H = np.zeros((value_range, value_range))
        for x, y in data.astype(int):
            if 0 <= x < value_range and 0 <= y < value_range:
                H[x, y] += 1
        ax_hist.imshow(H.T, origin='lower', cmap='viridis', aspect='auto')
        ax_hist.set_title("Grid density")

    plot_one(axes[0, 0], axes[1, 0], x_real, "Ground Truth", "steelblue")
    colors = ["coral", "seagreen", "darkorchid", "crimson", "dodgerblue"]
    for i, (name, data, metrics) in enumerate(results):
        title = f"{name}\nMMD={metrics['mmd']:.4f} W2={metrics['w2']:.2f}"
        plot_one(axes[0, i+1], axes[1, i+1], data, title, colors[i % len(colors)])

    fig.suptitle("Discrete Two Moons: Bridge Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved plot to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MPS-compatible Count/Moran Bridge")
    parser.add_argument("--bridge", type=str, default="all",
                        choices=["skellam", "moran", "cfm", "all"])
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--bandwidth", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--device", type=str, default="mps",
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", type=str, default="outputs/mps_moons")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = DiscreteMoonsDataset(size=args.dataset_size)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Get real target samples for evaluation
    x_real = np.stack([dataset[i]['x_0'].numpy() for i in range(min(5000, len(dataset)))])

    # Bridges to run
    bessel = BesselM(lam_p=32.0, lam_m=32.0, markov=True)
    bridges_to_run = {}

    if args.bridge in ("skellam", "all"):
        bridges_to_run["Skellam (Count Bridge)"] = SkellamBridge(
            n_steps=args.n_steps, slack_sampler=bessel,
            schedule_type="linear", backend="torch", device=0,
        )
    if args.bridge in ("moran", "all"):
        bridges_to_run[f"Moran (kappa={args.kappa})"] = MoranBridge(
            n_steps=args.n_steps, slack_sampler=bessel,
            kappa=args.kappa, bandwidth=args.bandwidth,
            schedule_type="linear", backend="torch", device=0,
        )
    if args.bridge in ("cfm", "all"):
        bridges_to_run["CFM (OT Flow)"] = CFMBridge(sigma=0.1)

    # Train and evaluate each bridge
    results = []
    all_metrics = {}

    for name, bridge in bridges_to_run.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"  {name}")
        logging.info(f"{'='*60}")

        model = SimpleMLP(
            data_dim=dataset.data_dim, hidden_dim=256, n_layers=4,
            noise_dim=16, m_samples=16,
        )

        losses = train(model, bridge, train_dataset, args.epochs,
                       args.batch_size, args.lr, device)

        logging.info(f"  Generating {args.n_samples} samples...")
        x_gen = generate(model, bridge, eval_dataset, args.n_samples,
                         args.n_steps, device)

        metrics = compute_metrics(x_real, x_gen, dataset.value_range)
        all_metrics[name] = metrics
        results.append((name, x_gen, metrics))

        logging.info(f"  MMD={metrics['mmd']:.4f}  W2={metrics['w2']:.2f}  "
                     f"off_support={metrics['off_support']:.3f}  "
                     f"coverage={metrics['coverage']:.3f}")

    # Save comparison plot
    plot_comparison(x_real, results, dataset.value_range, output_dir / "comparison.png")

    # Save metrics
    serializable = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_metrics.items()}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)

    logging.info(f"\n{'='*60}")
    logging.info(f"  RESULTS")
    logging.info(f"{'='*60}")
    for name, metrics in all_metrics.items():
        logging.info(f"  {name:30s}  MMD={metrics['mmd']:.4f}  W2={metrics['w2']:.2f}  "
                     f"off={metrics['off_support']:.3f}")
    logging.info(f"\nAll saved to {output_dir}/")


if __name__ == "__main__":
    main()
