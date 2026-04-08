"""
Regime A: Train and evaluate the particle Moran model on discrete two moons.

Completely standalone — no Count Bridge dependency.

Usage:
    python -m moran.regime_a.train [--epochs 500] [--n-particles 200]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons

from .forward import ParticleMoranForward
from .model import ParticleDenoiser


# ---------------------------------------------------------------------------
# Data: discrete two moons as a particle configuration
# ---------------------------------------------------------------------------

def sample_particle_config(
    n_particles: int, g_max: int = 127,
    scale: float = 35.0, offset: float = 50.0, noise: float = 0.05,
) -> np.ndarray:
    """
    Sample N particle positions from discretized two moons.
    returns: [N, 2] integer positions on {0, ..., g_max}^2
    """
    X, _ = make_moons(n_samples=n_particles, noise=noise)
    X = X - np.array([0.5, 0.25])
    X_int = np.round(np.clip(X * scale + offset, 0, g_max)).astype(np.int32)
    return X_int


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    epochs: int = 500,
    n_particles: int = 200,
    n_configs: int = 50,
    g_max: int = 127,
    gamma_mut: float = 5.0,
    kappa: float = 2.0,
    bandwidth: float = 5.0,
    lr: float = 1e-3,
    device: str = "cpu",
    output_dir: str = "outputs/regime_a",
):
    device_obj = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fwd = ParticleMoranForward(
        g_max=g_max, gamma_mut=gamma_mut, kappa=kappa,
        bandwidth=bandwidth, n_substeps=30,
    )
    model = ParticleDenoiser(hidden_dim=256, time_dim=32, n_layers=3, g_max=g_max)
    model = model.to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    logging.info(f"Regime A: {n_particles} particles, g_max={g_max}, "
                 f"theta={fwd.theta:.1f}, kappa={kappa}, gamma={gamma_mut}")
    logging.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Pre-generate particle configurations (batches of N particles)
    configs = [sample_particle_config(n_particles, g_max) for _ in range(n_configs)]

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()

        # Pick a random configuration
        x_0 = configs[np.random.randint(n_configs)]

        # Sample time
        t_val = np.random.uniform(0.01, 0.99)

        # Simulate Moran forward
        x_t = fwd.simulate(x_0, t_val)

        # Convert to tensors: [1, N, 2] (batch of 1 configuration)
        x_0_t = torch.from_numpy(x_0).float().unsqueeze(0).to(device_obj)
        x_t_t = torch.from_numpy(x_t).float().unsqueeze(0).to(device_obj)
        t_t = torch.tensor([t_val], device=device_obj)

        # Predict x_0 from x_t
        x_0_pred = model(x_t_t, t_t)

        # Loss: MSE on particle positions
        loss = F.mse_loss(x_0_pred, x_0_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if epoch % 100 == 0 or epoch == 1:
            logging.info(f"  Epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}")

    # --- Generation ---
    logging.info("Generating from de Finetti prior...")
    model.eval()

    generated_configs = []
    n_gen_configs = 20
    n_steps = 20

    with torch.no_grad():
        for _ in range(n_gen_configs):
            # Start from de Finetti prior
            x_t = fwd.sample_prior(n_particles)

            for k in range(n_steps, 0, -1):
                t_val = k / n_steps
                x_t_tensor = torch.from_numpy(x_t).float().unsqueeze(0).to(device_obj)
                t_tensor = torch.tensor([t_val], device=device_obj)

                x_0_hat = model(x_t_tensor, t_tensor).squeeze(0).cpu().numpy()
                x_0_hat = np.clip(np.round(x_0_hat), 0, g_max).astype(np.int32)

                t_next = (k - 1) / n_steps
                if t_next > 0.01:
                    x_t = fwd.simulate(x_0_hat, t_next)
                else:
                    x_t = x_0_hat

            generated_configs.append(x_t)

    # --- Evaluation ---
    # Aggregate all generated particles into a histogram and compare to data
    gen_all = np.concatenate(generated_configs, axis=0)
    real_all = np.concatenate(configs[:n_gen_configs], axis=0)

    gen_hist = fwd.to_histogram(gen_all)
    real_hist = fwd.to_histogram(real_all)

    # Metrics on histograms
    g = g_max + 1
    eps = 1e-10
    rh = real_hist.flatten().astype(float) + eps
    gh = gen_hist.flatten().astype(float) + eps
    rh /= rh.sum(); gh /= gh.sum()
    kl = float(np.sum(rh * np.log(rh / gh)))

    real_set = set(zip(*np.where(real_hist > 0)))
    gen_set = set(zip(*np.where(gen_hist > 0)))
    coverage = len(real_set & gen_set) / max(len(real_set), 1)
    off_support = 1.0 - len(gen_set & real_set) / max(len(gen_set), 1)

    logging.info(f"  KL={kl:.4f}  coverage={coverage:.3f}  off_support={off_support:.3f}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Real config
    pts = real_all.astype(float)
    axes[0, 0].scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3, c='steelblue')
    axes[0, 0].set_title("Real (particles)")
    axes[0, 0].set_xlim(0, g_max); axes[0, 0].set_ylim(0, g_max)
    axes[0, 0].set_aspect("equal")

    # Generated config
    pts = gen_all.astype(float)
    axes[0, 1].scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3, c='coral')
    axes[0, 1].set_title("Generated (particles)")
    axes[0, 1].set_xlim(0, g_max); axes[0, 1].set_ylim(0, g_max)
    axes[0, 1].set_aspect("equal")

    # Prior sample
    prior = fwd.sample_prior(n_particles * n_gen_configs)
    axes[0, 2].scatter(prior[:, 0], prior[:, 1], s=1, alpha=0.3, c='gray')
    axes[0, 2].set_title(f"De Finetti prior (theta={fwd.theta:.1f})")
    axes[0, 2].set_xlim(0, g_max); axes[0, 2].set_ylim(0, g_max)
    axes[0, 2].set_aspect("equal")

    # Histograms
    axes[1, 0].imshow(real_hist.T, origin='lower', cmap='viridis')
    axes[1, 0].set_title("Real histogram")
    axes[1, 1].imshow(gen_hist.T, origin='lower', cmap='viridis')
    axes[1, 1].set_title(f"Generated histogram\nKL={kl:.3f}")
    axes[1, 2].imshow(np.abs(real_hist - gen_hist).T, origin='lower', cmap='hot')
    axes[1, 2].set_title("Absolute difference")

    fig.suptitle(f"Regime A: Particle Moran (N={n_particles}, theta={fwd.theta:.1f}, "
                 f"kappa={kappa})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path / "regime_a_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save metrics
    metrics = {
        "regime": "A", "n_particles": n_particles, "theta": fwd.theta,
        "kappa": kappa, "gamma_mut": gamma_mut,
        "kl": kl, "coverage": coverage, "off_support": off_support,
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(model.state_dict(), output_path / "model.pt")
    logging.info(f"All saved to {output_path}/")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--gamma-mut", type=float, default=5.0)
    parser.add_argument("--bandwidth", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/regime_a")
    args = parser.parse_args()

    train(**vars(args))
