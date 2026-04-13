"""
Generation-only script for Adjoint Flux Matching evaluation.

Loads pre-trained checkpoints and runs ONLY backward generation
with the new Adjoint Coalescent Transport reverse sampler.
No retraining required.

Usage:
    .venv/bin/python -m moran.generate_only \
        --ckpt-dir outputs/moran_sweep_Nall/checkpoints \
        --n-particles 10 100 \
        --thetas 1.0 2.0 5.0 10.0 \
        --output-dir outputs/adjoint_test
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
from moran.regime_a.forward import ParticleMoranForward
from moran.regime_a.reverse import MoranReverseCTMC
from moran.experiment import MoranDenoiser, compute_metrics, plot_sweep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def generate_from_checkpoint(
    all_points: np.ndarray,
    n_particles: int,
    theta: float,
    ckpt_path: Path,
    g_max: int = 195,
    n_denoise_steps: int = 20,
    gamma_mut: float = 5.0,
    bandwidth: float = 5.0,
    device: str = "cpu",
):
    """Load a checkpoint and run generation only (no training)."""
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

    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location=device_obj, weights_only=True)
    model.load_state_dict(state_dict)
    logging.info(f"  Loaded checkpoint: {ckpt_path}")

    # --- Generation only ---
    logging.info("  Generating with Adjoint Coalescent Transport...")
    model.eval()
    n_gen = total_points // n_particles
    gen_batch = min(64, n_gen)
    all_gen = []

    with torch.no_grad():
        for start in range(0, n_gen, gen_batch):
            B = min(gen_batch, n_gen - start)
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
    return x_gen


def main():
    parser = argparse.ArgumentParser(description="Generation-only with Adjoint Flux Matching")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Directory containing pre-trained checkpoints")
    parser.add_argument("--n-particles", nargs="+", type=int, default=[10, 100])
    parser.add_argument("--thetas", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--output-dir", type=str, default="outputs/adjoint_test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--g-max", type=int, default=195)
    parser.add_argument("--gamma-mut", type=float, default=5.0)
    parser.add_argument("--bandwidth", type=float, default=5.0)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)

    # Load dataset
    g_max = args.g_max
    value_range = g_max + 1
    dataset = DiscreteMoonsDataset(
        size=50_000,
        value_range=value_range, scale=30.0, offset=80.0, noise=0.1,
    )
    all_x0 = np.stack([dataset[i]['x_0'].numpy() for i in range(len(dataset))])

    logging.info(f"Dataset: {len(all_x0)} points, value_range={value_range}")
    logging.info(f"Checkpoint dir: {ckpt_dir}")
    logging.info(f"Sweep: N={args.n_particles}, θ={args.thetas}")

    results = {}

    for N in args.n_particles:
        for theta in args.thetas:
            key = f"N={N}_theta={theta}"

            # Find checkpoint
            ckpt_file = ckpt_dir / f"N{N}_theta{theta}_final.pt"
            if not ckpt_file.exists():
                logging.warning(f"  SKIP {key}: checkpoint not found at {ckpt_file}")
                continue

            logging.info(f"\n{'='*60}")
            logging.info(f"  {key}")
            logging.info(f"{'='*60}")

            x_gen = generate_from_checkpoint(
                all_x0, N, theta,
                ckpt_path=ckpt_file,
                g_max=g_max,
                gamma_mut=args.gamma_mut,
                bandwidth=args.bandwidth,
                device=args.device,
            )

            metrics = compute_metrics(all_x0, x_gen, value_range)
            results[key] = {**metrics, "N": N, "theta": theta, "x_gen": x_gen}

            logging.info(f"  MMD={metrics['mmd']:.4f}  W2={metrics['w2']:.2f}  "
                         f"off={metrics['off_support']:.3f}  cov={metrics['coverage']:.3f}")

            # Save plot incrementally
            plot_sweep(all_x0, results, value_range, output_path / "adjoint_sweep.png")

            # Save metrics incrementally
            metrics_only = {
                k: {kk: vv for kk, vv in v.items() if kk != "x_gen"}
                for k, v in results.items()
            }
            with open(output_path / "adjoint_sweep_results.json", "w") as f:
                json.dump(metrics_only, f, indent=2)

    # Summary table
    logging.info(f"\n{'='*60}")
    logging.info(f"  ADJOINT FLUX MATCHING - RESULTS SUMMARY")
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
