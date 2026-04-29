"""
Quantitative evaluation script: compute distributional metrics for each model checkpoint
vs ground truth on the held-out validation set.

Metrics (per spot, then averaged):
  - MSE         : mean squared error per gene per cell
  - MMD (RBF)   : maximum mean discrepancy with median-bandwidth kernel
  - Energy dist : 2*E[d(X,Y)] - E[d(X,X)] - E[d(Y,Y)]  (unbiased)
  - Chamfer     : symmetric nearest-neighbour distance  (same loss used in training)

Outputs a LaTeX-ready table to stdout and saves results to outputs/metrics_table.json.

Usage:
    python -m moran.merfish.eval_metrics
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

from moran.merfish.dataset import MerfishMoranDataset, merfish_collate_fn
from moran.merfish.model import MerfishDenoiser, ipf_rescale, randomised_round
from moran.merfish.forward import MerfishMoranForward
from torch.utils.data import DataLoader, random_split

# ── metrics ───────────────────────────────────────────────────────────────────

def mse(pred, true):
    return float(np.mean((pred.astype(np.float32) - true.astype(np.float32)) ** 2))

def mmd_rbf(pred, true, subsample=300):
    """Maximum Mean Discrepancy with RBF kernel.
    Bandwidth: median pairwise distance on a small combined sample
    (standard heuristic; does NOT use a fixed seed so each call is independent).
    """
    rng = np.random.default_rng()   # non-deterministic — each call independent
    n = min(subsample, len(pred), len(true))
    P = pred[rng.choice(len(pred), n, replace=False)].astype(np.float32)
    T = true[rng.choice(len(true), n, replace=False)].astype(np.float32)

    # Bandwidth: median pairwise distance on a small combined sample
    # This is the standard "median heuristic" for RBF kernels.
    combined = np.concatenate([P[:50], T[:50]], axis=0)   # 100 points max
    combined_t = torch.from_numpy(combined)
    pairwise = torch.cdist(combined_t, combined_t)         # [100, 100]
    # Use upper triangle only (no diagonal)
    iu = torch.triu_indices(len(combined), len(combined), offset=1)
    bw = float(pairwise[iu[0], iu[1]].median().clamp(min=1.0))

    Pp = torch.from_numpy(P); Tt = torch.from_numpy(T)
    rbf = lambda a, b: torch.exp(-torch.cdist(a, b).pow(2) / (2 * bw ** 2))
    return (rbf(Pp, Pp).mean() + rbf(Tt, Tt).mean() - 2 * rbf(Pp, Tt).mean()).item()

def energy_dist(pred, true, subsample=200):
    rng = np.random.default_rng(0)
    n = min(subsample, len(pred), len(true))
    P = pred[rng.choice(len(pred), n, replace=False)].astype(np.float32)
    T = true[rng.choice(len(true), n, replace=False)].astype(np.float32)
    Dpp = cdist(P, P).mean()
    Dtt = cdist(T, T).mean()
    Dpt = cdist(P, T).mean()
    return float(2 * Dpt - Dpp - Dtt)

def chamfer(pred, true):
    """Symmetric Chamfer distance (same as training loss)."""
    P = torch.from_numpy(pred.astype(np.float32))
    T = torch.from_numpy(true.astype(np.float32))
    D = torch.cdist(P, T)  # [N, M]
    return (D.min(dim=1).values.mean() + D.min(dim=0).values.mean()).item() / 2.0

# ── one-step prediction ───────────────────────────────────────────────────────

def one_step_predict(model, x_0_gt, coords, X_0_bulk, fwd, device, t=0.5):
    """Corrupt x_0_gt at time t, query model, IPF-project back."""
    N, G = x_0_gt.shape
    x_t_np = fwd.simulate(x_0_gt.astype(np.int32), coords.astype(np.float32), t)
    x_t = torch.from_numpy(x_t_np.astype(np.float32)).unsqueeze(0).to(device)
    t_tensor = torch.tensor([t], device=device)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)
    X_0_tensor = torch.from_numpy(X_0_bulk.astype(np.float32)).unsqueeze(0)

    coords_tensor = torch.from_numpy(coords.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        x_hat = model.sample(x_t, t_tensor, mask, coords=coords_tensor).cpu()
    x_hat_ipf = ipf_rescale(x_hat, X_0_tensor, mask.cpu())
    x_hat_int = randomised_round(x_hat_ipf, X_0_tensor.long(), mask.cpu())
    return x_hat_int[0].numpy().astype(np.float32)

# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate_model(ckpt_path, dataset, val_indices, fwd, device, t=0.5, label="", use_coords=True,
                   hidden_dim=256, n_enc_layers=3, n_dec_layers=3, noise_dim=32):
    G = dataset.G
    
    # Pre-read the config to dynamically set model size
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        config = ckpt.get("config", {})
        hidden_dim = config.get("hidden_dim", hidden_dim)
        n_enc_layers = config.get("n_enc_layers", n_enc_layers)
        n_dec_layers = config.get("n_dec_layers", n_dec_layers)
        noise_dim = config.get("noise_dim", noise_dim)
    else:
        ckpt = None

    model = MerfishDenoiser(G=G, hidden_dim=hidden_dim, n_enc_layers=n_enc_layers, 
                            n_dec_layers=n_dec_layers, noise_dim=noise_dim, use_coords=use_coords).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded: {ckpt_path}")
    else:
        print(f"  WARNING: {ckpt_path} not found — using untrained model")
    model.eval()

    all_pred, all_true = [], []

    for idx in val_indices:
        spot = dataset[idx]
        x_0_gt  = spot["counts"].numpy().astype(np.float32)   # [N, G]
        coords   = spot["coords"].numpy().astype(np.float32)   # [N, 2]
        X_0_bulk = spot["X_0"].numpy().astype(np.float32)      # [G]

        pred = one_step_predict(model, x_0_gt, coords, X_0_bulk, fwd, device, t)
        all_pred.append(pred)
        all_true.append(x_0_gt)

    all_pred = np.concatenate(all_pred, axis=0)  # [total_cells, G]
    all_true = np.concatenate(all_true, axis=0)

    return {
        "label":       label,
        "n_spots":     len(val_indices),
        "n_cells":     len(all_pred),
        "mse":         round(mse(all_pred, all_true), 4),
        "mmd":         round(mmd_rbf(all_pred, all_true), 5),
        "energy_dist": round(energy_dist(all_pred, all_true), 4),
        "chamfer":     round(chamfer(all_pred, all_true), 4),
    }

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="data/merfish")
    parser.add_argument("--npz-name",  default="S1R1.npz")
    parser.add_argument("--device",    default="mps")
    parser.add_argument("--eval-t",    type=float, default=0.5)
    parser.add_argument("--out-json",  default="outputs/metrics_table.json")
    parser.add_argument("--sweep-dir", default=None,
                        help="If provided, override Moran checkpoint paths to look "
                             "inside this sweep directory (e.g. outputs/sweep_gene_level_XXXXX)")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or
             (args.device == "mps" and torch.backends.mps.is_available())) else "cpu")
    print(f"Device: {device}")

    dataset = MerfishMoranDataset(data_dir=args.data_dir, npz_name=args.npz_name,
                                  max_cells_per_spot=60, min_cells_per_spot=3)
    # Reproduce the exact same val split used in training (seed=42)
    rng = torch.Generator().manual_seed(42)
    train_size = int(0.85 * len(dataset))
    val_size   = len(dataset) - train_size
    _, val_ds  = random_split(dataset, [train_size, val_size], generator=rng)
    val_indices = list(val_ds.indices)
    print(f"Val set: {len(val_indices)} spots")

    fwd = MerfishMoranForward(G=dataset.G, n_substeps=5)

    if args.sweep_dir:
        sd = args.sweep_dir.rstrip("/")
        configs = [
            ("Count Bridge (Baseline)",         "outputs/experiment_ambitious_cb_20260425_223936/merfish_cb_baseline/checkpoints/epoch_040.pt", True),
            ("Moran θ=1.0 (Strong Drift)",     f"{sd}/merfish_moran_theta_1.0/checkpoints/epoch_060.pt", True),
            ("Moran θ=2.0 (Biological Limit)",  f"{sd}/merfish_moran_theta_2.0/checkpoints/epoch_060.pt", True),
            ("Moran θ=4.0 (Moderate Drift)",    f"{sd}/merfish_moran_theta_4.0/checkpoints/epoch_060.pt", True),
            ("Moran θ=10.0 (Weak Drift)",       f"{sd}/merfish_moran_theta_10.0/checkpoints/epoch_060.pt", True),
        ]
    else:
        configs = [
            ("Count Bridge (Independent Skellam)", "outputs/merfish_cb_run_50/checkpoints/epoch_050.pt", False),
            ("Moran θ=1.0 (Strong Drift)",         "outputs/merfish_moran_theta_1.0/checkpoints/epoch_060.pt", True),
            ("Moran θ=2.0 (Biological Limit)",     "outputs/merfish_moran_theta_2.0/checkpoints/epoch_060.pt", True),
            ("Moran θ=4.0 (Moderate Drift)",       "outputs/merfish_moran_theta_4.0/checkpoints/epoch_060.pt", True),
            ("Moran θ=10.0 (Weak Drift)",           "outputs/merfish_moran_theta_10.0/checkpoints/epoch_060.pt", True),
        ]

    results = []
    for label, ckpt, use_coords in configs:
        print(f"\nEvaluating: {label}")
        t0 = time.time()
        row = evaluate_model(ckpt, dataset, val_indices, fwd, device, t=args.eval_t, label=label, use_coords=use_coords)
        row["eval_time_s"] = round(time.time() - t0, 1)
        results.append(row)
        print(f"  MSE={row['mse']:.4f}  MMD={row['mmd']:.5f}  "
              f"Energy={row['energy_dist']:.4f}  Chamfer={row['chamfer']:.4f}")

    # Save JSON
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_json}")

    # Print LaTeX table
    print("\n% ── LaTeX table ──────────────────────────────────────────────")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(r"Model & MSE $\downarrow$ & MMD $\downarrow$ & Energy $\downarrow$ & Chamfer $\downarrow$ \\ \midrule")
    for r in results:
        print(f"{r['label']} & {r['mse']} & {r['mmd']} & {r['energy_dist']} & {r['chamfer']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Quantitative evaluation on held-out validation spots.}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
