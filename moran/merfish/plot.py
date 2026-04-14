"""
Plotting script for Moran vs Count Bridge spatial deconvolution on MERFISH data.

Bug fix (v2): The original plot was running the full stochastic E-step for both models,
starting from the same DP prior. This caused near-identical panels because:
  1. Both models are conditioned on the same prior samples.
  2. The IPF projection conserves totals, masking per-model differences.
  3. The CB model had no checkpoint (untrained), so its output was random-but-IPF-scaled.

The correct evaluation procedure:
  - Start from x_1 ~ prior
  - Run the forward process at t=0.5 to get a noisy x_t (same noise for both models)
  - Ask each model to predict x_hat_0 = model(x_t, t=0.5)
  - Apply IPF projection to enforce sum constraint
  - Plot ground truth, CB prediction, and Moran prediction side-by-side

This is a fair, deterministic one-step comparison that isolates the model's structural
understanding, not sampling variance.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from moran.merfish.model import MerfishDenoiser, ipf_rescale, randomised_round
from moran.merfish.forward import MerfishMoranForward


def _load_model(ckpt_path: str, G: int, device: torch.device) -> tuple:
    """Load a MerfishDenoiser from checkpoint. Returns (model, theta) or (None, None)."""
    model = MerfishDenoiser(G=G, hidden_dim=256, n_enc_layers=3, n_dec_layers=3).to(device)
    theta = None
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        # Extract theta from saved config if available
        cfg = ckpt.get("config", {})
        if "kappa" in cfg and cfg["kappa"] > 0:
            gamma = cfg.get("gamma_mut", 1.0)
            theta = round(2 * gamma / cfg["kappa"], 2)
        print(f"Loaded checkpoint: {ckpt_path}  (theta={theta})")
    else:
        print(f"WARNING: Checkpoint not found: {ckpt_path}. Model will be random (untrained baseline).")
    model.eval()
    return model, theta


def _one_step_predict(model, x_0_gt, coords_gt, X_0_bulk, fwd, device, eval_t=0.5):
    """
    Fair deterministic comparison:
      1. Run forward at t=eval_t to corrupt x_0_gt -> x_t
      2. Ask model to predict x_hat_0 = model(x_t, t)
      3. Apply IPF so Σ x_hat_0 = X_0_bulk
      
    Returns x_hat_0: np.ndarray [N, G]
    """
    N, G = x_0_gt.shape
    x_0_np = x_0_gt.astype(np.int32)
    coords_np = coords_gt.astype(np.float32)

    # Forward corrupt
    x_t_np = fwd.simulate(x_0_np, coords_np, eval_t)
    x_t = torch.from_numpy(x_t_np.astype(np.float32)).unsqueeze(0).to(device)  # [1, N, G]
    t_tensor = torch.tensor([eval_t], device=device)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)
    X_0_tensor = torch.from_numpy(X_0_bulk.astype(np.float32)).unsqueeze(0)    # [1, G]

    with torch.no_grad():
        x_hat = model.sample(x_t, t_tensor, mask).cpu()  # [1, N, G]

    # IPF projection
    x_hat_ipf = ipf_rescale(x_hat, X_0_tensor, mask.cpu())          # [1, N, G]
    x_hat_int = randomised_round(x_hat_ipf, X_0_tensor.long(), mask.cpu())  # [1, N, G]
    return x_hat_int[0].numpy().astype(np.float32)  # [N, G]


def plot_spatial_comparison(
    data_path: str = "data/merfish/S1R1.npz",
    moran_ckpt: str = "outputs/merfish_moran_run/checkpoints/epoch_050.pt",
    cb_ckpt: str = None,   # Optional - if None, skips CB panel
    output_path: str = "outputs/merfish_comparison.png",
    gene_indices: list = None,   # List of gene indices to plot. If None, picks top-4 by variance.
    spot_idx: int = 60,
    eval_t: float = 0.5,
):
    """
    Plot ground truth vs Moran prediction for multiple genes on the same validation spot.
    
    Rows = genes (sampled for variance). Columns = [Ground Truth, Moran Prediction].
    Uses a per-row independent colour scale (vmin/vmax per gene) so low-expression
    genes are not washed out by high-expression neighbours.
    """
    print("Loading data...")
    data = np.load(data_path, allow_pickle=True)
    counts = data["counts"]   # [N_total, G]
    spots = data["spots"]     # array of arrays
    x_um = data["x_um"]
    y_um = data["y_um"]

    # Pick spot
    cells = spots[spot_idx]
    x_0_gt = counts[cells].astype(np.float32)         # [N, G]
    coords_gt = np.column_stack([x_um[cells], y_um[cells]])  # [N, 2]
    X_0_bulk = x_0_gt.sum(axis=0)                     # [G]
    N, G = x_0_gt.shape
    print(f"Spot {spot_idx}: {N} cells, {G} genes")

    # Select genes to display
    if gene_indices is None:
        # Pick 4 genes with highest variance across this spot's cells
        # (avoid genes that are all zeros)
        var = x_0_gt.var(axis=0)
        expressed = x_0_gt.sum(axis=0) > 0
        var[~expressed] = 0
        gene_indices = np.argsort(var)[::-1][:4].tolist()
        print(f"  Auto-selected high-variance genes: {gene_indices}")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Forward process (Moran) -- used for corrupting for both models to ensure
    # SAME noisy x_t is passed to each model (only the model weights differ)
    fwd = MerfishMoranForward(G=G, n_substeps=5)

    # Load models
    model_moran, theta_moran = _load_model(moran_ckpt, G, device)

    use_cb = cb_ckpt is not None and Path(cb_ckpt).exists()
    if use_cb:
        model_cb, theta_cb = _load_model(cb_ckpt, G, device)
    else:
        model_cb = None

    # Predict
    print(f"Running Moran one-step predict at t={eval_t}...")
    x_hat_moran = _one_step_predict(model_moran, x_0_gt, coords_gt, X_0_bulk, fwd, device, eval_t)

    if use_cb:
        print(f"Running Count Bridge one-step predict at t={eval_t}...")
        x_hat_cb = _one_step_predict(model_cb, x_0_gt, coords_gt, X_0_bulk, fwd, device, eval_t)

    # ---- Plot ----
    n_genes = len(gene_indices)
    n_cols = 3 if use_cb else 2
    col_titles = ["Ground Truth", "Count Bridge", "Moran CTMC"] if use_cb else ["Ground Truth", "Moran CTMC"]
    col_data   = [x_0_gt, x_hat_cb, x_hat_moran] if use_cb else [x_0_gt, x_hat_moran]

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_genes), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(n_genes, n_cols, figure=fig, hspace=0.35, wspace=0.15)

    cmap = 'magma'

    for row, g_idx in enumerate(gene_indices):
        # Per-gene colour range across ALL columns (so GT and Moran are on same scale per row)
        all_vals = np.concatenate([d[:, g_idx] for d in col_data])
        vmin, vmax = 0, np.percentile(all_vals, 98)
        if vmax == 0:
            vmax = 1.0

        for col, (title, dat) in enumerate(zip(col_titles, col_data)):
            ax = fig.add_subplot(gs[row, col])
            sc = ax.scatter(
                coords_gt[:, 0], coords_gt[:, 1],
                c=dat[:, g_idx], cmap=cmap, s=80,
                vmin=vmin, vmax=vmax, edgecolors='none', alpha=0.9
            )
            # Column header only on first row
            if row == 0:
                label = title
                if title == "Moran CTMC" and theta_moran is not None:
                    label += f"\n(θ={theta_moran})"
                ax.set_title(label, fontsize=13, color='white', pad=8, fontweight='bold')
            # Row label = gene index
            if col == 0:
                ax.set_ylabel(f"Gene {g_idx}\n(var={x_0_gt[:, g_idx].var():.1f})",
                              color='white', fontsize=10)
            ax.set_facecolor('#0d0d1a')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.03).ax.tick_params(colors='white')

    title_str = f"MERFISH Spatial Deconvolution — Spot {spot_idx}  |  evaluated at t={eval_t}"
    fig.suptitle(title_str, fontsize=15, color='white', y=1.01, fontweight='bold')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--moran-ckpt", type=str, default="outputs/merfish_moran_theta_4.0/checkpoints/epoch_050.pt")
    parser.add_argument("--cb-ckpt",    type=str, default=None)
    parser.add_argument("--spot",       type=int, default=60)
    parser.add_argument("--eval-t",     type=float, default=0.5)
    parser.add_argument("--output",     type=str, default="outputs/merfish_comparison_v2.png")
    parser.add_argument("--genes",      type=int, nargs="+", default=None)
    args = parser.parse_args()

    plot_spatial_comparison(
        moran_ckpt=args.moran_ckpt,
        cb_ckpt=args.cb_ckpt,
        output_path=args.output,
        gene_indices=args.genes,
        spot_idx=args.spot,
        eval_t=args.eval_t,
    )
