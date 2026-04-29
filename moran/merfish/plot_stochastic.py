"""
Plot script v3: Full stochastic E-step evaluation.

We restore the full adjoint reverse continuous-time Markov chain from t=1 -> 0.
A one-step expected value prediction E[X_0 | X_{0.5}] collapses to the spot average
due to permutation uncertainty (symmetry) at high noise scales. To recover the
discrete clumps and distinct cell profiles, we must sequentially break symmetry
via stochastic Langevin-like dynamics (the Adjoint Reverse CTMC).
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from moran.merfish.model import MerfishDenoiser
from moran.merfish.forward import MerfishMoranForward
from moran.merfish.reverse import MerfishFastAdjointReverse
from moran.merfish.experiment import e_step
from moran.merfish.count_bridge import MerfishCountBridgeForward, MerfishCountBridgeReverse

def _load_model(ckpt_path: str, G: int, device: torch.device, use_coords: bool = True) -> tuple:
    theta = None
    cfg = {}
    # Defaults for old checkpoints that don't store architecture config
    hidden_dim, n_enc_layers, n_dec_layers, noise_dim = 256, 3, 3, 32
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model_state"]
        cfg = ckpt.get("config", {})
        # Read architecture from saved config (preferred)
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
        n_enc_layers = cfg.get("n_enc_layers", n_enc_layers)
        n_dec_layers = cfg.get("n_dec_layers", n_dec_layers)
        noise_dim = cfg.get("noise_dim", noise_dim)
        if "kappa" in cfg and cfg["kappa"] > 0:
            gamma = cfg.get("gamma_mut", 1.0)
            theta = round(2 * gamma / cfg["kappa"], 2)
        model = MerfishDenoiser(G=G, hidden_dim=hidden_dim, n_enc_layers=n_enc_layers,
                                n_dec_layers=n_dec_layers, noise_dim=noise_dim,
                                use_coords=use_coords).to(device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}  (theta={theta}, hidden={hidden_dim}, enc={n_enc_layers}, dec={n_dec_layers})")
    else:
        model = MerfishDenoiser(G=G, hidden_dim=hidden_dim, n_enc_layers=n_enc_layers,
                                n_dec_layers=n_dec_layers, noise_dim=noise_dim,
                                use_coords=use_coords).to(device)
        print(f"WARNING: Checkpoint not found: {ckpt_path}. Model will be random.")
    model.eval()
    return model, theta, cfg

def plot_spatial_comparison_full_traj(
    data_path: str = "data/merfish/S1R1.npz",
    moran_ckpt: str = "outputs/merfish_moran_theta_4.0/checkpoints/epoch_050.pt",
    cb_ckpt: str = "outputs/merfish_cb_run_50/checkpoints/epoch_020.pt",
    output_path: str = "outputs/merfish_comparison_stochastic.png",
    gene_indices: list = None,
    spot_idx: int = 60,
    no_cb: bool = False,
):
    print("Loading data...")
    data = np.load(data_path, allow_pickle=True)
    counts = data["counts"]
    spots = data["spots"]
    x_um = data["x_um"]
    y_um = data["y_um"]

    cells = spots[spot_idx]
    x_0_gt = counts[cells].astype(np.float32)
    coords_gt = np.column_stack([x_um[cells], y_um[cells]])
    X_0_bulk = x_0_gt.sum(axis=0)
    N, G = x_0_gt.shape
    print(f"Spot {spot_idx}: {N} cells, {G} genes")

    if gene_indices is None:
        var = x_0_gt.var(axis=0)
        expressed = x_0_gt.sum(axis=0) > 0
        var[~expressed] = 0
        gene_indices = np.argsort(var)[::-1][:4].tolist()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_moran, theta, cfg = _load_model(moran_ckpt, G, device)
    if not no_cb and cb_ckpt:
        model_cb, _, _ = _load_model(cb_ckpt, G, device, use_coords=True)
    else:
        model_cb = None

    gamma_mut = cfg.get("gamma_mut", 1.0)
    kappa = cfg.get("kappa", 0.5)
    h_kernel = cfg.get("h_kernel", 20.0)

    fwd = MerfishMoranForward(gamma_mut, kappa, h_kernel, G=G, n_substeps=5)
    rev = MerfishFastAdjointReverse(gamma_mut, kappa, h_kernel, n_substeps=5, device=device)
    
    cb_fwd = MerfishCountBridgeForward(G=G)
    cb_rev = MerfishCountBridgeReverse()

    # Pre-pad data batch
    batch = {
        "counts": torch.from_numpy(x_0_gt).unsqueeze(0).float(),
        "X_0": torch.from_numpy(X_0_bulk).unsqueeze(0).long(),
        "coords": torch.from_numpy(coords_gt).unsqueeze(0).float(),
        "n_cells": torch.tensor([len(cells)]),
        "mask": torch.ones(1, len(cells), dtype=torch.bool)
    }

    all_counts = counts.astype(np.float32)

    print("Running FULL Moran reverse CTMC trajectory...")
    with torch.no_grad():
        moran_out = e_step(model_moran, fwd, rev, batch, device, n_denoise_steps=10, ref_counts=all_counts)
        x_hat_moran = moran_out["x_0"][0][:N].cpu().numpy()

    if model_cb is not None and not no_cb:
        print("Running FULL Count Bridge reverse Skellam trajectory...")
        with torch.no_grad():
            cb_out = e_step(model_cb, cb_fwd, cb_rev, batch, device, n_denoise_steps=10, ref_counts=all_counts)
            x_hat_cb = cb_out["x_0"][0][:N].cpu().numpy()
    else:
        x_hat_cb = np.zeros_like(x_0_gt)

    # ---- Plot ----
    n_genes = len(gene_indices)
    n_cols = 3
    col_titles = ["Ground Truth", "Count Bridge Baseline", f"Moran Stoch (θ={theta})"]
    col_data   = [x_0_gt, x_hat_cb, x_hat_moran]

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_genes), facecolor='white')
    gs  = gridspec.GridSpec(n_genes, n_cols, figure=fig, hspace=0.35, wspace=0.15)
    cmap = 'viridis'

    for row, g_idx in enumerate(gene_indices):
        all_vals = np.concatenate([d[:, g_idx] for d in col_data])
        vmin, vmax = 0, np.percentile(all_vals, 98)
        if vmax <= 0: vmax = 1.0

        for col, (title, dat) in enumerate(zip(col_titles, col_data)):
            ax = fig.add_subplot(gs[row, col])
            sc = ax.scatter(
                coords_gt[:, 0], coords_gt[:, 1],
                c=dat[:, g_idx], cmap=cmap, s=80,
                vmin=vmin, vmax=vmax, edgecolors='black', linewidth=0.3, alpha=0.9
            )
            if row == 0:
                ax.set_title(title, fontsize=13, color='black', pad=8, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"Gene {g_idx}\n(var={x_0_gt[:, g_idx].var():.1f})",
                              color='black', fontsize=10)
            ax.set_facecolor('white')
            ax.tick_params(colors='black')
            for spine in ax.spines.values():
                spine.set_edgecolor('#ccc')
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.03).ax.tick_params(colors='black')

    title_str = f"MERFISH Spatial Deconvolution — Spot {spot_idx} (Full Stochastic Chain)"
    fig.suptitle(title_str, fontsize=15, color='black', y=1.01, fontweight='bold')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--moran-ckpt", required=True)
    parser.add_argument("--cb-ckpt", required=False, default="outputs/merfish_cb_run/checkpoints/epoch_020.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--spot-idx", type=int, default=60, help="Tissue spot geometry layout to plot.")
    parser.add_argument("--no-cb", action="store_true",
                        help="Skip Count Bridge baseline (useful when no valid CB checkpoint exists)")
    args = parser.parse_args()
    plot_spatial_comparison_full_traj(
        moran_ckpt=args.moran_ckpt,
        cb_ckpt=args.cb_ckpt,
        output_path=args.output,
        spot_idx=args.spot_idx,
        no_cb=args.no_cb,
    )
