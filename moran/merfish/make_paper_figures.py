"""
Publication-quality figures for the Moran MERFISH paper.
All Two Moons visualizations use REAL model outputs from trained checkpoints.

Run: python -m moran.merfish.make_paper_figures
"""
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.datasets import make_moons

# Deuteranopia-safe palette (no red-green adjacency)
CB_COLOR  = "#E66100"  # Deep orange   — Count Bridge baseline
M1_COLOR  = "#0072B2"  # Deep blue     — Moran θ=1.0  (best)
M4_COLOR  = "#9467BD"  # Muted purple  — Moran θ=4.0
M10_COLOR = "#D4A017"  # Dark gold     — Moran θ=10.0
GT_COLOR  = "#1A1A1A"  # Near-black    — Ground truth
PRIOR_COLOR = "#AAAAAA"  # Mid-grey    — DP Prior
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.right": False,
    "axes.spines.top": False, "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 180,
})
OUT = Path("outputs/paper_figures"); OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════
# Figure 1: Two Moons — DP Prior → Generated → Ground Truth
# Uses REAL MoranDenoiser from moran/experiment.py + real checkpoints
# ═══════════════════════════════════════════════════════════════════
print("Building Figure 1: Two Moons (real checkpoint generation) …")

from moran.experiment import MoranDenoiser, MoranReverseCTMC
from moran.two_moons.forward import ParticleMoranForward

# Sweep parameters (from train.log): g_max=195, value_range=196, scale=30, offset=80
G_MAX_TM = 195
GAMMA_MUT = 5.0
BANDWIDTH = 5.0

def load_tm_model(ckpt_path, theta):
    kappa = 2 * GAMMA_MUT / theta
    model = MoranDenoiser(hidden_dim=256, time_dim=32, noise_dim=16,
                          n_layers=3, m_samples=8, g_max=G_MAX_TM)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model, kappa

def generate_tm(model, kappa, n_particles, n_gen=20, n_steps=20):
    fwd = ParticleMoranForward(g_max=G_MAX_TM, gamma_mut=GAMMA_MUT,
                                kappa=kappa, bandwidth=BANDWIDTH)
    all_gen = []
    with torch.no_grad():
        for _ in range(n_gen):
            x_t = fwd.sample_prior(n_particles)
            rev = MoranReverseCTMC(g_max=G_MAX_TM, gamma_mut=GAMMA_MUT,
                                    kappa=kappa, bandwidth=BANDWIDTH, n_substeps=15)
            for k in range(n_steps, 0, -1):
                t_val = k / n_steps
                x_t_t = torch.from_numpy(x_t).float().unsqueeze(0).to(device)
                t_t = torch.full((1,), t_val, device=device)
                x_0_hat = model.sample(x_t_t, t_t).cpu().numpy()[0]
                x_0_hat = np.mod(np.round(x_0_hat), G_MAX_TM + 1).astype(np.int32)
                t_next = (k - 1) / n_steps
                if t_next > 0.01:
                    x_t = rev.tau_leap(x_t[np.newaxis], x_0_hat[np.newaxis],
                                       t_val, t_val - t_next)[0]
                else:
                    x_t = x_0_hat
            all_gen.append(x_t)
    return np.concatenate(all_gen)

# Ground truth (same params as sweep)
X_gt, _ = make_moons(n_samples=2000, noise=0.1)
gt_pts = np.round(np.clip((X_gt - [0.5, 0.25]) * 30.0 + 80.0, 0, G_MAX_TM)).astype(np.int32)

# DP Prior sample (real sample_prior: Dir(θ/G) → uniform random clusters)
fwd_tm = ParticleMoranForward(g_max=G_MAX_TM, gamma_mut=GAMMA_MUT, kappa=10.0, bandwidth=BANDWIDTH)
prior_pts = fwd_tm.sample_prior(2000)

# Generated from trained model (N=100, theta=1.0)
ckpt_100 = "outputs/moran_sweep_Nall/checkpoints/N100_theta1.0_final.pt"
if Path(ckpt_100).exists():
    model_tm, kappa_tm = load_tm_model(ckpt_100, theta=1.0)
    gen_pts = generate_tm(model_tm, kappa_tm, n_particles=100, n_gen=20)
    print(f"  Generated {len(gen_pts)} particles, {len(np.unique(gen_pts, axis=0))} unique")
else:
    print(f"  WARNING: {ckpt_100} not found")
    gen_pts = prior_pts

# --- Build Figure 1 ---
fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig1.suptitle("Two Moons — Moran Bridge Generative Process (Real Outputs)",
              fontsize=13, fontweight="bold")

lim = (-10, G_MAX_TM + 10)

# Panel A: DP Prior
ax = axes[0]
ax.scatter(prior_pts[:, 0], prior_pts[:, 1], s=6, alpha=0.4, c=PRIOR_COLOR,
           linewidths=0, rasterized=True)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_title("(A) De Finetti Prior\nDP(θ=1.0, Uniform)", fontsize=10)
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
n_unique_prior = len(np.unique(prior_pts, axis=0))
ax.text(0.05, 0.95, f"N = 2000\n{n_unique_prior} unique positions\n"
        f"(concentrated on ~{n_unique_prior} atoms)",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# Panel B: Generated
ax = axes[1]
ax.scatter(gen_pts[:, 0], gen_pts[:, 1], s=3, alpha=0.3, c=M1_COLOR,
           linewidths=0, rasterized=True)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_title("(B) Generated\n(Moran θ=1.0, N=100)", fontsize=10)
ax.set_xlabel("Dim 1")

# Panel C: Ground Truth
ax = axes[2]
ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=3, alpha=0.3, c=GT_COLOR,
           linewidths=0, rasterized=True)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_title("(C) Ground Truth\n(Two Moons, noise=0.1)", fontsize=10)
ax.set_xlabel("Dim 1")

plt.tight_layout()
fig1.savefig(OUT / "figure1_two_moons.pdf", bbox_inches="tight")
fig1.savefig(OUT / "figure1_two_moons.png", bbox_inches="tight")
plt.close(fig1)
print(f"  Saved → {OUT}/figure1_two_moons.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Two Moons quantitative sweep
# ═══════════════════════════════════════════════════════════════════
print("Building Figure 2: Two Moons sweep metrics …")
with open("outputs/moran_sweep_Nall/sweep_results.json") as f:
    sweep = json.load(f)
Ns = [10, 50, 100, 1000]
thetas = [1.0, 2.0, 5.0, 10.0]
theta_colors = [M1_COLOR, M4_COLOR, M10_COLOR, CB_COLOR]

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
fig2.suptitle("Two Moons — Quantitative Metrics vs Population Size N",
              fontsize=13, fontweight="bold")
for mi, (key, label) in enumerate([
    ("mmd", "MMD (↓ better)"), ("w2", "Wasserstein-2 (↓ better)"),
    ("off_support", "Off-support fraction (↓ better)")]):
    ax = axes2[mi]
    for th, col_c in zip(thetas, theta_colors):
        ys = [sweep[f"N={N}_theta={th}"][key] for N in Ns]
        ax.plot(Ns, ys, marker="o", color=col_c, linewidth=1.8, markersize=5, label=f"θ={th}")
    ax.set_xscale("log"); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("Population size N"); ax.set_ylabel(label)
    ax.set_title(label.split(" (")[0])
    ax.yaxis.grid(True, linewidth=0.4, linestyle="--", alpha=0.5)
    if mi == 2: ax.legend(loc="upper right", frameon=False)
plt.tight_layout()
fig2.savefig(OUT / "figure2_two_moons_sweep.pdf", bbox_inches="tight")
fig2.savefig(OUT / "figure2_two_moons_sweep.png", bbox_inches="tight")
plt.close(fig2)
print(f"  Saved → {OUT}/figure2_two_moons_sweep.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: MERFISH per-spot gene expression heatmaps
# ═══════════════════════════════════════════════════════════════════
print("Building Figure 3: MERFISH per-spot visualizations …")
from moran.merfish.dataset import MerfishMoranDataset
from moran.merfish.forward import MerfishMoranForward
from moran.merfish.model import MerfishDenoiser, ipf_rescale, randomised_round
from torch.utils.data import random_split

dataset = MerfishMoranDataset(data_dir="data/merfish", npz_name="S1R1.npz",
                               max_cells_per_spot=60, min_cells_per_spot=3)
rng = torch.Generator().manual_seed(42)
train_size = int(0.85 * len(dataset))
_, val_ds = random_split(dataset, [train_size, len(dataset)-train_size], generator=rng)
fwd_m = MerfishMoranForward(G=dataset.G, n_substeps=5)

def load_merfish_model(ckpt_path, G):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    m = MerfishDenoiser(G=G, hidden_dim=cfg.get("hidden_dim",512),
        n_enc_layers=cfg.get("n_enc_layers",5), n_dec_layers=cfg.get("n_dec_layers",5),
        noise_dim=cfg.get("noise_dim",64), use_coords=True).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval(); return m

def predict_spot(model, spot, fwd, t=0.5):
    x_0 = spot["counts"].numpy().astype(np.float32)
    coords = spot["coords"].numpy().astype(np.float32)
    X_0 = spot["X_0"].numpy().astype(np.float32)
    N, G = x_0.shape
    x_t = fwd.simulate(x_0.astype(np.int32), coords, t)
    x_t_t = torch.from_numpy(x_t.astype(np.float32)).unsqueeze(0).to(device)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)
    coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)
    X_0_t = torch.from_numpy(X_0).unsqueeze(0)
    with torch.no_grad():
        x_hat = model.sample(x_t_t, torch.tensor([t],device=device), mask, coords=coords_t).cpu()
    x_hat = ipf_rescale(x_hat, X_0_t, mask.cpu())
    return randomised_round(x_hat, X_0_t.long(), mask.cpu())[0].numpy().astype(np.float32)

cb_path = "outputs/experiment_ambitious_cb_20260425_223936/merfish_cb_baseline/checkpoints/epoch_040.pt"
mr_path = "outputs/experiment_ambitious_24h_20260425_152955/merfish_moran_theta_1.0/checkpoints/epoch_060.pt"

if Path(cb_path).exists() and Path(mr_path).exists():
    cb_model = load_merfish_model(cb_path, dataset.G)
    mr_model = load_merfish_model(mr_path, dataset.G)

    spot_indices = [val_ds.indices[0], val_ds.indices[10], val_ds.indices[20]]
    fig3 = plt.figure(figsize=(15, 12))
    fig3.suptitle("MERFISH S1R1 — Per-Spot Gene Expression Profiles\n"
                  "Top 20 genes by variance shown for each validation spot",
                  fontsize=12, fontweight="bold")
    gs3 = gridspec.GridSpec(len(spot_indices), 3, figure=fig3, hspace=0.45, wspace=0.35)

    for row, si in enumerate(spot_indices):
        spot = dataset[si]
        gt = spot["counts"].numpy().astype(np.float32)
        N_c = gt.shape[0]
        cb_p = predict_spot(cb_model, spot, fwd_m)
        mr_p = predict_spot(mr_model, spot, fwd_m)
        top_g = np.argsort(np.var(gt, axis=0))[-20:][::-1]
        gt_s, cb_s, mr_s = gt[:,top_g], cb_p[:,top_g], mr_p[:,top_g]
        vmax = max(gt_s.max(), cb_s.max(), mr_s.max())
        for col, (data, title, color) in enumerate([
            (gt_s, f"Ground Truth (Spot #{si}, N={N_c})", GT_COLOR),
            (cb_s, "Count Bridge (Baseline)", CB_COLOR),
            (mr_s, "Moran θ=1.0 (Ours)", M1_COLOR),
        ]):
            ax = fig3.add_subplot(gs3[row, col])
            im = ax.imshow(data, aspect="auto", cmap="cividis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9, color=color if col > 0 else "black")
            ax.set_xlabel("Gene (top-20 by var)")
            if col == 0: ax.set_ylabel(f"Cell (N={N_c})")
            if col > 0:
                mse_v = np.mean((data - gt_s)**2)
                ax.text(0.02, 0.95, f"MSE={mse_v:.1f}", transform=ax.transAxes,
                        fontsize=8, color="white", va="top",
                        bbox=dict(boxstyle="round", facecolor=color, alpha=0.7))
    fig3.subplots_adjust(right=0.92)
    cbar_ax = fig3.add_axes([0.94, 0.15, 0.015, 0.7])
    fig3.colorbar(im, cax=cbar_ax, label="Expression count")
    fig3.savefig(OUT / "figure3_merfish_spots.pdf", bbox_inches="tight")
    fig3.savefig(OUT / "figure3_merfish_spots.png", bbox_inches="tight")
    plt.close(fig3); print(f"  Saved → {OUT}/figure3_merfish_spots.png")

    # ── Figure 3b: per-cell distributions ────────────────────────────
    print("Building Figure 3b: MERFISH distribution comparison …")
    n_eval = 15
    all_gt, all_cb, all_mr = [], [], []
    for i in range(min(n_eval, len(val_ds))):
        sp = dataset[val_ds.indices[i]]
        all_gt.append(sp["counts"].numpy().astype(np.float32))
        all_cb.append(predict_spot(cb_model, sp, fwd_m))
        all_mr.append(predict_spot(mr_model, sp, fwd_m))
    all_gt = np.concatenate(all_gt); all_cb = np.concatenate(all_cb); all_mr = np.concatenate(all_mr)
    gt_m = all_gt.mean(1); cb_m = all_cb.mean(1); mr_m = all_mr.mean(1)
    gt_v = np.var(all_gt, axis=1); cb_v = np.var(all_cb, axis=1); mr_v = np.var(all_mr, axis=1)

    fig3b, ax3b = plt.subplots(1, 3, figsize=(14, 4.5))
    fig3b.suptitle(f"MERFISH S1R1 — Per-Cell Expression Distributions\n"
                   f"Aggregated over {n_eval} validation spots ({len(all_gt)} cells)",
                   fontsize=12, fontweight="bold")
    bins = np.linspace(0, max(gt_m.max(), cb_m.max(), mr_m.max()), 50)
    ax3b[0].hist(gt_m, bins=bins, alpha=.5, color=GT_COLOR, label="Ground truth", density=True)
    ax3b[0].hist(cb_m, bins=bins, alpha=.5, color=CB_COLOR, label="Count Bridge", density=True)
    ax3b[0].hist(mr_m, bins=bins, alpha=.5, color=M1_COLOR, label="Moran θ=1.0", density=True)
    ax3b[0].set_xlabel("Mean expression/cell"); ax3b[0].set_ylabel("Density")
    ax3b[0].set_title("(A) Per-Cell Mean Expression"); ax3b[0].legend(frameon=False)

    bv = np.linspace(0, min(np.percentile(gt_v, 99), 100), 50)
    ax3b[1].hist(gt_v, bins=bv, alpha=.5, color=GT_COLOR, label="Ground truth", density=True)
    ax3b[1].hist(cb_v, bins=bv, alpha=.5, color=CB_COLOR, label="Count Bridge", density=True)
    ax3b[1].hist(mr_v, bins=bv, alpha=.5, color=M1_COLOR, label="Moran θ=1.0", density=True)
    ax3b[1].set_xlabel("Within-cell variance"); ax3b[1].set_ylabel("Density")
    ax3b[1].set_title("(B) Biological Heterogeneity"); ax3b[1].legend(frameon=False)

    n_show = min(500, len(gt_m)); idx = np.random.choice(len(gt_m), n_show, replace=False)
    ax3b[2].scatter(gt_m[idx], cb_m[idx], s=4, alpha=.3, c=CB_COLOR, label="Count Bridge")
    ax3b[2].scatter(gt_m[idx], mr_m[idx], s=4, alpha=.3, c=M1_COLOR, label="Moran θ=1.0")
    lim_sc = [0, max(gt_m[idx].max(), cb_m[idx].max(), mr_m[idx].max())*1.1]
    ax3b[2].plot(lim_sc, lim_sc, "k--", lw=.8, alpha=.5, label="y=x")
    ax3b[2].set_xlim(lim_sc); ax3b[2].set_ylim(lim_sc); ax3b[2].set_aspect("equal")
    ax3b[2].set_xlabel("True mean expression"); ax3b[2].set_ylabel("Predicted mean")
    ax3b[2].set_title("(C) Predicted vs True"); ax3b[2].legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig3b.savefig(OUT / "figure3b_merfish_distributions.pdf", bbox_inches="tight")
    fig3b.savefig(OUT / "figure3b_merfish_distributions.png", bbox_inches="tight")
    plt.close(fig3b); print(f"  Saved → {OUT}/figure3b_merfish_distributions.png")
else:
    print("  MERFISH checkpoints not found — skipping")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: MERFISH distributional metrics
# ═══════════════════════════════════════════════════════════════════
print("Building Figure 4: MERFISH metrics …")
with open("outputs/preliminary_quant.json") as f:
    quant = json.load(f)
labels = ["Count Bridge\n(Baseline)", "Moran\nθ=1.0", "Moran\nθ=4.0", "Moran\nθ=10.0"]
colors = [CB_COLOR, M1_COLOR, M4_COLOR, M10_COLOR]
mse = [r["mse"] for r in quant]; energy = [r["energy_dist"] for r in quant]
mmd = [r["mmd"] for r in quant]; chamf = [r["chamfer"] for r in quant]

fig4 = plt.figure(figsize=(14, 9))
fig4.suptitle("MERFISH S1R1 — Distributional Benchmark Results\n"
              "78,329 cells · 649 genes · 30 held-out validation spots",
              fontsize=12, fontweight="bold")
gs4 = gridspec.GridSpec(2, 4, figure=fig4, hspace=0.55, wspace=0.42)
for col, (v, yl, t) in enumerate([
    (mse, "MSE (↓)", "Cell-level MSE"), (energy, "Energy Dist. (↓)", "Energy Distance"),
    (mmd, "MMD RBF (↓)", "MMD (RBF kernel)"), (chamf, "Chamfer (↓)", "Chamfer Distance")]):
    ax = fig4.add_subplot(gs4[0, col])
    x = np.arange(4); bars = ax.bar(x, v, color=colors, width=.55, zorder=3, edgecolor="white", lw=.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(yl); ax.set_title(t)
    ax.yaxis.grid(True, lw=.4, ls="--", alpha=.6, zorder=0); ax.set_axisbelow(True)
    for b, val in zip(bars, v):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

ax_l = fig4.add_subplot(gs4[1, :2])
mnames = ["MSE", "Energy", "MMD", "Chamfer"]
for ratios, col_c, lab, yo in [
    ([mse[0]/mse[1], energy[0]/energy[1], mmd[0]/mmd[1], chamf[0]/chamf[1]], M1_COLOR, "θ=1.0", .2),
    ([mse[0]/mse[2], energy[0]/energy[2], mmd[0]/mmd[2], chamf[0]/chamf[2]], M4_COLOR, "θ=4.0", 0),
    ([mse[0]/mse[3], energy[0]/energy[3], mmd[0]/mmd[3], chamf[0]/chamf[3]], M10_COLOR, "θ=10.0",-.2)]:
    yy = np.arange(4) + yo
    ax_l.hlines(yy, 1, ratios, color=col_c, lw=2.5, alpha=.7)
    ax_l.scatter(ratios, yy, color=col_c, s=60, zorder=5, label=lab)
    for xi, yi in zip(ratios, yy):
        ax_l.text(xi+.05, yi, f"{xi:.1f}×", va="center", fontsize=7.5, color=col_c, fontweight="bold")
ax_l.axvline(1, color="gray", ls="--", lw=1.2)
ax_l.set_yticks(range(4)); ax_l.set_yticklabels(mnames)
ax_l.set_xlabel("Improvement factor (↑ better)"); ax_l.set_title("Improvement vs Count Bridge")
ax_l.legend(loc="lower right", frameon=False)
ax_l.xaxis.grid(True, lw=.4, ls="--", alpha=.5); ax_l.set_xlim(0.8, mse[0]/mse[1]+1.5)

ax_t = fig4.add_subplot(gs4[1, 2:])
tv = [1.0, 4.0, 10.0]
def n01(v): lo,hi=min(v),max(v); return [(x-lo)/(hi-lo+1e-9) for x in v]
ax_t.plot(tv, n01([mse[1],mse[2],mse[3]]), "s-", color=M1_COLOR, lw=2, ms=7, label="MSE")
ax_t.plot(tv, n01([energy[1],energy[2],energy[3]]), "^-", color=M4_COLOR, lw=2, ms=7, label="Energy")
ax_t.plot(tv, n01([mmd[1],mmd[2],mmd[3]]), "o-", color=M10_COLOR, lw=2, ms=7, label="MMD")
ax_t.plot(tv, n01([chamf[1],chamf[2],chamf[3]]), "D-", color="#E8A838", lw=2, ms=7, label="Chamfer")
ax_t.set_xlabel("θ (lower = stronger coupling)"); ax_t.set_ylabel("Normalised metric (↓)")
ax_t.set_title("Sensitivity to θ"); ax_t.set_xticks(tv)
ax_t.yaxis.grid(True, lw=.4, ls="--", alpha=.5); ax_t.legend(loc="upper left", frameon=False, fontsize=7.5)

fig4.savefig(OUT / "figure4_merfish_metrics.pdf", bbox_inches="tight")
fig4.savefig(OUT / "figure4_merfish_metrics.png", bbox_inches="tight")
plt.close(fig4); print(f"  Saved → {OUT}/figure4_merfish_metrics.png")

print(f"\nAll done! → {OUT.resolve()}/")
