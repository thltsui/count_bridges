"""
E-M training loop for Moran MERFISH spatial deconvolution.

Structure (mirrors Count Bridges DeconvTrainer):
  E-step: generate pseudo-ground-truth cell profiles via Moran reverse
          sampler (adjoint transport), conditioned on X_0 via IPF
  M-step: train on (x_0_inferred, t, x_t) via Chamfer + aggregation loss

MPS-compatible: all torch ops use float32, no CuPy, no scipy.
"""

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from moran.merfish.dataset import MerfishMoranDataset, merfish_collate_fn
from moran.merfish.forward import MerfishMoranForward
from moran.merfish.reverse import MerfishFastAdjointReverse
from moran.merfish.model import MerfishDenoiser, ipf_rescale, randomised_round


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics (matching Count Bridges evaluation)
# ---------------------------------------------------------------------------

def compute_metrics_merfish(pred: np.ndarray, true: np.ndarray, prefix: str = "") -> dict:
    """
    Compute distributional metrics comparing predicted vs true cell profiles.

    pred: [N_cells, G] predicted expression
    true: [N_cells, G] ground truth expression

    Returns dict with energy_distance, mmd, mse.
    """
    from scipy.spatial.distance import cdist

    pred_f = pred.astype(np.float32)
    true_f = true.astype(np.float32)

    # MSE
    mse = float(np.mean((pred_f - true_f) ** 2))

    # MMD with RBF kernel (bandwidth = median gene std)
    bw = float(np.median(np.std(true_f, axis=0)) + 1.0)
    Xp = torch.from_numpy(pred_f[:500])   # subsample for speed
    Xt = torch.from_numpy(true_f[:500])
    rbf = lambda a, b: torch.exp(-torch.cdist(a, b).pow(2) / (2 * bw ** 2))
    mmd = (rbf(Xp, Xp).mean() + rbf(Xt, Xt).mean() - 2 * rbf(Xp, Xt).mean()).item()

    # Energy distance (scaled for stability)
    n = min(200, len(pred_f), len(true_f))
    idx_p = np.random.choice(len(pred_f), n, replace=False)
    idx_t = np.random.choice(len(true_f), n, replace=False)
    Dp = cdist(pred_f[idx_p], pred_f[idx_p])
    Dt = cdist(true_f[idx_t], true_f[idx_t])
    Dpt = cdist(pred_f[idx_p], true_f[idx_t])
    energy = float(2 * Dpt.mean() - Dp.mean() - Dt.mean())

    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}energy_dist": energy,
        f"{key}mmd": mmd,
        f"{key}mse": mse,
    }


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def e_step(
    model: MerfishDenoiser,
    fwd: MerfishMoranForward,
    rev: MerfishFastAdjointReverse,
    batch: dict,
    device: torch.device,
    n_denoise_steps: int,
    ref_counts: np.ndarray,
) -> dict:
    """
    E-step: sample pseudo-ground-truth cell profiles via Moran reverse.

    Takes X_0 (bulk) as a conditioning signal.
    Returns the batch with 'x_0' replaced by inferred clean profiles.
    """
    model.eval()

    X_0 = batch["X_0"].float()                # [B, G]
    mask = batch["mask"]                       # [B, N]
    coords_np = batch["coords"].numpy()        # [B, N, 2]
    B, N, G = batch["counts"].shape

    # Sample from DP prior (per spot)
    x_inferred = []
    for b in range(B):
        n_cells = batch["n_cells"][b].item()
        x_prior = fwd.sample_prior(n_cells, ref_counts)   # [n, G]
        x_inferred.append(x_prior)

    # Iterative Moran reverse denoising
    for k in range(n_denoise_steps, 0, -1):
        t_val = k / n_denoise_steps

        # Stack current inferred states (padded)
        x_t_padded = torch.zeros(B, N, G)
        for b in range(B):
            n = batch["n_cells"][b].item()
            x_t_padded[b, :n] = torch.from_numpy(x_inferred[b].astype(np.float32))

        # Model prediction
        x_t_tensor = x_t_padded.to(device)
        t_tensor = torch.full((B,), t_val, device=device)
        mask_device = mask.to(device)
        coords_device = batch["coords"].to(device)

        x_hat = model.sample(x_t_tensor, t_tensor, mask_device, coords=coords_device).cpu()  # [B, N, G]

        # IPF projection to enforce Σ x = X_0
        x_hat_ipf = ipf_rescale(x_hat, X_0, mask)  # [B, N, G]

        t_next = (k - 1) / n_denoise_steps
        if t_next < 0.01:
            # Final step: round to integers
            x_hat_int = randomised_round(x_hat_ipf, X_0.long(), mask)
            for b in range(B):
                n = batch["n_cells"][b].item()
                x_inferred[b] = x_hat_int[b, :n].numpy().astype(np.int32)
        else:
            # Adjoint reverse step
            dt = t_val - t_next
            for b in range(B):
                n = batch["n_cells"][b].item()
                x_t_b = torch.from_numpy(x_inferred[b].astype(np.float32))
                x_hat_b = x_hat_ipf[b, :n]
                coords_b = batch["coords"][b, :n]
                x_new = rev.reverse_step(x_t_b, x_hat_b, coords_b, t_val, dt)
                x_new_np = x_new.cpu().numpy() if hasattr(x_new, 'numpy') else np.array(x_new)
                x_inferred[b] = x_new_np.astype(np.int32)

    # Pack inferred x_0 back into padded tensor
    x_0_padded = torch.zeros(B, N, G)
    for b in range(B):
        n = batch["n_cells"][b].item()
        x_0_padded[b, :n] = torch.from_numpy(x_inferred[b].astype(np.float32))

    batch = dict(batch)
    batch["x_0"] = x_0_padded
    return batch


def m_step(
    model: MerfishDenoiser,
    optimizer: torch.optim.Optimizer,
    fwd: MerfishMoranForward,
    batch: dict,
    device: torch.device,
) -> float:
    """
    M-step: train on one batch.

    Applies the Moran forward to the (inferred) clean profiles x_0,
    then computes loss against x_0 with the aggregation constraint.
    """
    model.train()
    B, N, G = batch["x_0"].shape

    x_0_np = batch["x_0"].numpy()  # [B, N, G]
    coords_np = batch["coords"].numpy()  # [B, N, 2]
    mask = batch["mask"]  # [B, N]

    # Sample random time per batch item
    t_vals = np.random.uniform(0.05, 0.95, size=B)

    # Forward Moran: produce x_t for each spot
    x_t_list = []
    for b in range(B):
        n = batch["n_cells"][b].item()
        x_0_b = x_0_np[b, :n].astype(np.int32)
        coords_b = coords_np[b, :n]
        t_b = float(t_vals[b])
        x_t_b = fwd.simulate(x_0_b, coords_b, t_b)
        # Pad back to N
        x_t_pad = np.zeros((N, G), dtype=np.float32)
        x_t_pad[:n] = x_t_b.astype(np.float32)
        x_t_list.append(x_t_pad)

    x_t = torch.from_numpy(np.stack(x_t_list)).float().to(device)   # [B, N, G]
    x_0 = batch["x_0"].float().to(device)                           # [B, N, G]
    t_tensor = torch.from_numpy(t_vals).float().to(device)           # [B]
    mask_d = mask.to(device)                                          # [B, N]
    coords_d = batch["coords"].float().to(device)                     # [B, N, 2]
    X_0 = batch["X_0"].float().to(device)                            # [B, G]

    optimizer.zero_grad()
    pred = model.forward(x_t, t_tensor, mask_d, coords=coords_d)     # [B, N, G]
    loss = model.loss(pred, x_0, mask_d, X_0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train_merfish(
    data_dir: str,
    npz_name: str = "S1R1.npz",
    output_dir: str = "outputs/merfish_moran",
    bridge_type: str = "moran",
    G: int = 649,
    hidden_dim: int = 256,
    n_enc_layers: int = 3,
    n_dec_layers: int = 3,
    noise_dim: int = 32,
    gamma_mut: float = 1.0,
    kappa: float = 0.5,
    h_kernel: float = 20.0,
    theta: float = None,       # if set, overrides kappa = 2*gamma_mut/theta
    n_em_epochs: int = 20,
    n_m_steps: int = 50,
    n_denoise_steps: int = 10,
    batch_size: int = 8,
    lr: float = 3e-4,
    device: str = "mps",
    max_cells: int = 50,
    min_cells: int = 3,
    save_every: int = 5,
):
    """
    Full E-M training loop for Moran MERFISH spatial deconvolution.

    Parameters
    ----------
    data_dir : str
        Path containing S1R1.npz (or equivalent).
    theta : float (optional)
        If provided, kappa = 2*gamma_mut / theta (overrides kappa parameter).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    if theta is not None:
        kappa = 2.0 * gamma_mut / theta
        log.info(f"theta={theta:.2f} → kappa={kappa:.4f}")

    # Device
    if device == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS not available, falling back to CPU")
        device = "cpu"
    device_obj = torch.device(device)
    log.info(f"Device: {device_obj}")

    # Dataset
    log.info(f"Loading MERFISH data from {data_dir}/{npz_name}")
    dataset = MerfishMoranDataset(
        data_dir=data_dir,
        npz_name=npz_name,
        max_cells_per_spot=max_cells,
        min_cells_per_spot=min_cells,
    )

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    log.info(f"Train={train_size} spots, Val={val_size} spots")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=merfish_collate_fn,
        num_workers=0,
    )

    # Reference counts for DP prior sampling
    all_counts = dataset.counts.astype(np.float32)  # [N_total, G]
    log.info(f"Reference counts: {all_counts.shape}, "
             f"mean count per gene: {all_counts.mean():.2f}")

    # Model
    model = MerfishDenoiser(
        G=G,
        hidden_dim=hidden_dim,
        n_enc_layers=n_enc_layers,
        n_dec_layers=n_dec_layers,
        noise_dim=noise_dim,
    ).to(device_obj)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_em_epochs * n_m_steps, eta_min=lr * 0.01
    )

    if bridge_type == "count_bridge":
        from moran.merfish.count_bridge import MerfishCountBridgeForward, MerfishCountBridgeReverse
        fwd = MerfishCountBridgeForward(n_substeps=10, G=G)
        rev = MerfishCountBridgeReverse(n_substeps=5)
        log.info(f"Bridge: Count Bridge (Skellam) baseline")
    else:
        fwd = MerfishMoranForward(
            gamma_mut=gamma_mut,
            kappa=kappa,
            h_kernel=h_kernel,
            G=G,
            n_substeps=10,
        )
        rev = MerfishFastAdjointReverse(
            gamma_mut=gamma_mut,
            kappa=kappa,
            h_kernel=h_kernel,
            guidance_weight=2.0,
            n_substeps=5,
            device=device,
        )
        log.info(f"Moran: gamma_mut={gamma_mut}, kappa={kappa:.4f}, "
                 f"theta={fwd.theta:.2f}, h_kernel={h_kernel}µm")

    # ------- E-M loop -------
    for em_epoch in range(1, n_em_epochs + 1):
        log.info(f"\n{'='*60}")
        log.info(f"  E-M Epoch {em_epoch}/{n_em_epochs}")
        log.info(f"{'='*60}")
        t0 = time.time()

        m_losses = []
        batches_seen = 0

        for batch in train_loader:
            # ---- E-step: infer clean profiles via adjoint reverse ----
            batch_with_x0 = e_step(
                model, fwd, rev, batch, device_obj,
                n_denoise_steps, all_counts
            )

            # ---- M-step: train model on inferred (x_0, x_t) pairs ----
            for _ in range(n_m_steps):
                loss_val = m_step(model, optimizer, fwd, batch_with_x0, device_obj)
                scheduler.step()
                m_losses.append(loss_val)

            batches_seen += 1
            if batches_seen % 5 == 0:
                log.info(f"    [{batches_seen}/{len(train_loader)}] "
                         f"loss={np.mean(m_losses[-10:]):.4f}")

        avg_loss = np.mean(m_losses)
        elapsed = time.time() - t0
        log.info(f"  Epoch {em_epoch}: avg M-loss={avg_loss:.4f}  ({elapsed:.0f}s)")

        # Save checkpoint
        if em_epoch % save_every == 0 or em_epoch == n_em_epochs:
            ckpt_path = ckpt_dir / f"epoch_{em_epoch:03d}.pt"
            torch.save({
                "epoch": em_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": {
                    "gamma_mut": gamma_mut, "kappa": kappa,
                    "h_kernel": h_kernel, "G": G,
                    "hidden_dim": hidden_dim,
                    "n_enc_layers": n_enc_layers,
                    "n_dec_layers": n_dec_layers,
                    "noise_dim": noise_dim,
                },
            }, ckpt_path)
            log.info(f"  Saved checkpoint: {ckpt_path}")
            
            # --- Auto-plotting hook (Safe Subprocess) ---
            # Generate a plot natively right away so the user can track progress "along the way"
            try:
                import os
                plot_dir = out_dir / "plots"
                plot_dir.mkdir(exist_ok=True)
                png_path = plot_dir / f"spatial_epoch_{em_epoch:03d}.png"
                log.info(f"  Triggering auto-plot subprocess for {ckpt_path} -> {png_path}")
                val_cmd = f"python -m moran.merfish.plot_stochastic " \
                          f"--moran-ckpt {ckpt_path} " \
                          f"--no-cb --output {png_path}"
                os.system(f"{val_cmd} > /dev/null 2>&1 &")
            except Exception as e:
                log.warning(f"Auto-plotting trigger failed: {e}")

    log.info(f"\nTraining complete. Checkpoints saved in {ckpt_dir}")
    return model

