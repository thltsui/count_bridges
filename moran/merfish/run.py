"""
Entry point for Moran MERFISH spatial deconvolution.

Usage (on Mac Mini with MPS):
    python -m moran.merfish.run \\
        --data-dir /path/to/MERFISH/S1R1 \\
        --npz-name S1R1.npz \\
        --device mps \\
        --theta 4.0 \\
        --h-kernel 20.0 \\
        --em-epochs 20 \\
        --output-dir outputs/merfish_moran

    # Evaluation against ground truth (requires data):
    python -m moran.merfish.run \\
        --data-dir /path/to/MERFISH/S1R1 \\
        --eval-only \\
        --checkpoint outputs/merfish_moran/checkpoints/epoch_020.pt

For a quick smoke test on synthetic data (no MERFISH files required):
    python -m moran.merfish.run --smoke-test
"""

import argparse
import logging
import sys

import numpy as np
import torch


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Moran MERFISH spatial deconvolution (MPS-compatible)"
    )

    # Data
    p.add_argument("--data-dir", type=str, default=None,
                   help="Directory containing S1R1.npz (or equivalent)")
    p.add_argument("--npz-name", type=str, default="S1R1.npz")
    p.add_argument("--output-dir", type=str, default="outputs/merfish_moran")

    # Model
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-enc-layers", type=int, default=3)
    p.add_argument("--n-dec-layers", type=int, default=3)
    p.add_argument("--noise-dim", type=int, default=32)
    p.add_argument("--G", type=int, default=649,
                   help="Number of genes")

    # Moran process
    p.add_argument("--bridge-type", type=str, default="moran",
                   choices=["moran", "count_bridge"],
                   help="Which generative bridge to use for training")
    p.add_argument("--gamma-mut", type=float, default=1.0,
                   help="Per-gene mutation rate")
    p.add_argument("--kappa", type=float, default=0.5,
                   help="Resampling rate (overridden by --theta if provided)")
    p.add_argument("--theta", type=float, default=None,
                   help="Mutation-drift ratio (sets kappa=2*gamma_mut/theta)")
    p.add_argument("--h-kernel", type=float, default=20.0,
                   help="Spatial kernel bandwidth in microns")

    # Training
    p.add_argument("--em-epochs", type=int, default=20)
    p.add_argument("--n-m-steps", type=int, default=50,
                   help="M-step gradient steps per batch")
    p.add_argument("--n-denoise-steps", type=int, default=10,
                   help="Denoising steps in E-step reverse")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-cells", type=int, default=50)
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--save-every", type=int, default=5)

    # Device
    p.add_argument("--device", type=str, default="mps",
                   choices=["mps", "cpu", "cuda"])

    # Modes
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a quick smoke test on synthetic data (no files needed)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    return p


# ---------------------------------------------------------------------------
# Smoke test: validate code paths without real data
# ---------------------------------------------------------------------------

def smoke_test(args):
    """
    Runs a fast end-to-end test with synthetic MERFISH-like data.

    Creates a tiny in-memory dataset and runs 2 E-M epochs to verify
    all code paths work on MPS before receiving real data.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    log = logging.getLogger("smoke_test")

    from moran.merfish.model import MerfishDenoiser, ipf_rescale, randomised_round
    from moran.merfish.forward import MerfishMoranForward
    from moran.merfish.reverse import MerfishFastAdjointReverse

    G = 50       # toy gene count
    N = 8        # cells per spot
    B = 4        # batch size
    n_spots = 20

    device_str = args.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS not available, using CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    log.info(f"Smoke test on device: {device}")

    # Synthetic data: 2 cell types, each a Poisson(lam) count vector
    np.random.seed(42)
    torch.manual_seed(42)

    type_A = np.random.poisson(5.0, size=G).astype(np.float32)   # high in first half
    type_B = np.random.poisson(3.0, size=G).astype(np.float32)   # high in second half
    type_A[:G//2] += 10
    type_B[G//2:] += 10

    def make_spot():
        n = np.random.randint(4, N+1)
        labels = np.random.choice(2, size=n)
        counts = np.stack([
            np.random.poisson(type_A if l == 0 else type_B)
            for l in labels
        ]).astype(np.int32)
        # Random coords within a 200µm patch
        coords = np.random.rand(n, 2).astype(np.float32) * 200.0
        X_0 = counts.sum(axis=0)
        return {"counts": counts, "coords": coords, "X_0": X_0, "n_cells": n}

    spots = [make_spot() for _ in range(n_spots)]
    ref_counts = np.concatenate([s["counts"] for s in spots], axis=0)

    # Model
    model = MerfishDenoiser(
        G=G, hidden_dim=64, n_enc_layers=2, n_dec_layers=2, noise_dim=8
    ).to(device)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    fwd = MerfishMoranForward(gamma_mut=1.0, kappa=0.5, h_kernel=20.0, G=G, n_substeps=5)
    rev = MerfishFastAdjointReverse(
        gamma_mut=1.0, kappa=0.5, h_kernel=20.0, n_substeps=3, device=device_str
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Collate a tiny batch
    def collate(spot_list):
        N_max = max(s["n_cells"] for s in spot_list)
        B = len(spot_list)
        counts = torch.zeros(B, N_max, G)
        coords = torch.zeros(B, N_max, 2)
        mask = torch.zeros(B, N_max, dtype=torch.bool)
        X_0 = torch.zeros(B, G, dtype=torch.long)
        n_cells = torch.tensor([s["n_cells"] for s in spot_list])
        for i, s in enumerate(spot_list):
            n = s["n_cells"]
            counts[i, :n] = torch.from_numpy(s["counts"].astype(np.float32))
            coords[i, :n] = torch.from_numpy(s["coords"])
            mask[i, :n] = True
            X_0[i] = torch.from_numpy(s["X_0"])
        return {"counts": counts, "coords": coords, "mask": mask,
                "X_0": X_0, "n_cells": n_cells}

    # --- E-M smoke: 2 epochs ---
    from moran.merfish.experiment import e_step, m_step
    for em_epoch in range(1, 3):
        log.info(f"E-M epoch {em_epoch}")
        batch = collate(spots[:B])
        batch_with_x0 = e_step(model, fwd, rev, batch, device, n_denoise_steps=3, ref_counts=ref_counts)
        for _ in range(5):
            loss = m_step(model, optimizer, fwd, batch_with_x0, device)
        log.info(f"  M-step loss: {loss:.4f}")

    # IPF and rounding test
    log.info("Testing IPF rescaling + randomised rounding...")
    pred = torch.rand(B, N, G) * 5
    X_0_test = torch.randint(0, 20, (B, G))
    mask_test = torch.ones(B, N, dtype=torch.bool)
    y = ipf_rescale(pred, X_0_test.float(), mask_test)
    y_int = randomised_round(y, X_0_test, mask_test)
    sums_ok = all(
        (y_int[b][mask_test[b]].sum(0) == X_0_test[b]).all().item()
        for b in range(B)
    )
    log.info(f"  IPF sums match X_0: {sums_ok}")

    log.info("✓ Smoke test passed — all code paths functional")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = make_parser().parse_args()

    if args.smoke_test:
        success = smoke_test(args)
        sys.exit(0 if success else 1)

    if args.data_dir is None:
        print("ERROR: --data-dir is required (or use --smoke-test)")
        sys.exit(1)

    if args.eval_only:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
        log = logging.getLogger("eval")
        log.info("Evaluation mode not yet implemented — run training first.")
        sys.exit(0)

    from moran.merfish.experiment import train_merfish
    train_merfish(
        data_dir=args.data_dir,
        npz_name=args.npz_name,
        output_dir=args.output_dir,
        bridge_type=args.bridge_type,
        G=args.G,
        hidden_dim=args.hidden_dim,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        noise_dim=args.noise_dim,
        gamma_mut=args.gamma_mut,
        kappa=args.kappa,
        h_kernel=args.h_kernel,
        theta=args.theta,
        n_em_epochs=args.em_epochs,
        n_m_steps=args.n_m_steps,
        n_denoise_steps=args.n_denoise_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_cells=args.max_cells,
        min_cells=args.min_cells,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
