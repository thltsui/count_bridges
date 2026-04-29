#!/bin/bash
set -e

# ── Train Count Bridge baseline with the same loss fixes ──
CB_OUT="outputs/merfish_cb_baseline"
mkdir -p "${CB_OUT}/plots"

echo "═══════════════════════════════════════════════════════"
echo "  Count Bridge Baseline Training (L² + Zero Penalty)  "
echo "═══════════════════════════════════════════════════════"

.venv/bin/python -m moran.merfish.run \
    --data-dir data/merfish \
    --npz-name S1R1.npz \
    --device mps \
    --bridge-type count_bridge \
    --em-epochs 20 \
    --output-dir "${CB_OUT}"

echo "CB training complete."

# ── Train Moran (θ=4.0) with the new L² squared Chamfer ──
MORAN_OUT="outputs/merfish_moran_L2"
mkdir -p "${MORAN_OUT}/plots"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Moran θ=4.0 Training (L² + Zero Penalty)           "
echo "═══════════════════════════════════════════════════════"

.venv/bin/python -m moran.merfish.run \
    --data-dir data/merfish \
    --npz-name S1R1.npz \
    --device mps \
    --theta 4.0 \
    --h-kernel 20.0 \
    --em-epochs 20 \
    --output-dir "${MORAN_OUT}"

echo "Moran training complete."

# ── Generate comparison plots ──
echo ""
echo "Generating plots..."

CB_CKPT="${CB_OUT}/checkpoints/epoch_020.pt"
MORAN_CKPT="${MORAN_OUT}/checkpoints/epoch_020.pt"
PLOT_OUT="${MORAN_OUT}/plots/spatial_L2_comparison.png"

.venv/bin/python -m moran.merfish.plot_stochastic \
    --moran-ckpt "${MORAN_CKPT}" \
    --cb-ckpt "${CB_CKPT}" \
    --output "${PLOT_OUT}"

echo "Plot saved: ${PLOT_OUT}"
echo "Done."
