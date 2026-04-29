#!/bin/bash
set -e

OUT_DIR="outputs/merfish_verify_s1r1"
mkdir -p "${OUT_DIR}/plots"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TINY VERIFICATION RUN — Real S1R1 Vizgen Dataset          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# 1. Run training (1 epoch only, small dim just to ensure pipeline doesn't crash)
.venv/bin/python -m moran.merfish.run \
    --data-dir data/merfish \
    --npz-name S1R1.npz \
    --device mps \
    --theta 4.0 \
    --hidden-dim 128 \
    --n-enc-layers 2 \
    --n-dec-layers 2 \
    --em-epochs 1 \
    --n-m-steps 10 \
    --n-denoise-steps 5 \
    --batch-size 8 \
    --output-dir "${OUT_DIR}"

echo "Training complete."

# 2. Generate plot post-training using the final checkpoint (epoch 1)
CKPT="${OUT_DIR}/checkpoints/epoch_001.pt"
PLOT="${OUT_DIR}/plots/spatial_plot_theta_4.0_verify.png"

if [ -f "${CKPT}" ]; then
    echo "Generating spatial plot..."
    .venv/bin/python -m moran.merfish.plot_stochastic \
        --moran-ckpt "${CKPT}" \
        --no-cb \
        --output "${PLOT}"
    
    echo "Plot generated at ${PLOT}"
else
    echo "ERROR: Checkpoint ${CKPT} not found."
    exit 1
fi
