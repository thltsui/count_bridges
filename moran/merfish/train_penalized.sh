#!/bin/bash
set -e

# Output directory for this specific run
OUT_DIR="outputs/merfish_moran_penalized"
mkdir -p "${OUT_DIR}/plots"

echo "Starting Penalized Moran MERFISH Training..."

# 1. Run training (Checkpoints are automatically saved every 5 epochs and at the end)
.venv/bin/python -m moran.merfish.run \
    --data-dir data/merfish \
    --npz-name S1R1.npz \
    --device mps \
    --theta 4.0 \
    --h-kernel 20.0 \
    --em-epochs 20 \
    --output-dir "${OUT_DIR}"

echo "Training complete. Checkpoints saved in ${OUT_DIR}/checkpoints/"

# 2. Generate plot post-training using the final checkpoint (epoch 20)
CKPT="${OUT_DIR}/checkpoints/epoch_020.pt"
PLOT="${OUT_DIR}/plots/spatial_plot_theta_4.0_penalized.png"

if [ -f "${CKPT}" ]; then
    echo "Generating spatial plot..."
    .venv/bin/python -m moran.merfish.plot_stochastic \
        --moran-ckpt "${CKPT}" \
        --no-cb \
        --output "${PLOT}"
    
    echo "Plot generated at ${PLOT}"
else
    echo "ERROR: Checkpoint ${CKPT} not found. Training may have failed."
    exit 1
fi
