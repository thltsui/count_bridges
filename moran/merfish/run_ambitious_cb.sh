#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# AMBITIOUS 24h+ SWEEP: Count Bridge Baseline (Real S1R1 Dataset)
#
# Very heavy model: hidden=512, enc=5, dec=5 (~3.5M params)
# Training on 78k cells / 1958 spots -> 40 epochs
# Est. time per epoch ~10 mins on MPS -> ~7 hrs total
#
# Checkpoints & visual spatial plots will auto-generate every 10 epochs.
# ══════════════════════════════════════════════════════════════════════════════
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="outputs/experiment_ambitious_cb_${TIMESTAMP}"
LOG_DIR="${SWEEP_DIR}/logs"
PLOT_DIR="${SWEEP_DIR}/plots"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

RESULTS_FILE="${SWEEP_DIR}/RESULTS.txt"

# ── Model config ──
HIDDEN=512
ENC_LAYERS=5
DEC_LAYERS=5
NOISE_DIM=64

EM_EPOCHS=40
M_STEPS=100
DENOISE_STEPS=20
BATCH_SIZE=8
LR=3e-4
SAVE_EVERY=10

echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "${RESULTS_FILE}"
echo "║  AMBITIOUS SWEEP — COUNT BRIDGE BASELINE (S1R1)            ║" | tee -a "${RESULTS_FILE}"
echo "║  Model: hidden=${HIDDEN}, enc=${ENC_LAYERS}, dec=${DEC_LAYERS}                       ║" | tee -a "${RESULTS_FILE}"
echo "║  Training: ${EM_EPOCHS} epochs × ${M_STEPS} M-steps × ${DENOISE_STEPS} denoise      ║" | tee -a "${RESULTS_FILE}"
echo "║  Output: ${SWEEP_DIR}                                      ║" | tee -a "${RESULTS_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Started at: $(date)" | tee -a "${RESULTS_FILE}"

caffeinate -d -m -i -s -t 86400 &
CAFF_PID=$!
echo "caffeinate PID: ${CAFF_PID} (24 hours duration safeguard)"

RUN_DIR="${SWEEP_DIR}/merfish_cb_baseline"
RUN_LOG="${LOG_DIR}/train_cb.log"

echo "" | tee -a "${RESULTS_FILE}"
echo "══════════════════════════════════════════════════════════" | tee -a "${RESULTS_FILE}"
echo "  Training Count Bridge baseline  started $(date +%H:%M:%S)" | tee -a "${RESULTS_FILE}"
echo "══════════════════════════════════════════════════════════" | tee -a "${RESULTS_FILE}"

# Model training (eval checkpoints and plots generated natively every SAVE_EVERY)
.venv/bin/python -m moran.merfish.run \
    --data-dir data/merfish \
    --npz-name S1R1.npz \
    --device mps \
    --bridge-type count_bridge \
    --h-kernel 20.0 \
    --hidden-dim ${HIDDEN} \
    --n-enc-layers ${ENC_LAYERS} \
    --n-dec-layers ${DEC_LAYERS} \
    --noise-dim ${NOISE_DIM} \
    --em-epochs ${EM_EPOCHS} \
    --n-m-steps ${M_STEPS} \
    --n-denoise-steps ${DENOISE_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --save-every ${SAVE_EVERY} \
    --output-dir "${RUN_DIR}" > "${RUN_LOG}" 2>&1

echo "  ✓ CB done at $(date +%H:%M:%S)" | tee -a "${RESULTS_FILE}"

# Extract final loss
FINAL_LOSS=$(grep "avg M-loss" "${RUN_LOG}" | tail -1 | sed 's/.*avg M-loss=\([0-9.]*\).*/\1/')
echo "  Final loss: ${FINAL_LOSS}" | tee -a "${RESULTS_FILE}"

echo "" | tee -a "${RESULTS_FILE}"
echo "Finished at: $(date)" | tee -a "${RESULTS_FILE}"

# Explicitly plot the comparison between the CB baseline and Moran Theta 1.0 (Epoch 20)
MORAN_CKPT="outputs/experiment_ambitious_24h_20260425_152955/merfish_moran_theta_1.0/checkpoints/epoch_020.pt"
CB_CKPT="${RUN_DIR}/checkpoints/epoch_040.pt"

echo "Generating formal comparison plot with Moran Theta=1.0..."
.venv/bin/python -m moran.merfish.plot_stochastic \
    --moran-ckpt "${MORAN_CKPT}" \
    --cb-ckpt "${CB_CKPT}" \
    --output "${PLOT_DIR}/formal_comparison_s1r1.png" > /dev/null 2>&1

kill ${CAFF_PID} 2>/dev/null || true
