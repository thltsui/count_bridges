#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# AMBITIOUS 24h+ SWEEP: Gene-Level Moran MERFISH (Real S1R1 Dataset)
#
# Very heavy model: hidden=512, enc=5, dec=5 (~3.5M params)
# Training on 78k cells / 1958 spots -> 60 epochs per theta
# Est. time per epoch ~10 mins on MPS -> ~10 hrs per theta -> ~30 hrs total
#
# Checkpoints & visual spatial plots will auto-generate every 10 epochs.
# ══════════════════════════════════════════════════════════════════════════════
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="outputs/experiment_ambitious_24h_${TIMESTAMP}"
LOG_DIR="${SWEEP_DIR}/logs"
PLOT_DIR="${SWEEP_DIR}/plots"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

RESULTS_FILE="${SWEEP_DIR}/RESULTS.txt"

# ── Model config ──
HIDDEN=512
ENC_LAYERS=5
DEC_LAYERS=5
NOISE_DIM=64

EM_EPOCHS=60
M_STEPS=100
DENOISE_STEPS=20
BATCH_SIZE=8
LR=3e-4
SAVE_EVERY=10

echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "${RESULTS_FILE}"
echo "║  AMBITIOUS 24h+ SWEEP — Real Vizgen Dataset (S1R1)         ║" | tee -a "${RESULTS_FILE}"
echo "║  Model: hidden=${HIDDEN}, enc=${ENC_LAYERS}, dec=${DEC_LAYERS}                       ║" | tee -a "${RESULTS_FILE}"
echo "║  Training: ${EM_EPOCHS} epochs × ${M_STEPS} M-steps × ${DENOISE_STEPS} denoise      ║" | tee -a "${RESULTS_FILE}"
echo "║  Output: ${SWEEP_DIR}                                      ║" | tee -a "${RESULTS_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Started at: $(date)" | tee -a "${RESULTS_FILE}"

# ── Keep Mac awake for 48 hours (in case it takes a bit longer) ──
caffeinate -d -m -i -s -t 172800 &
CAFF_PID=$!
echo "caffeinate PID: ${CAFF_PID} (48 hours duration safeguard)"

# ── Training sweep (Thetas tailored for exploration) ──
THETAS="1.0 4.0 10.0"
for theta in ${THETAS}; do
    RUN_DIR="${SWEEP_DIR}/merfish_moran_theta_${theta}"
    RUN_LOG="${LOG_DIR}/train_theta_${theta}.log"

    echo "" | tee -a "${RESULTS_FILE}"
    echo "══════════════════════════════════════════════════════════" | tee -a "${RESULTS_FILE}"
    echo "  Training θ=${theta}  started $(date +%H:%M:%S)" | tee -a "${RESULTS_FILE}"
    echo "══════════════════════════════════════════════════════════" | tee -a "${RESULTS_FILE}"

    # Model training (eval checkpoints and plots generated natively every SAVE_EVERY)
    .venv/bin/python -m moran.merfish.run \
        --data-dir data/merfish \
        --npz-name S1R1.npz \
        --device mps \
        --theta "${theta}" \
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

    echo "  ✓ θ=${theta} done at $(date +%H:%M:%S)" | tee -a "${RESULTS_FILE}"

    # Extract final loss
    FINAL_LOSS=$(grep "avg M-loss" "${RUN_LOG}" | tail -1 | sed 's/.*avg M-loss=\([0-9.]*\).*/\1/')
    echo "  Final loss: ${FINAL_LOSS}" | tee -a "${RESULTS_FILE}"
done

# ── Compute metrics across entire sweep (end of run) ──
METRICS_JSON="${SWEEP_DIR}/metrics_table.json"
METRICS_LOG="${LOG_DIR}/eval_metrics.log"

echo "" | tee -a "${RESULTS_FILE}"
echo "Computing metrics at $(date +%H:%M:%S)..." | tee -a "${RESULTS_FILE}"

.venv/bin/python -m moran.merfish.eval_metrics \
    --data-dir data/merfish \
    --npz-name S1R1.npz \
    --device mps \
    --sweep-dir "${SWEEP_DIR}" \
    --out-json "${METRICS_JSON}" > "${METRICS_LOG}" 2>&1 || true

echo "  ✓ Metrics saved to ${METRICS_JSON}" | tee -a "${RESULTS_FILE}"

echo "" >> "${RESULTS_FILE}"
echo "═══ Quantitative Metrics ═══" >> "${RESULTS_FILE}"
cat "${METRICS_LOG}" >> "${RESULTS_FILE}"

echo "" | tee -a "${RESULTS_FILE}"
echo "Finished at: $(date)" | tee -a "${RESULTS_FILE}"

kill ${CAFF_PID} 2>/dev/null || true
