#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# OVERNIGHT SWEEP (HEAVY): Gene-Level Moran MERFISH
#
# Bigger model, more epochs, more gradient steps — fills the whole night.
#
# Model: hidden=512, enc=5, dec=5 (~3.5M params vs prev 887K)
# Training: 150 E-M epochs × 100 M-steps × 20 denoise steps
# Sweep: θ ∈ {1.0, 2.0, 4.0, 10.0}
#
# Est. time: ~7-8 hours on MPS (Mac Mini)
# ══════════════════════════════════════════════════════════════════════════════
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="outputs/overnight_heavy_${TIMESTAMP}"
LOG_DIR="${SWEEP_DIR}/logs"
PLOT_DIR="${SWEEP_DIR}/plots"
mkdir -p "${LOG_DIR}" "${PLOT_DIR}"

RESULTS_FILE="${SWEEP_DIR}/RESULTS.txt"

# ── Model config ──
HIDDEN=512
ENC_LAYERS=5
DEC_LAYERS=5
NOISE_DIM=64
EM_EPOCHS=150
M_STEPS=100
DENOISE_STEPS=20
BATCH_SIZE=8
LR=3e-4
SAVE_EVERY=25

echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "${RESULTS_FILE}"
echo "║  OVERNIGHT HEAVY SWEEP — Gene-Level Moran MERFISH          ║" | tee -a "${RESULTS_FILE}"
echo "║  Model: hidden=${HIDDEN}, enc=${ENC_LAYERS}, dec=${DEC_LAYERS}                       ║" | tee -a "${RESULTS_FILE}"
echo "║  Training: ${EM_EPOCHS} epochs × ${M_STEPS} M-steps × ${DENOISE_STEPS} denoise      ║" | tee -a "${RESULTS_FILE}"
echo "║  Loss: L² Squared Chamfer + Zero-Match Penalty (λ=5.0)    ║" | tee -a "${RESULTS_FILE}"
echo "║  Output: ${SWEEP_DIR}                                      ║" | tee -a "${RESULTS_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${RESULTS_FILE}"
echo "" | tee -a "${RESULTS_FILE}"
echo "Started at: $(date)" | tee -a "${RESULTS_FILE}"

# ── Keep Mac awake for 10 hours ──
caffeinate -d -m -i -s -t 36000 &
CAFF_PID=$!
echo "caffeinate PID: ${CAFF_PID} (10 hours)"

# ── Training sweep ──
THETAS="1.0 2.0 4.0 10.0"
for theta in ${THETAS}; do
    RUN_DIR="${SWEEP_DIR}/merfish_moran_theta_${theta}"
    RUN_LOG="${LOG_DIR}/train_theta_${theta}.log"

    echo "" | tee -a "${RESULTS_FILE}"
    echo "══════════════════════════════════════════════════════════" | tee -a "${RESULTS_FILE}"
    echo "  Training θ=${theta}  started $(date +%H:%M:%S)" | tee -a "${RESULTS_FILE}"
    echo "══════════════════════════════════════════════════════════" | tee -a "${RESULTS_FILE}"

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
    N_PARAMS=$(grep "parameters" "${RUN_LOG}" | head -1 | sed 's/.*: \(.*\) parameters/\1/')
    echo "  Params: ${N_PARAMS}  Final loss: ${FINAL_LOSS}" | tee -a "${RESULTS_FILE}"

    # ── Generate spatial plot ──
    CKPT="${RUN_DIR}/checkpoints/epoch_150.pt"
    # Fall back to latest checkpoint if epoch_150 doesn't exist
    if [ ! -f "${CKPT}" ]; then
        CKPT=$(ls -t "${RUN_DIR}/checkpoints/"*.pt 2>/dev/null | head -1)
    fi
    PLOT="${PLOT_DIR}/spatial_theta_${theta}.png"
    PLOT_LOG="${LOG_DIR}/plot_theta_${theta}.log"

    if [ -n "${CKPT}" ] && [ -f "${CKPT}" ]; then
        echo "  Generating plot from ${CKPT}..." | tee -a "${RESULTS_FILE}"
        .venv/bin/python -m moran.merfish.plot_stochastic \
            --moran-ckpt "${CKPT}" \
            --no-cb \
            --output "${PLOT}" > "${PLOT_LOG}" 2>&1
        echo "  ✓ Plot → ${PLOT}" | tee -a "${RESULTS_FILE}"
    else
        echo "  ⚠ No checkpoint found, skipping plot." | tee -a "${RESULTS_FILE}"
    fi
done

# ── Compute metrics across entire sweep ──
METRICS_JSON="${SWEEP_DIR}/metrics_table.json"
METRICS_LOG="${LOG_DIR}/eval_metrics.log"

echo "" | tee -a "${RESULTS_FILE}"
echo "══════════════════════════════════════════════════════════" | tee -a "${RESULTS_FILE}"
echo "  Computing metrics at $(date +%H:%M:%S)" | tee -a "${RESULTS_FILE}"
echo "══════════════════════════════════════════════════════════" | tee -a "${RESULTS_FILE}"

# Update eval_metrics to look for epoch_150 checkpoints
.venv/bin/python -c "
import json, sys, os, time, torch
import numpy as np
from pathlib import Path
from torch.utils.data import random_split
from moran.merfish.dataset import MerfishMoranDataset
from moran.merfish.forward import MerfishMoranForward
from moran.merfish.eval_metrics import evaluate_model

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

dataset = MerfishMoranDataset(data_dir='data/merfish', npz_name='S1R1.npz',
                              max_cells_per_spot=60, min_cells_per_spot=3)
rng = torch.Generator().manual_seed(42)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = random_split(dataset, [train_size, val_size], generator=rng)
val_indices = list(val_ds.indices)
print(f'Val set: {len(val_indices)} spots')

fwd = MerfishMoranForward(G=dataset.G, n_substeps=5)
sweep_dir = '${SWEEP_DIR}'

results = []
for theta in [1.0, 2.0, 4.0, 10.0]:
    ckpt_dir = f'{sweep_dir}/merfish_moran_theta_{theta}/checkpoints'
    # Find latest checkpoint
    ckpts = sorted(Path(ckpt_dir).glob('*.pt')) if Path(ckpt_dir).exists() else []
    if not ckpts:
        print(f'  Skip θ={theta}: no checkpoints')
        continue
    ckpt = str(ckpts[-1])
    label = f'Moran θ={theta}'
    print(f'\nEvaluating: {label} ({ckpt})')
    t0 = time.time()
    row = evaluate_model(ckpt, dataset, val_indices, fwd, device, t=0.5, label=label, use_coords=True,
                         hidden_dim=${HIDDEN}, n_enc_layers=${ENC_LAYERS}, n_dec_layers=${DEC_LAYERS})
    row['eval_time_s'] = round(time.time() - t0, 1)
    results.append(row)
    print(f'  MSE={row[\"mse\"]:.4f}  MMD={row[\"mmd\"]:.5f}  Energy={row[\"energy_dist\"]:.4f}  Chamfer={row[\"chamfer\"]:.4f}')

Path('${METRICS_JSON}').parent.mkdir(parents=True, exist_ok=True)
with open('${METRICS_JSON}', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to ${METRICS_JSON}')
" > "${METRICS_LOG}" 2>&1

echo "  ✓ Metrics saved to ${METRICS_JSON}" | tee -a "${RESULTS_FILE}"

# Append metrics to results
echo "" >> "${RESULTS_FILE}"
echo "═══ Quantitative Metrics ═══" >> "${RESULTS_FILE}"
cat "${METRICS_LOG}" >> "${RESULTS_FILE}"

echo "" | tee -a "${RESULTS_FILE}"
echo "Finished at: $(date)" | tee -a "${RESULTS_FILE}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  OVERNIGHT HEAVY SWEEP COMPLETE                             ║"
echo "║  Results:  ${SWEEP_DIR}/RESULTS.txt                         ║"
echo "║  Metrics:  ${METRICS_JSON}                                  ║"
echo "║  Plots:    ${PLOT_DIR}/                                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"

kill ${CAFF_PID} 2>/dev/null || true
