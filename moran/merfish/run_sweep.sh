#!/bin/bash
# Sweep script explicitly assessing Moran MERFISH boundaries across theta parameters
# Caffeinate is integrated automatically structurally via nohup tracking

echo "Starting Moran MERFISH Parameter Sweep!"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="outputs/sweep_logs_${timestamp}"
mkdir -p "$log_dir"

# Sweep over structurally relevant interaction ratios
for theta in 1.0 2.0 4.0 10.0; do
    out_dir="outputs/sweep_gene_level_${timestamp}/merfish_moran_theta_${theta}"
    log_file="${log_dir}/sweep_theta_${theta}.log"
    
    echo "=================================================="
    echo "Running Moran MERFISH with theta = ${theta}"
    echo "Output Directory: ${out_dir}"
    echo "Log path: ${log_file}"
    echo "=================================================="
    
    # We use .venv/bin/python precisely mapping over the correct site-packages
    .venv/bin/python -m moran.merfish.run \
        --data-dir data/merfish \
        --npz-name S1R1.npz \
        --device mps \
        --em-epochs 50 \
        --theta "${theta}" \
        --h-kernel 20.0 \
        --save-every 10 \
        --output-dir "${out_dir}" > "${log_file}" 2>&1
        
    echo "Completed theta = ${theta}. Results dumped into ${out_dir}."

    # ── Post-training: generate spatial comparison PNGs ──
    ckpt_file="${out_dir}/checkpoints/epoch_050.pt"
    plot_file="outputs/sweep_gene_level_${timestamp}/plots/spatial_theta_${theta}.png"
    plot_log="${log_dir}/plot_theta_${theta}.log"

    if [ -f "${ckpt_file}" ]; then
        echo "  Generating spatial plot → ${plot_file}"
        .venv/bin/python -m moran.merfish.plot_stochastic \
            --moran-ckpt "${ckpt_file}" \
            --no-cb \
            --output "${plot_file}" > "${plot_log}" 2>&1
        echo "  Plot saved."
    else
        echo "  WARNING: ${ckpt_file} not found — skipping plot."
    fi
done

# ── Post-sweep: compute distributional metrics across all thetas ──
sweep_dir="outputs/sweep_gene_level_${timestamp}"
metrics_json="${sweep_dir}/metrics_table.json"
metrics_log="${log_dir}/eval_metrics.log"

echo ""
echo "=================================================="
echo "Running eval_metrics over sweep: ${sweep_dir}"
echo "Output JSON: ${metrics_json}"
echo "=================================================="

.venv/bin/python -m moran.merfish.eval_metrics \
    --data-dir data/merfish \
    --npz-name S1R1.npz \
    --device mps \
    --sweep-dir "${sweep_dir}" \
    --out-json "${metrics_json}" > "${metrics_log}" 2>&1

echo "Metrics saved to ${metrics_json}. Log: ${metrics_log}"
echo ""
echo "Parameter sweep sequence completely finished!"
