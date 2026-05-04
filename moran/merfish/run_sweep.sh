#!/bin/bash
# Sweep script explicitly assessing Moran MERFISH boundaries across theta parameters
# Caffeinate is integrated automatically structurally via nohup tracking

echo "Starting Moran MERFISH Parameter Sweep!"
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="outputs/sweep_logs_${timestamp}"
mkdir -p "$log_dir"

# Sweep over structurally relevant interaction ratios
for theta in 1.0 2.0 4.0 10.0; do
    out_dir="outputs/merfish_moran_theta_${theta}"
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
done

echo "Parameter sweep sequence completely finished!"
