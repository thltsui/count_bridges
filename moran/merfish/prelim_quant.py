import torch
import numpy as np
from pathlib import Path
from torch.utils.data import random_split
from moran.merfish.dataset import MerfishMoranDataset
from moran.merfish.forward import MerfishMoranForward
from moran.merfish.eval_metrics import evaluate_model
import json

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

dataset = MerfishMoranDataset(data_dir="data/merfish", npz_name="S1R1.npz",
                                max_cells_per_spot=60, min_cells_per_spot=3)
rng = torch.Generator().manual_seed(42)
train_size = int(0.85 * len(dataset))
val_size   = len(dataset) - train_size
_, val_ds  = random_split(dataset, [train_size, val_size], generator=rng)
# We evaluate on a tiny subset (30 spots) just to give the user a quick "preliminary quant" without hanging for 10 minutes.
val_indices = list(val_ds.indices)[:30]
print(f"Evaluating Fast Preliminary Quant on {len(val_indices)} validation spots...")

fwd = MerfishMoranForward(G=dataset.G, n_substeps=5)

configs = [
    ("Count Bridge (Baseline)", 
     "outputs/experiment_ambitious_cb_20260425_223936/merfish_cb_baseline/checkpoints/epoch_040.pt", 
     True),
    ("Moran θ=1.0 (Strong Drift)", 
     "outputs/experiment_ambitious_24h_20260425_152955/merfish_moran_theta_1.0/checkpoints/epoch_060.pt", 
     True),
    ("Moran θ=4.0", 
     "outputs/experiment_ambitious_24h_20260425_152955/merfish_moran_theta_4.0/checkpoints/epoch_060.pt", 
     True),
    ("Moran θ=10.0 (Weak Drift)", 
     "outputs/experiment_ambitious_24h_20260425_152955/merfish_moran_theta_10.0/checkpoints/epoch_060.pt", 
     True),
]

results = []
for label, ckpt, use_coords in configs:
    print(f"\nEvaluating: {label}")
    # The new networks were heavily parameterized (512, 5 layers). We need to pass this to evaluate_model so it loads them.
    # evaluate_model takes hidden_dim=256 by default. Let's load the checkpoint to see what it is.
    checkpoint = torch.load(ckpt, map_location=device)
    config = checkpoint.get("config", {})
    hidden_dim = config.get("hidden_dim", 512)
    n_enc_layers = config.get("n_enc_layers", 5)
    n_dec_layers = config.get("n_dec_layers", 5)
    noise_dim = config.get("noise_dim", 64)
    
    row = evaluate_model(ckpt, dataset, val_indices, fwd, device, t=0.5, label=label, use_coords=use_coords,
                         hidden_dim=hidden_dim, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers, noise_dim=noise_dim)
    results.append(row)
    print(f"  MSE={row['mse']:.4f}  MMD={row['mmd']:.5f}  Energy={row['energy_dist']:.4f}  Chamfer={row['chamfer']:.4f}")

with open("outputs/preliminary_quant.json", "w") as f:
    json.dump(results, f, indent=4)
