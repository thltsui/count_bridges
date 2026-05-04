"""
MERFISH dataset loader for Moran spatial deconvolution.

Loads S1R1.npz (or S1R2.npz) and builds:
  - Per-cell gene expression counts [N_total, 649]
  - Per-cell spatial coordinates (x_um, y_um) [N_total, 2]
  - Per-spot grouping (which cells belong to which spot)
  - Per-spot bulk expression X_0 = sum of cell counts
  - Optional DAPI images [N_total, 1, 256, 256]

Usage:
    dataset = MerfishMoranDataset(data_dir="/path/to/data/S1R1")
    spot = dataset[42]
    # spot['counts']  : [N_spot, 649] int   cell gene expression
    # spot['coords']  : [N_spot, 2]   float cell spatial positions (µm)
    # spot['X_0']     : [649]         int   bulk expression (sum)
    # spot['n_cells'] : int
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MerfishMoranDataset(Dataset):
    """
    One sample = one tissue spot = one Moran population.

    Unlike Count Bridges (which flattens all cells and uses a sparse
    aggregation matrix), the Moran model treats each spot as a self-contained
    population of N interacting particles.
    """

    def __init__(
        self,
        data_dir: str,
        npz_name: str = "S1R1.npz",
        max_cells_per_spot: int = 60,
        min_cells_per_spot: int = 3,
        load_images: bool = False,
        img_size: int = 256,
    ):
        data_path = Path(data_dir)
        npz_path = data_path / npz_name

        if not npz_path.exists():
            # Try alternate layout
            npz_path = data_path / "merfish_idx.npz"

        assert npz_path.exists(), f"Data file not found: {npz_path}"

        npz = np.load(npz_path, allow_pickle=True)

        self.counts = npz["counts"]           # [N_total, G] int
        self.spots = npz["spots"]             # list of arrays (cell indices per spot)
        self.G = self.counts.shape[1]         # number of genes (649)

        # Spatial coordinates
        if "x_um" in npz and "y_um" in npz:
            self.x_um = npz["x_um"].astype(np.float32)  # [N_total]
            self.y_um = npz["y_um"].astype(np.float32)  # [N_total]
            self.has_coords = True
        else:
            self.has_coords = False
            self.x_um = np.zeros(self.counts.shape[0], dtype=np.float32)
            self.y_um = np.zeros(self.counts.shape[0], dtype=np.float32)

        # Optional: cell type annotations
        if "annotations" in npz:
            self.annotations = npz["annotations"]
        else:
            self.annotations = None

        # Optional: DAPI images
        self.load_images = load_images
        if load_images and "imgs" in npz:
            self.imgs = npz["imgs"]  # array of variable-size images
        else:
            self.imgs = None
            self.load_images = False

        self.img_size = img_size
        self.max_cells = max_cells_per_spot
        self.min_cells = min_cells_per_spot

        # Filter spots by cell count
        self.valid_spots = []
        for i, spot_indices in enumerate(self.spots):
            n = len(spot_indices)
            if self.min_cells <= n <= self.max_cells:
                self.valid_spots.append(i)

        print(f"MERFISH dataset: {len(self.valid_spots)} valid spots "
              f"(of {len(self.spots)} total), "
              f"{self.counts.shape[0]} cells, {self.G} genes")
        if self.has_coords:
            print(f"  Spatial coordinates available: x_um ∈ [{self.x_um.min():.0f}, {self.x_um.max():.0f}], "
                  f"y_um ∈ [{self.y_um.min():.0f}, {self.y_um.max():.0f}]")

        # Precompute per-spot statistics
        cell_counts = [len(self.spots[i]) for i in self.valid_spots]
        print(f"  Cells per spot: min={min(cell_counts)}, max={max(cell_counts)}, "
              f"mean={np.mean(cell_counts):.1f}, median={np.median(cell_counts):.0f}")

    def __len__(self):
        return len(self.valid_spots)

    def __getitem__(self, index):
        spot_idx = self.valid_spots[index]
        cell_indices = self.spots[spot_idx]
        n_cells = len(cell_indices)

        # Gene expression counts [N, G]
        counts = torch.from_numpy(
            self.counts[cell_indices].astype(np.float32)
        )

        # Spatial coordinates [N, 2]
        coords = torch.stack([
            torch.from_numpy(self.x_um[cell_indices]),
            torch.from_numpy(self.y_um[cell_indices]),
        ], dim=-1)

        # Bulk expression (sum constraint)
        X_0 = counts.sum(dim=0).long()  # [G]

        result = {
            "counts": counts,           # [N, G] float
            "coords": coords,           # [N, 2] float (µm)
            "X_0": X_0,                 # [G]   long
            "n_cells": n_cells,
        }

        # Optional DAPI images
        if self.load_images and self.imgs is not None:
            imgs = []
            for ci in cell_indices:
                img = self.imgs[ci]
                if img.shape[0] <= self.img_size and img.shape[1] <= self.img_size:
                    padded = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                    padded[:img.shape[0], :img.shape[1]] = img.astype(np.float32) / 65535.0
                else:
                    padded = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                imgs.append(padded)
            result["imgs"] = torch.from_numpy(
                np.stack(imgs)[:, None, :, :]  # [N, 1, H, W]
            )

        return result


def merfish_collate_fn(batch):
    """
    Collate variable-size spots into a padded batch.

    Returns:
        counts:  [B, N_max, G] padded
        coords:  [B, N_max, 2] padded
        X_0:     [B, G]
        mask:    [B, N_max] bool (True = real cell, False = padding)
        n_cells: [B] int
    """
    G = batch[0]["counts"].shape[1]
    n_cells_list = [item["n_cells"] for item in batch]
    N_max = max(n_cells_list)
    B = len(batch)

    counts = torch.zeros(B, N_max, G)
    coords = torch.zeros(B, N_max, 2)
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    X_0 = torch.stack([item["X_0"] for item in batch])
    n_cells = torch.tensor(n_cells_list, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["n_cells"]
        counts[i, :n] = item["counts"]
        coords[i, :n] = item["coords"]
        mask[i, :n] = True

    result = {
        "counts": counts,
        "coords": coords,
        "mask": mask,
        "X_0": X_0,
        "n_cells": n_cells,
    }

    if "imgs" in batch[0]:
        H = batch[0]["imgs"].shape[-1]
        imgs = torch.zeros(B, N_max, 1, H, H)
        for i, item in enumerate(batch):
            n = item["n_cells"]
            imgs[i, :n] = item["imgs"]
        result["imgs"] = imgs

    return result
