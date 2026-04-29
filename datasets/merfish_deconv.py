import numpy as np
import torch

def pad_image(image, pad_height, pad_width):
    return np.pad(image, ((0, pad_height - image.shape[0]), (0, pad_width - image.shape[1])), mode="constant")

class MerfishDeconv:
    def __init__(
        self, 
        data_dim=None, 
        context_dim=None,
        data_path="data/merfish/", 
        num_replicates=1,
        img_size=16
    ):
        import os
        base_dir = os.path.dirname(os.path.dirname(__file__))
        target_file = os.path.join(base_dir, data_path, "S1R1.npz")
        npz = np.load(target_file, allow_pickle=True)
        self.data_dim = data_dim

        self.dapi, self.counts, self.group_idxs = self.process_npz(target_file, img_size=img_size, idx_offset=0)
        
        self.size = self.group_idxs.shape[0]
        print(f"Found {self.size} groups corresponding to valid tissue spots.")

    def process_npz(self, npz_path, img_size=256, idx_offset=0):
        npz = np.load(npz_path, allow_pickle=True)

        counts = npz["counts"]
        counts = torch.from_numpy(counts).long()
        
        group_idxs = npz["spots"]
        
        # Keep groups that actually have cells
        group_idxs_to_keep = [g for g in group_idxs if len(g) > 0]
        group_idxs = np.array(group_idxs_to_keep, dtype=object)

        # Mock blank DAPI images since our synthetic fallback purely parameterized the mathematical matrices
        dapi = torch.zeros((counts.shape[0], 1, img_size, img_size), dtype=torch.float32)

        print(f"Found {len(group_idxs)} valid groups")
        return dapi, counts, group_idxs

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        group_idxs = self.group_idxs[index]
        x_0_img = self.dapi[group_idxs]
        x_0_count = self.counts[group_idxs]
        X_0 = x_0_count.sum(axis=0)

        x_1_count = torch.round(torch.abs(torch.normal(mean=0, std=10, size=x_0_count.shape))).long()

        return {
            "x_0": x_0_count,
            "x_1": x_1_count,
            "img": x_0_img,
            "context": X_0.unsqueeze(0).repeat(x_0_count.shape[0], 1),
            "X_0": X_0
        }
