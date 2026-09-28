"""torch Dataset of (observed, ideal) windows cropped on demand from the full images
written by gen_data.

No transforms: the images are already normalized and (for now) un-augmented, so
__getitem__ just crops and hands back (1, P, P) tensors. `stride` keeps only corners on
that grid, e.g. stride=patch_size gives the non-overlapping subset.

The images are memory-mapped so DataLoader workers share the OS page cache instead of
each holding a copy. They are opened lazily, on first access: workers are spawned on
macOS, which pickles the dataset, and pickling an np.memmap copies its data.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from core.config import load_config


class PatchDataset(Dataset):
    def __init__(self, data_dir, split, stride=1):
        self.data_dir = Path(data_dir)
        self.patch_size = load_config(self.data_dir / "resolved_config.yaml").data.patch_size
        corners = np.load(self.data_dir / f"{split}_corners.npy")
        self.corners = corners[(corners % stride == 0).all(axis=1)]
        self.observed = None
        self.ideal = None

    def __len__(self):
        return len(self.corners)

    def __getitem__(self, index):
        if self.observed is None:
            self.observed = np.load(self.data_dir / "observed.npy", mmap_mode="r")
            self.ideal = np.load(self.data_dir / "ideal.npy", mmap_mode="r")
        y, x = self.corners[index]
        size = self.patch_size
        observed = np.array(self.observed[y:y + size, x:x + size])[None]
        ideal = np.array(self.ideal[y:y + size, x:x + size])[None]
        return torch.from_numpy(observed), torch.from_numpy(ideal)
