# -*- coding: utf-8 -*-
"""
B_E2E_Train_DDP_DataLoader.py

PyTorch Dataset for stage-2 full-volume training stored as .pt files.

Each .pt is expected to contain:
- x: (14, D, H, W) with channels [CT, 9 structure masks, 4 strip masks]
- y: (1, D, H, W) ground-truth dose

Optional lightweight augmentation: rot90/flip and additive noise on the CT channel.

Author: Boda Ning
"""

import os, glob, random
from typing import List, Optional
import torch
from torch.utils.data import Dataset

class FullVolumePTDataset2nd(Dataset):
    def __init__(self, root_dir: str,
                 augment: bool=False, aug_prob: float=0.5,
                 noise_std: float=1e-2, add_noise_on=['ct']):
        self.files = sorted(glob.glob(os.path.join(root_dir, "**", "*.pt"), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"未找到 .pt: {root_dir}")
        self.augment  = bool(augment)
        self.aug_prob = float(aug_prob)
        self.noise_std = float(noise_std)
        self.add_noise_on = set(add_noise_on or [])


        self.CT_IDX     = 0
        self.STRUCT_BEG = 1
        self.STRUCT_END = 10
        self.STRIP_BEG  = 10
        self.STRIP_END  = 14


        self._rot_planes = ((2, 3),)
        self._flip_axes = (1, 2, 3)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        x = data["x"].float()
        y = data["y"].float()

        if self.augment and random.random() < self.aug_prob:
            op = random.choice(("rot90", "flip", "noise"))
            if op == "rot90":
                dims = random.choice(self._rot_planes)
                k = random.choice((1, 2, 3))
                x = torch.rot90(x, k=k, dims=dims)
                y = torch.rot90(y, k=k, dims=dims)
            elif op == "flip":
                axis = random.choice(self._flip_axes)
                x = torch.flip(x, dims=(axis,))
                y = torch.flip(y, dims=(axis,))
            elif op == "noise" and self.noise_std > 0:

                if 'ct' in self.add_noise_on:
                    x[self.CT_IDX].add_(torch.randn_like(x[self.CT_IDX]) * self.noise_std)
                x.clamp_(0.0, 1.0)

        return x, y
