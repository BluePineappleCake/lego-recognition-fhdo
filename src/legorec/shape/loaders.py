"""Batch iterators for preprocessed arrays: x (N, H, W) uint8, z (N,) float, y (N,) int."""

import torch
from torch.utils.data import Dataset

from legorec.shape.augment import normalize


class GPUBatches:
    """Whole array on the GPU (train: 32k x 96 x 96 uint8 ~ 0.3 GB): no DataLoader,
    no worker processes, no CPU bottleneck. Also works on the CPU for testing."""

    def __init__(self, x, z, y, device, bs, shuffle=False, drop_last=False):
        self.x = torch.from_numpy(x).to(device).unsqueeze(1)
        self.z = torch.from_numpy(z).to(device)
        self.y = torch.as_tensor(y, device=device)
        self.bs, self.shuffle, self.drop_last = bs, shuffle, drop_last

    def __len__(self):
        n = len(self.y)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):
        n, d = len(self.y), self.y.device
        idx = torch.randperm(n, device=d) if self.shuffle else torch.arange(n, device=d)
        for i in range(len(self)):
            j = idx[i * self.bs : (i + 1) * self.bs]
            yield normalize(self.x[j]), self.z[j], self.y[j]


class ArrayDataset(Dataset):
    """CPU fallback for torch DataLoader; tf works on uint8 (1, H, W)."""

    def __init__(self, x, z, y, tf=None):
        self.x = torch.from_numpy(x).unsqueeze(1)
        self.z = torch.from_numpy(z)
        self.y = torch.as_tensor(y)
        self.tf = tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.tf:
            x = self.tf(x)
        return normalize(x), self.z[i], self.y[i]
