"""Mini-batches from preprocessed arrays that fit into memory.

The whole training set is small (32,000 x 96 x 96 bytes = 0.3 GB), so it is
moved to the GPU once and batches are cut from it directly. No DataLoader and
no worker processes needed.
"""

import math

import torch

from legorec.shape.augment import normalize


class Batches:
    """x: (N, H, W) uint8 images, z: (N,) size feature, y: (N,) class index.

    Iterating yields (normalised images (B, 1, H, W), z, y) on `device`.
    """

    def __init__(self, x, z, y, device, batch_size, shuffle=False):
        self.x = torch.from_numpy(x).to(device).unsqueeze(1)
        self.z = torch.from_numpy(z).to(device)
        self.y = torch.as_tensor(y, device=device)
        self.batch_size, self.shuffle = batch_size, shuffle

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)  # number of batches

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n) if self.shuffle else torch.arange(n)
        order = order.to(self.y.device)
        for start in range(0, n, self.batch_size):
            i = order[start : start + self.batch_size]
            yield normalize(self.x[i]), self.z[i], self.y[i]
