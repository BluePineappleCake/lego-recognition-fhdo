"""Training augmentation for the shape classifier.

Two implementations of the same augmentation:
- gpu_augment: a whole batch at once on the GPU (default, fast)
- AffineBG / cpu_transform: per image in DataLoader workers (fallback)

Rotation is in-plane only, which keeps the handedness of a part. Horizontal
flips are off by default: only enable them if no class has a mirror twin.
"""

import math
import random

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T

MEAN, STD = 0.5, 0.25


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float().div(255).sub(MEAN).div(STD)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * STD + MEAN).clamp(0, 1)


def gpu_augment(x, hflip=False, deg=180, scale=(0.9, 1.1), shift=0.1, light=(0.7, 1.3)):
    """x: normalised float batch (B, 1, H, W). padding_mode='border' repeats the
    edge pixels, so there are no black corners or seams after rotation."""
    b, d = x.shape[0], x.device
    a = (torch.rand(b, device=d) * 2 - 1) * math.radians(deg)
    s = torch.empty(b, device=d).uniform_(*scale)
    t = (torch.rand(b, 2, device=d) * 2 - 1) * shift  # 0.1 = 5 % of the image size
    cos, sin = torch.cos(a) / s, torch.sin(a) / s
    theta = torch.stack(
        [torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1
    )
    if hflip:
        theta[:, :, 0] *= torch.where(torch.rand(b, device=d) < 0.5, -1.0, 1.0)[:, None]
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    bright = torch.empty(b, 1, 1, 1, device=d).uniform_(*light)
    contrast = torch.empty(b, 1, 1, 1, device=d).uniform_(*light)
    x01 = denormalize(x) * bright
    m = x01.mean((2, 3), keepdim=True)
    x01 = ((x01 - m) * contrast + m).clamp(0, 1)
    return (x01 - MEAN) / STD


class AffineBG:
    """CPU version of the geometric part of gpu_augment, for one uint8 image (1, H, W)."""

    def __init__(self, deg=180, scale=(0.9, 1.1), shift=0.05):
        self.deg, self.scale, self.shift = deg, scale, shift

    def __call__(self, x):
        n = x.shape[-1]
        p = int(0.35 * n)
        # replicate-pad first so the interpolation seam ends up outside the crop
        xp = F.pad(x[None].float(), (p, p, p, p), mode="replicate")[0]
        s = n * self.shift
        xp = T.functional.affine(
            xp,
            angle=random.uniform(-self.deg, self.deg),
            translate=[random.uniform(-s, s), random.uniform(-s, s)],
            scale=random.uniform(*self.scale),
            shear=[0.0],
            interpolation=T.InterpolationMode.BILINEAR,
        )
        return xp[:, p : p + n, p : p + n].round().clamp(0, 255).to(torch.uint8)


def cpu_transform(hflip=False):
    t = [AffineBG(), T.ColorJitter(brightness=0.3, contrast=0.3)]
    if hflip:
        t.append(T.RandomHorizontalFlip())
    return T.Compose(t)
