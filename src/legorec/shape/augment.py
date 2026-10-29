"""Training augmentation for the shape classifier.

Every training batch gets a random rotation, a small zoom and shift, and a
random change of brightness and contrast. This is done for the whole batch at
once on the GPU (or CPU), which is much faster than augmenting image by image.

Rotation is in-plane only, which keeps the handedness of a part. Horizontal
flips are off by default: only enable them if no class has a mirror twin.
"""

import math

import torch
import torch.nn.functional as F

# images are scaled to roughly [-2, 2]: (x/255 - MEAN) / STD
MEAN, STD = 0.5, 0.25


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float().div(255).sub(MEAN).div(STD)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * STD + MEAN).clamp(0, 1)


def augment_batch(
    x, hflip=False, deg=180, scale=(0.9, 1.1), shift=0.1, light=(0.7, 1.3)
):
    """x: normalised batch (B, 1, H, W) -> augmented batch of the same shape.

    How the geometry works: for every image we build a 2 x 3 matrix `theta`
    (rotation + zoom in the left 2 x 2 part, shift in the last column).
    `affine_grid` turns it into "for each output pixel, where do I read from
    in the input?", and `grid_sample` reads those positions (bilinear).
    Coordinates run from -1 to 1, so a shift of 0.1 is 5 % of the image size.
    """
    b, dev = x.shape[0], x.device

    # one random angle, zoom and shift per image
    angle = (torch.rand(b, device=dev) * 2 - 1) * math.radians(deg)
    zoom = torch.empty(b, device=dev).uniform_(*scale)
    move = (torch.rand(b, 2, device=dev) * 2 - 1) * shift

    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    row1 = torch.stack([cos, -sin, move[:, 0]], 1)
    row2 = torch.stack([sin, cos, move[:, 1]], 1)
    theta = torch.stack([row1, row2], 1)  # (B, 2, 3)
    if hflip:  # mirror the x axis for a random half of the batch
        flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        theta[:, :, 0] *= flip[:, None]

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # padding_mode="border": pixels rotated in from outside the image repeat the
    # edge colour, so there are no black corners (real photos never have them)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    # brightness (multiply) and contrast (stretch around the image mean)
    brightness = torch.empty(b, 1, 1, 1, device=dev).uniform_(*light)
    contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(*light)
    x01 = denormalize(x) * brightness
    mean = x01.mean((2, 3), keepdim=True)
    x01 = ((x01 - mean) * contrast + mean).clamp(0, 1)
    return (x01 - MEAN) / STD
