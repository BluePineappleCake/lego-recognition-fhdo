"""Shared data structures passed between pipeline steps."""

from dataclasses import dataclass

import numpy as np


@dataclass
class BrickCrop:
    """One brick cut out of a photo by the segmentation step."""

    image: np.ndarray  # RGB cut-out, H x W x 3, uint8
    mask: np.ndarray  # binary mask, H x W, bool
    bbox: tuple[int, int, int, int]  # x, y, width, height in the original photo
