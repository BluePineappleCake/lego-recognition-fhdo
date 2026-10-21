"""Shared data structures passed between pipeline steps."""

from dataclasses import dataclass

import numpy as np


@dataclass
class BrickCrop:
    """One brick cut out of a photo by the segmentation step."""

    image: np.ndarray  # RGB cut-out, H x W x 3, uint8
    mask: np.ndarray  # binary mask, H x W, bool
    bbox: tuple[int, int, int, int]  # x, y, width, height in the original photo


@dataclass
class Prediction:
    """Shape and colour result for one brick."""

    part_id: str  # Rebrickable part number, e.g. "3022"
    part_conf: float  # 0..1, softmax probability of the top class
    colour_id: int  # Rebrickable colour id
    colour_dist: float  # distance to the nearest reference colour (lower = surer)
