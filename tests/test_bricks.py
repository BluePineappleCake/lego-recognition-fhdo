import numpy as np
import pytest
from test_segment import _fake_sheet

from legorec.segment.bricks import bricks_on_sheet, drop_inside


def test_threshold_method_gives_one_crop_per_brick():
    crops = bricks_on_sheet(_fake_sheet(), method="threshold")
    assert len(crops) == 2
    for crop in crops:
        assert crop.image.shape[:2] == crop.mask.shape and crop.mask.any()
        assert crop.bbox[2:] == crop.mask.shape[::-1]  # width, height


def test_a_photo_without_markers_is_refused():
    with pytest.raises(ValueError):
        bricks_on_sheet(np.full((300, 300, 3), 255, np.uint8), method="threshold")


def test_a_piece_inside_a_surer_brick_is_dropped():
    whole = np.zeros((10, 10), bool)
    whole[1:9, 2:4] = True  # a long brick
    piece = np.zeros((10, 10), bool)
    piece[5:9, 2:4] = True  # its lower half, found again
    other = np.zeros((10, 10), bool)
    other[1:3, 6:9] = True  # a second brick

    kept = drop_inside([piece, whole, other], [0.6, 0.9, 0.8])

    assert len(kept) == 2
    assert kept[0] is whole
