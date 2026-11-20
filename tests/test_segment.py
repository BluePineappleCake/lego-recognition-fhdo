import cv2
import numpy as np

from legorec.segment import brick_area, cut_bricks, find_bricks
from legorec.segment.sheet import AREA_MM, MARKERS_MM

SCALE = 3  # pixels per mm of the fake photo
OFFSET = 30  # mm of white paper around marker 0


def _fake_sheet():
    """A flat sheet: 4 markers, a blue brick area and two red bricks on it."""
    photo = np.full(
        ((160 + 2 * OFFSET) * SCALE, (247 + 2 * OFFSET) * SCALE, 3), 255, np.uint8
    )
    aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = 30 * SCALE
    for i, (mx, my) in MARKERS_MM.items():
        cx, cy = (mx + OFFSET) * SCALE, (my + OFFSET) * SCALE
        marker = cv2.cvtColor(
            cv2.aruco.generateImageMarker(aruco, i, side), cv2.COLOR_GRAY2BGR
        )
        photo[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2] = marker
    x0, x1, y0, y1 = ((v + OFFSET) * SCALE for v in AREA_MM)
    photo[y0:y1, x0:x1] = (160, 90, 30)  # blue area (BGR)
    photo[y0 + 60 : y0 + 120, x0 + 60 : x0 + 180] = (40, 40, 200)  # red brick
    photo[y0 + 250 : y0 + 310, x0 + 100 : x0 + 160] = (40, 40, 200)  # red brick
    return photo


def test_brick_area_has_the_size_of_the_area():
    area = brick_area(_fake_sheet(), px_per_mm=4)
    x0, x1, y0, y1 = AREA_MM
    assert area.shape == ((y1 - y0) * 4, (x1 - x0) * 4, 3)


def test_no_markers_gives_none():
    assert brick_area(np.full((300, 300, 3), 255, np.uint8)) is None


def test_two_bricks_give_two_crops():
    crops = cut_bricks(brick_area(_fake_sheet(), px_per_mm=4))
    assert len(crops) == 2


def test_find_bricks_gives_one_mask_per_crop():
    area = brick_area(_fake_sheet(), px_per_mm=4)
    masks = find_bricks(area)
    assert len(masks) == len(cut_bricks(area)) == 2
    assert all(m.shape == area.shape[:2] and m.dtype == bool for m in masks)


def test_a_light_gradient_is_not_a_brick_with_the_local_background():
    """A dark corner (vignetting) differs from the border colour, but not from
    the paper around it: only the local background ignores it."""
    area = np.full((800, 600, 3), (160, 90, 30), np.uint8)
    yy, xx = np.mgrid[:800, :600]
    darker = np.clip((xx + yy - 900) / 500, 0, 1)[..., None]  # bottom-right corner
    area = (area * (1 - 0.6 * darker)).astype(np.uint8)
    area[100:160, 100:220] = (40, 40, 200)  # one red brick

    assert len(find_bricks(area, px_per_mm=4)) == 1
    assert len(find_bricks(area, px_per_mm=4, background="border")) == 2
