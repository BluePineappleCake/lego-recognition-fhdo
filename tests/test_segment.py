import cv2
import numpy as np

from legorec.segment import brick_area, cut_bricks
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
