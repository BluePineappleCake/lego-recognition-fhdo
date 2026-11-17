"""Cut single bricks out of a photo of the capture sheet (template v2).

Two steps:

    1. brick_area: find the 4 ArUco markers and warp the coloured brick area
       to a flat, top-down image at PX_PER_MM pixels per millimetre.
    2. cut_bricks: everything that differs in colour from the area's border
       is a brick; return one crop per brick, the rest filled with the
       background colour.

Known weak points (first test on 838 crops, 2026-11-16): bricks that touch
are returned as one crop, and bricks close to the background colour can
be split or missed. About a quarter of the crops were not a single brick.
"""

import cv2
import numpy as np

PX_PER_MM = 8
# centres of the markers on the printed sheet in mm, relative to marker 0
MARKERS_MM = {0: (0, 0), 1: (247, 0), 2: (247, 160), 3: (0, 160)}
# brick area relative to marker 0 in mm: x from, x to, y from, y to
AREA_MM = (25, 160, 15, 170)

_DETECTOR = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters(),
)


def brick_area(photo, px_per_mm=PX_PER_MM):
    """Return the brick area of a BGR photo, warped flat, or None.

    None means fewer than 4 markers were found.
    """
    corners, ids, _ = _DETECTOR.detectMarkers(photo)
    if ids is None:
        return None
    found = {
        int(i): c[0].mean(axis=0)
        for c, i in zip(corners, ids.flatten())
        if int(i) in MARKERS_MM
    }
    if len(found) < 4:
        return None
    seen = np.array([found[i] for i in sorted(found)], np.float64)
    wanted = np.array([MARKERS_MM[i] for i in sorted(found)], np.float64) * px_per_mm
    H, _ = cv2.findHomography(seen, wanted)
    x0, x1, y0, y1 = AREA_MM
    shift = np.array(
        [[1, 0, -x0 * px_per_mm], [0, 1, -y0 * px_per_mm], [0, 0, 1]], np.float64
    )
    size = ((x1 - x0) * px_per_mm, (y1 - y0) * px_per_mm)
    return cv2.warpPerspective(photo, shift @ H, size)


def cut_bricks(area, min_pixels=1500, margin=25, threshold=12, border=20):
    """Return one BGR crop per brick found on the brick area.

    A pixel belongs to a brick if its Lab colour is more than `threshold`
    away from the median colour of the area's border. Blobs smaller than
    `min_pixels` are dust. Crops are ordered as OpenCV labels them (top to
    bottom), which gives the _b01, _b02, ... numbers of the labelled crops.
    """
    lab = cv2.cvtColor(area, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = border
    edge = np.concatenate(
        [
            lab[:b].reshape(-1, 3),
            lab[-b:].reshape(-1, 3),
            lab[:, :b].reshape(-1, 3),
            lab[:, -b:].reshape(-1, 3),
        ]
    )
    is_brick = np.linalg.norm(lab - np.median(edge, axis=0), axis=2) > threshold
    is_brick = cv2.morphologyEx(
        is_brick.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(is_brick)
    fill = np.median(area[:b].reshape(-1, 3), axis=0).astype(np.uint8)
    crops = []
    for i in range(1, n):  # 0 is the background
        x, y, w, h, size = stats[i]
        if size < min_pixels:
            continue
        only_this = np.where((labels == i)[..., None], area, fill)
        crops.append(
            only_this[
                max(0, y - margin) : y + h + margin, max(0, x - margin) : x + w + margin
            ]
        )
    return crops
