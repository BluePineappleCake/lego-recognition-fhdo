"""The capture sheet (template v2): find the 4 ArUco markers and warp the
coloured brick area to a flat, top-down image at PX_PER_MM pixels per mm.

On that image every brick has its real size: 8 pixels are 1 mm. The bricks
themselves are found by YOLO (legorec.segment.bricks).

The colour-threshold finder used before YOLO (find_bricks) is in the Git tag
v1-legacy.
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
