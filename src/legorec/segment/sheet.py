"""Cut single bricks out of a photo of the capture sheet (template v2).

Two steps:

    1. brick_area: find the 4 ArUco markers and warp the coloured brick area
       to a flat, top-down image at PX_PER_MM pixels per millimetre.
    2. find_bricks: everything that differs in colour from the background
       (estimated locally around every pixel) is a brick; one mask per brick.
       cut_bricks turns the masks into crops, the rest filled with the
       background colour.

History: the first crops (2026-11-16, 838 crops) compared every pixel with
one colour from the area's border. Light gradients and shadows became
"bricks": about a quarter of the crops were not a single brick. That method
is still there as background="border". Bricks that touch still come back as
one mask: that is what the YOLO model (segment.yolo) is for.
"""

import cv2
import numpy as np

PX_PER_MM = 8
# centres of the markers on the printed sheet in mm, relative to marker 0
MARKERS_MM = {0: (0, 0), 1: (247, 0), 2: (247, 160), 3: (0, 160)}
# brick area relative to marker 0 in mm: x from, x to, y from, y to
AREA_MM = (25, 160, 15, 170)
FRAME = 0.9  # a blob as wide or high as 90 % of the area is its edge, not a brick
# window for the local background colour; larger than a brick (2x4: 32 x 16 mm)
BACKGROUND_WINDOW_MM = 41

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


def background_colour(area, background="local", border=20, px_per_mm=PX_PER_MM):
    """Lab colour (float32) the area would have without bricks, per pixel.

    "local": median in a BACKGROUND_WINDOW_MM window around every pixel,
    computed at 1 px/mm. Follows light gradients, vignetting and shadows of
    the room, which the "border" colour cannot. Bricks are smaller than the
    window, so they do not shift the median. Dense piles would.
    "border": one colour for the whole area, the median of its border (the
    method of the first crops, 2026-11-16; keeps their _bNN numbers).
    """
    lab = cv2.cvtColor(area, cv2.COLOR_BGR2LAB)
    if background == "border":
        b = border
        edge = np.concatenate(
            [lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)]
        )  # fmt: skip
        return np.broadcast_to(np.median(edge, axis=0), lab.shape).astype(np.float32)
    if background != "local":
        raise ValueError(f"background must be 'local' or 'border', not {background!r}")
    small = cv2.resize(
        lab, None, fx=1 / px_per_mm, fy=1 / px_per_mm, interpolation=cv2.INTER_AREA
    )
    window = BACKGROUND_WINDOW_MM | 1  # medianBlur needs an odd size
    small = cv2.merge([cv2.medianBlur(channel, window) for channel in cv2.split(small)])
    return cv2.resize(
        small, (area.shape[1], area.shape[0]), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)


def find_bricks(
    area,
    min_pixels=1500,
    threshold=12,
    border=20,
    background="local",
    px_per_mm=PX_PER_MM,
):
    """Return one boolean mask (same size as `area`) per brick.

    A pixel belongs to a brick if its Lab colour is more than `threshold`
    away from the background colour (see background_colour). Blobs smaller
    than `min_pixels` are dust. Masks are ordered as OpenCV labels them (top
    to bottom); with background="border" that gives the _b01, _b02, ...
    numbers of the first labelled crops.

    Check of "local" against our hand-checked crops (2026-11-19, 246 photos):
    all 175 crops we had thrown away (light gradients, shadows, paper
    texture) are gone, 594 of 604 kept bricks are found; the rest were
    bricks merged with such blobs, which are now found on their own. Still
    weak: a white brick on white paper (only its shadow is found) and
    bricks that touch (one mask).
    """
    lab = cv2.cvtColor(area, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference = background_colour(area, background, border, px_per_mm)
    is_brick = (np.linalg.norm(lab - reference, axis=2) > threshold).astype(np.uint8)
    if background == "local":  # remove single noisy pixels of the paper grain
        is_brick = cv2.morphologyEx(is_brick, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    is_brick = cv2.morphologyEx(is_brick, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(is_brick)
    return [
        labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_pixels
    ]


def is_frame(mask: np.ndarray) -> bool:
    """The warp is never perfect: a thin line of the paper around the coloured
    area can survive as one big ring-shaped blob. That is not a brick."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (xs.max() - xs.min()) > FRAME * w or (ys.max() - ys.min()) > FRAME * h


def cut_bricks(
    area, min_pixels=1500, margin=25, threshold=12, border=20, background="local"
):
    """Return one BGR crop per brick found on the brick area (see find_bricks).

    The rest of each crop is filled with the background colour.
    """
    fill = np.median(area[:border].reshape(-1, 3), axis=0).astype(np.uint8)
    crops = []
    for mask in find_bricks(area, min_pixels, threshold, border, background):
        ys, xs = np.nonzero(mask)
        y, x = ys.min(), xs.min()
        h, w = ys.max() - y + 1, xs.max() - x + 1
        only_this = np.where(mask[..., None], area, fill)
        crops.append(
            only_this[
                max(0, y - margin) : y + h + margin, max(0, x - margin) : x + w + margin
            ]
        )
    return crops
