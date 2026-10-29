"""Turn an image of one brick into the classifier input.

grey -> crop to the brick -> pad to a square -> resize

The SAME code must be used for renders and for real crops, otherwise we measure
a preprocessing difference instead of the render-to-real gap.

Both functions also return the brick size in pixels (longest side of its
bounding box) measured BEFORE the resize, because the resize removes the
absolute size (1x3 and 1x4 bricks then look alike). See docs/experiments.md.
"""

import cv2
import numpy as np
from PIL import Image

from legorec.types import BrickCrop

# Bump this whenever the output of the preprocessing changes: it is part of the
# cache file names, so old caches are not silently reused.
PREPROCESS_VERSION = "mask2"

DEFAULT_SIZE = 96
DEFAULT_MARGIN = 0.08


def _to_grey(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.int16)
    a = np.asarray(img)
    if a.ndim == 3:  # RGB -> luminance (ITU-R 601-2, same as PIL "L")
        a = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    return a.astype(np.int16)


def _to_rgb(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    a = np.asarray(img)
    if a.ndim == 2:  # grey -> 3 identical channels
        a = np.repeat(a[..., None], 3, 2)
    return a[..., :3].astype(np.uint8)


def brick_mask(rgb: np.ndarray, thr: float = 12.0) -> np.ndarray:
    """RGB image (H, W, 3) -> bool mask of the single largest object.

    1. Colour distance (Lab) to the background colour instead of grey values:
       a white brick on blue card is easy to separate in colour, not in grey.
    2. Closing fills holes and gaps from highlights and bright edges.
    3. Only the largest connected region counts: stripes, crumbs and dust at
       the edge are dropped.
    Returns an all-False mask if nothing stands out from the background.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)  # background colour
    dist = np.linalg.norm(lab - bg, axis=2)  # colour distance per pixel
    mask = (dist > thr).astype(np.uint8)
    k = max(3, int(0.02 * max(mask.shape))) | 1  # ~2 % of the image, odd so
    # the kernel is centred (an even size shifts the mask by one pixel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:  # label 0 is the background, nothing else found
        return np.zeros(mask.shape, bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


CONTRAST_MODES = ("none", "stretch")
STRETCH_BG = 128  # background grey value in "stretch" mode


def _apply_contrast(g, mask, bg, contrast):
    """contrast="stretch": spread the grey values INSIDE the brick mask over the
    full range 0..255 (2nd and 98th percentile -> 0 and 255), so a red brick is
    no longer almost black and a yellow one no longer almost white. Pixels
    outside the (slightly shrunk) mask become one fixed background value,
    otherwise the background would be stretched too and look different in
    every photo. The measured brick size is not affected.

    Returns (grey image, background value for padding).
    """
    if contrast not in CONTRAST_MODES:
        raise ValueError(f"contrast must be one of {CONTRAST_MODES}, not {contrast!r}")
    if contrast == "none":
        return g, bg
    # Shrink the mask a little: the brick edge is a light seam (anti-aliasing,
    # JPEG). Left inside, it would set the 98th percentile and turn into a
    # white outline that renders never have.
    ys, xs = np.nonzero(mask)
    size = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    k = max(3, int(0.03 * size)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    if not inner.any():  # tiny brick: keep the full mask
        inner = mask
    lo, hi = np.percentile(g[inner], [2, 98])
    if hi - lo >= 1:  # a completely flat brick has nothing to stretch
        g = (g.astype(np.float32) - lo) * (255.0 / (hi - lo))
    g = np.clip(np.where(inner, g, STRETCH_BG), 0, 255).astype(np.int16)
    return g, STRETCH_BG


def _square_crop(g, y0, y1, x0, x1, bg, size, margin) -> np.ndarray:
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin)) + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    canvas = np.full((side, side), bg, np.int16)
    sy0, sx0 = max(cy - side // 2, 0), max(cx - side // 2, 0)
    crop = g[sy0 : sy0 + side, sx0 : sx0 + side]
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    return np.asarray(out.resize((size, size), Image.BILINEAR))


def preprocess_image(
    img, size=DEFAULT_SIZE, margin=DEFAULT_MARGIN, thr=12.0, contrast="none"
):
    """One brick on a plain background (renders, single-brick photos).

    The brick is the largest region whose colour differs from the background
    by more than `thr` (see brick_mask). `contrast`: see _apply_contrast.

    Returns (uint8 array size x size, brick size in px).
    """
    g = _to_grey(img)
    mask = brick_mask(_to_rgb(img), thr)
    if not mask.any():  # nothing found: use the whole image
        out = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
        return np.asarray(out.resize((size, size), Image.BILINEAR)), float(max(g.shape))
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = int(np.median(border))
    g, bg = _apply_contrast(g, mask, bg, contrast)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    px = float(max(y1 - y0, x1 - x0) + 1)
    return _square_crop(g, y0, y1, x0, x1, bg, size, margin), px


def preprocess_crop(
    crop: BrickCrop, size=DEFAULT_SIZE, margin=DEFAULT_MARGIN, contrast="none"
):
    """A brick cut out by the segmentation step.

    Uses the segmentation mask instead of a background threshold. Pixels outside
    the mask are replaced by one plain background value, like in the renders.

    Returns (uint8 array size x size, brick size in px of the original photo).
    """
    g = _to_grey(crop.image)
    mask = np.asarray(crop.mask, bool)
    if not mask.any():
        return preprocess_image(crop.image, size, margin, contrast=contrast)
    bg = int(np.median(g[~mask])) if (~mask).any() else 255
    g = np.where(mask, g, bg)
    g, bg = _apply_contrast(g, mask, bg, contrast)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    px = float(max(y1 - y0, x1 - x0) + 1)
    return _square_crop(g, y0, y1, x0, x1, bg, size, margin), px
