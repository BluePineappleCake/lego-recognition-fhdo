"""Turn an image of one brick into the classifier input.

grey -> crop to the brick -> pad to a square -> resize

The SAME code must be used for renders and for real crops, otherwise we measure
a preprocessing difference instead of the render-to-real gap.

Both functions also return the brick size in pixels (longest side of its
bounding box) measured BEFORE the resize, because the resize removes the
absolute size (1x3 and 1x4 bricks then look alike). See docs/experiments.md.
"""

import numpy as np
from PIL import Image

from legorec.types import BrickCrop

DEFAULT_SIZE = 96
DEFAULT_MARGIN = 0.08


def _to_grey(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.int16)
    a = np.asarray(img)
    if a.ndim == 3:  # RGB -> luminance (ITU-R 601-2, same as PIL "L")
        a = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    return a.astype(np.int16)


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


def preprocess_image(img, size=DEFAULT_SIZE, margin=DEFAULT_MARGIN, thr=20):
    """One brick on a plain background (renders, single-brick photos).

    The background value is estimated from the image border; every pixel that
    differs by more than `thr` counts as brick.

    Returns (uint8 array size x size, brick size in px).
    """
    g = _to_grey(img)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = int(np.median(border))
    ys, xs = np.nonzero(np.abs(g - bg) > thr)
    if not len(xs):  # nothing found: use the whole image
        out = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
        return np.asarray(out.resize((size, size), Image.BILINEAR)), float(max(g.shape))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    px = float(max(y1 - y0, x1 - x0) + 1)
    return _square_crop(g, y0, y1, x0, x1, bg, size, margin), px


def preprocess_crop(crop: BrickCrop, size=DEFAULT_SIZE, margin=DEFAULT_MARGIN):
    """A brick cut out by the segmentation step.

    Uses the segmentation mask instead of a background threshold. Pixels outside
    the mask are replaced by one plain background value, like in the renders.

    Returns (uint8 array size x size, brick size in px of the original photo).
    """
    g = _to_grey(crop.image)
    mask = np.asarray(crop.mask, bool)
    if not mask.any():
        return preprocess_image(crop.image, size, margin)
    bg = int(np.median(g[~mask])) if (~mask).any() else 255
    g = np.where(mask, g, bg)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    px = float(max(y1 - y0, x1 - x0) + 1)
    return _square_crop(g, y0, y1, x0, x1, bg, size, margin), px
