"""Turn one brick into the input of the shape classifier.

    preprocess_crop(crop) -> uint8 array 96 x 96

A BrickCrop is an image plus a mask that says which pixels are the brick. It
comes from a sheet photo (legorec.segment.bricks_on_sheet) or from an image
with one brick (legorec.segment.single, the training images). The steps:

    1. convert to grey      (the classifier only looks at the shape)
    2. stretch the contrast inside the brick, background a fixed grey
    3. cut out a square around the brick and resize it to 96 x 96 pixels

Renders and real photos go through the same code. If they were preprocessed
differently, we would measure that difference instead of the render-to-real gap.
"""

import cv2
import numpy as np
from PIL import Image

from legorec.types import BrickCrop

# Change this whenever the images produced here change. It is part of the cache
# file names, so a cache made with the old code is never reused by accident.
# mask2: robust brick mask (#14). mask2-pad: square padded at the image edge.
# mask2-pad-grey: numpy images get exactly PIL's grey values (#36).
PREPROCESS_VERSION = "mask2-pad-grey"

DEFAULT_SIZE = 96  # side length of the classifier input in pixels
DEFAULT_MARGIN = 0.08  # free space around the brick: 8 % of its size per side
BACKGROUND = 128  # grey value of everything outside the brick


def preprocess_crop(
    crop: BrickCrop, size: int = DEFAULT_SIZE, margin: float = DEFAULT_MARGIN
) -> np.ndarray:
    """One BrickCrop (image + mask of the brick) -> uint8 array size x size.

    The segmentation provides the mask. Pixels outside it (e.g. a neighbouring
    brick) become the plain background, so the result looks like a single
    brick on a plain background.
    """
    mask = np.asarray(crop.mask, dtype=bool)
    grey = _to_grey(crop.image)
    if not mask.any():  # nothing found: the whole image, as it is
        return _resize_whole_image(grey, size)
    return _cut_out_square(_stretch_contrast(grey, mask), mask, size, margin)


def _stretch_contrast(grey, mask) -> np.ndarray:
    """Step 2: spread the grey values inside the brick over the full range
    0..255, so a red brick is no longer almost black and a yellow one no longer
    almost white. Example: if the brick only uses grey values 40..90, 40
    becomes 0 and 90 becomes 255. The 2nd and 98th percentile are used instead
    of min and max, so a few extreme pixels do not decide.

    Everything outside the brick becomes BACKGROUND. Otherwise the background
    would be stretched too and look different in every photo.
    """
    # Use a slightly smaller mask: the brick edge is a light seam (smoothing,
    # JPEG). If it stayed inside, it would set the upper percentile and turn
    # into a white outline that renders never have.
    brick_size = _longest_side(_bounding_box(mask))
    kernel = _odd_kernel(int(0.03 * brick_size))
    inner = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    if not inner.any():  # tiny brick: shrinking removed everything
        inner = mask

    darkest, brightest = np.percentile(grey[inner], [2, 98])
    if brightest - darkest >= 1:  # a completely flat brick has nothing to stretch
        grey = (grey.astype(np.float32) - darkest) * (255.0 / (brightest - darkest))

    grey = np.where(inner, grey, BACKGROUND)
    return np.clip(grey, 0, 255).astype(np.int16)


def _cut_out_square(grey, mask, size, margin) -> np.ndarray:
    """Step 3: square around the brick (plus margin), resized to size x size,
    as uint8.

    The square is always centred on the brick. Parts of it that lie outside
    the image are filled with BACKGROUND, so a brick at the image edge still
    ends up in the middle of the result.
    """
    top, bottom, left, right = _bounding_box(mask)
    brick_side = max(bottom - top, right - left)
    square_side = int(brick_side * (1 + 2 * margin)) + 1

    # top-left corner of the square in image coordinates; negative if the
    # square sticks out over the top or left edge of the image
    square_top = (top + bottom) // 2 - square_side // 2
    square_left = (left + right) // 2 - square_side // 2

    # copy the part of the square that lies inside the image (end = exclusive)
    # to the same position in a square full of background
    first_row, first_column = max(square_top, 0), max(square_left, 0)
    end_row = min(square_top + square_side, grey.shape[0])
    end_column = min(square_left + square_side, grey.shape[1])
    square = np.full((square_side, square_side), BACKGROUND, dtype=np.int16)
    square[
        first_row - square_top : end_row - square_top,
        first_column - square_left : end_column - square_left,
    ] = grey[first_row:end_row, first_column:end_column]

    square_image = Image.fromarray(np.clip(square, 0, 255).astype(np.uint8))
    return np.asarray(square_image.resize((size, size), Image.BILINEAR))


def _resize_whole_image(grey, size) -> np.ndarray:
    image = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
    return np.asarray(image.resize((size, size), Image.BILINEAR))


# Small helpers


def _to_grey(image) -> np.ndarray:
    """PIL image or numpy array -> grey values as int16 array (H, W)."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.int16)

    array = np.asarray(image)
    if array.ndim == 3:
        # exactly PIL's "L" mode (ITU-R 601-2 weights in 16-bit fixed point,
        # rounded), so a photo gives the same grey values whether it arrives
        # as a PIL image (renders) or as a numpy array (BrickCrop)
        rgb = array[..., :3].astype(np.int32)
        array = (
            rgb[..., 0] * 19595 + rgb[..., 1] * 38470 + rgb[..., 2] * 7471 + 0x8000
        ) >> 16
    return array.astype(np.int16)


def _odd_kernel(size_in_pixels: int) -> np.ndarray:
    """Round structuring element for OpenCV's morphology functions.

    The size must be odd: with an even size the kernel has no centre pixel and
    OpenCV shifts the result by one pixel.
    """
    size = max(3, size_in_pixels)
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Smallest box around all True pixels: (top, bottom, left, right), inclusive."""
    rows, columns = np.nonzero(mask)
    return rows.min(), rows.max(), columns.min(), columns.max()


def _longest_side(box: tuple[int, int, int, int]) -> float:
    """Longer side of a bounding box in pixels."""
    top, bottom, left, right = box
    height = bottom - top + 1
    width = right - left + 1
    return float(max(height, width))
