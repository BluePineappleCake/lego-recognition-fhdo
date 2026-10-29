"""Turn an image of one brick into the input of the shape classifier.

The steps, in this order:

    1. find the brick       (brick_mask: which pixels belong to the brick?)
    2. measure its size     (longest side of its bounding box, in pixels)
    3. convert to grey      (the classifier only looks at the shape)
    4. optional: stretch the contrast inside the brick
    5. cut out a square around the brick and resize it to 96 x 96 pixels

The SAME code is used for the training renders and for real photos. If the two
were preprocessed differently, we would measure that difference instead of
the render-to-real gap.

Why step 2 exists: resizing every brick to 96 x 96 throws away how big it
really was, so a 1x3 and a 1x4 brick look alike. The measured size is returned
separately and can be given to the model (size_feature). See
docs/experiments.md.
"""

import cv2
import numpy as np
from PIL import Image

from legorec.types import BrickCrop

# Change this whenever the images produced here change. It is part of the cache
# file names, so a cache made with the old code is never reused by accident.
PREPROCESS_VERSION = "mask2"

DEFAULT_SIZE = 96  # side length of the classifier input in pixels
DEFAULT_MARGIN = 0.08  # free space around the brick: 8 % of its size per side

CONTRAST_MODES = ("none", "stretch")
STRETCH_BACKGROUND = 128  # grey value of the background in "stretch" mode


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _to_grey(image) -> np.ndarray:
    """PIL image or numpy array -> grey values as int16 array (H, W)."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.int16)

    array = np.asarray(image)
    if array.ndim == 3:
        # RGB -> brightness, same weights as PIL's "L" mode (ITU-R 601-2)
        red, green, blue = array[..., 0], array[..., 1], array[..., 2]
        array = red * 0.299 + green * 0.587 + blue * 0.114
    return array.astype(np.int16)


def _to_rgb(image) -> np.ndarray:
    """PIL image or numpy array -> uint8 RGB array (H, W, 3)."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    array = np.asarray(image)
    if array.ndim == 2:
        # grey image: use the same value for all three colour channels
        array = np.repeat(array[..., None], 3, axis=2)
    return array[..., :3].astype(np.uint8)


def _border_pixels(image: np.ndarray) -> np.ndarray:
    """All pixels of the outermost row and column on every side."""
    top, bottom = image[0], image[-1]
    left, right = image[:, 0], image[:, -1]
    return np.concatenate([top, bottom, left, right])


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
    """Smallest box around all True pixels: (top, bottom, left, right)."""
    rows, columns = np.nonzero(mask)
    return rows.min(), rows.max(), columns.min(), columns.max()


def _longest_side(box: tuple[int, int, int, int]) -> float:
    """Brick size in pixels = longer side of its bounding box."""
    top, bottom, left, right = box
    height = bottom - top + 1
    width = right - left + 1
    return float(max(height, width))


# --------------------------------------------------------------------------
# Step 1: find the brick
# --------------------------------------------------------------------------


def brick_mask(rgb: np.ndarray, thr: float = 12.0) -> np.ndarray:
    """RGB image (H, W, 3) -> True for every pixel of the brick.

    Idea: the image border shows the background. Everything whose COLOUR is
    clearly different from it belongs to the brick.

    1. Colour distance in the Lab colour space instead of grey values:
       a white brick on blue card is easy to separate in colour, but in grey
       both can have almost the same brightness.
    2. "Closing" fills small holes and gaps, e.g. from bright highlights on
       the studs that look like background.
    3. Only the largest connected region counts: a stripe at the image edge,
       crumbs or dust are smaller and are dropped.

    `thr`: how different (in Lab units) a pixel must be to count as brick.
    Returns an all-False mask if nothing stands out from the background.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # background colour = median colour of the image border
    background_colour = np.median(_border_pixels(lab), axis=0)

    # how far is every pixel's colour from the background colour?
    colour_distance = np.linalg.norm(lab - background_colour, axis=2)
    mask = (colour_distance > thr).astype(np.uint8)

    # close holes and gaps; kernel size ~2 % of the image
    kernel = _odd_kernel(int(0.02 * max(mask.shape)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # label every connected region; label 0 is the background
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return np.zeros(mask.shape, dtype=bool)

    region_areas = stats[1:, cv2.CC_STAT_AREA]  # areas of labels 1, 2, ...
    largest_label = 1 + int(np.argmax(region_areas))
    return labels == largest_label


# --------------------------------------------------------------------------
# Step 4: optional contrast stretch
# --------------------------------------------------------------------------


def _apply_contrast(grey, mask, background, contrast):
    """contrast="none": change nothing.

    contrast="stretch": spread the grey values INSIDE the brick over the full
    range 0..255, so a red brick is no longer almost black and a yellow one
    no longer almost white. Example: if the brick only uses grey values 40..90,
    40 becomes 0 and 90 becomes 255. (The 2nd and 98th percentile are used
    instead of min and max, so a few extreme pixels do not decide.)

    Everything outside the brick becomes one fixed grey value. Otherwise the
    background would be stretched too and look different in every photo.

    Returns (new grey image, background value to use for padding).
    """
    if contrast not in CONTRAST_MODES:
        raise ValueError(f"contrast must be one of {CONTRAST_MODES}, not {contrast!r}")
    if contrast == "none":
        return grey, background

    # Use a slightly smaller mask: the brick edge is a light seam (smoothing,
    # JPEG). If it stayed inside, it would set the upper percentile and turn
    # into a white outline that renders never have.
    brick_size = _longest_side(_bounding_box(mask))
    kernel = _odd_kernel(int(0.03 * brick_size))
    inner = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    if not inner.any():  # very small brick: shrinking removed everything
        inner = mask

    darkest, brightest = np.percentile(grey[inner], [2, 98])
    if brightest - darkest >= 1:  # a completely flat brick has nothing to stretch
        grey = (grey.astype(np.float32) - darkest) * (255.0 / (brightest - darkest))

    grey = np.where(inner, grey, STRETCH_BACKGROUND)
    grey = np.clip(grey, 0, 255).astype(np.int16)
    return grey, STRETCH_BACKGROUND


# --------------------------------------------------------------------------
# Step 5: cut out a square and resize
# --------------------------------------------------------------------------


def _cut_out_square(grey, box, background, size, margin) -> np.ndarray:
    """Square around `box` (plus margin), resized to size x size, as uint8.

    Parts of the square that lie outside the image are filled with the
    background value, so a brick at the image edge still ends up centred.
    """
    top, bottom, left, right = box
    brick_side = max(bottom - top, right - left)
    square_side = int(brick_side * (1 + 2 * margin)) + 1

    centre_row = (top + bottom) // 2
    centre_column = (left + right) // 2

    # start with a square full of background ...
    square = np.full((square_side, square_side), background, dtype=np.int16)

    # ... take the part of the image that lies inside the square ...
    first_row = max(centre_row - square_side // 2, 0)
    first_column = max(centre_column - square_side // 2, 0)
    piece = grey[
        first_row : first_row + square_side,
        first_column : first_column + square_side,
    ]

    # ... and paste it into the middle of the square
    row_offset = (square_side - piece.shape[0]) // 2
    column_offset = (square_side - piece.shape[1]) // 2
    square[
        row_offset : row_offset + piece.shape[0],
        column_offset : column_offset + piece.shape[1],
    ] = piece

    square_image = Image.fromarray(np.clip(square, 0, 255).astype(np.uint8))
    return np.asarray(square_image.resize((size, size), Image.BILINEAR))


def _resize_whole_image(grey, size) -> np.ndarray:
    image = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
    return np.asarray(image.resize((size, size), Image.BILINEAR))


# --------------------------------------------------------------------------
# The two entry points
# --------------------------------------------------------------------------


def preprocess_image(
    img, size=DEFAULT_SIZE, margin=DEFAULT_MARGIN, thr=12.0, contrast="none"
):
    """A photo or render with ONE brick on a plain background.

    Used for the training renders and for single-brick photos.

    Returns (uint8 array size x size, brick size in pixels).
    Example: preprocess_image(Image.open("3001.jpg")) -> (array 96 x 96, 305.0)
    """
    grey = _to_grey(img)
    mask = brick_mask(_to_rgb(img), thr)

    if not mask.any():
        # nothing stands out from the background: use the whole image
        return _resize_whole_image(grey, size), float(max(grey.shape))

    background = int(np.median(_border_pixels(grey)))
    grey, background = _apply_contrast(grey, mask, background, contrast)

    box = _bounding_box(mask)
    brick_size = _longest_side(box)
    return _cut_out_square(grey, box, background, size, margin), brick_size


def preprocess_crop(
    crop: BrickCrop, size=DEFAULT_SIZE, margin=DEFAULT_MARGIN, contrast="none"
):
    """A brick that the segmentation step has already cut out of a photo.

    The segmentation provides the mask, so step 1 is skipped. Pixels outside
    the mask (e.g. a neighbouring brick) are replaced by one plain background
    value, so the result looks like a single brick on a plain background.

    Returns (uint8 array size x size, brick size in pixels of the photo).
    """
    grey = _to_grey(crop.image)
    mask = np.asarray(crop.mask, dtype=bool)

    if not mask.any():
        # empty mask: fall back to finding the brick ourselves
        return preprocess_image(crop.image, size, margin, contrast=contrast)

    outside = ~mask
    background = int(np.median(grey[outside])) if outside.any() else 255
    grey = np.where(mask, grey, background)
    grey, background = _apply_contrast(grey, mask, background, contrast)

    box = _bounding_box(mask)
    brick_size = _longest_side(box)
    return _cut_out_square(grey, box, background, size, margin), brick_size
