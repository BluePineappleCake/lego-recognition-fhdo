"""Images with one brick each (internet photos, renders) -> BrickCrops.

    crops, n_fallbacks = single_brick_crops(images)

The training images come without a mask. They get one the same way as the
bricks on our sheet photos: YOLO finds the brick (single_brick_masks). Where
YOLO finds nothing (often a light brick on a light background), brick_mask
takes over: everything whose colour differs clearly from the image border.

Then every image goes through legorec.shape.preprocess.preprocess_crop like a
crop of a sheet photo. If training images were cut out differently from the
sheet photos, the model would learn that difference.
"""

import cv2
import numpy as np
from PIL import Image

from legorec.segment.bricks import load_yolo
from legorec.segment.sheet import PX_PER_MM
from legorec.types import BrickCrop

# a brick is put on a canvas at about the size it has on a sheet (20 mm at
# PX_PER_MM), so YOLO sees what it was trained on
SINGLE_LONG_SIDE = 20 * PX_PER_MM
SINGLE_CANVAS = 512


def single_brick_crops(images, model=None) -> tuple[list[BrickCrop], int]:
    """PIL images or RGB arrays -> (one BrickCrop per image, the number of
    images where YOLO found nothing and brick_mask was used).

    The crop is the whole image; only its mask says where the brick is.
    """
    if not images:
        return [], 0
    rgbs = [_to_rgb(image) for image in images]
    crops, fallbacks = [], 0
    for rgb, mask in zip(rgbs, single_brick_masks(rgbs, model)):
        if mask is None:
            mask = brick_mask(rgb)
            fallbacks += 1
        h, w = mask.shape
        crops.append(BrickCrop(rgb, mask, (0, 0, w, h)))
    return crops, fallbacks


def single_brick_masks(
    images_rgb: list[np.ndarray], model=None, conf: float = 0.25, batch_size: int = 32
) -> list[np.ndarray | None]:
    """Images with one brick each (internet photos, renders) -> YOLO masks.

    For training images that come without a mask. YOLO only knows sheet
    photos, so every image is scaled to a long side of SINGLE_LONG_SIDE
    pixels and put in the middle of a SINGLE_CANVAS square filled with the
    colour of its border. The surest mask is taken and scaled back to the
    image. None where YOLO finds nothing (often a light brick on a light
    background).

    Unlike brick_mask, YOLO leaves the shadow out, as on the sheet photos.
    """
    yolo = load_yolo(model)
    canvases, places = [], []
    for rgb in images_rgb:
        h, w = rgb.shape[:2]
        scale = SINGLE_LONG_SIDE / max(h, w)
        small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
        sh, sw = small.shape[:2]
        edge = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
        canvas = np.empty((SINGLE_CANVAS, SINGLE_CANVAS, 3), np.uint8)
        canvas[:] = np.median(edge, axis=0)
        y0, x0 = (SINGLE_CANVAS - sh) // 2, (SINGLE_CANVAS - sw) // 2
        canvas[y0 : y0 + sh, x0 : x0 + sw] = small
        canvases.append(canvas[:, :, ::-1])  # YOLO was trained on BGR (cv2)
        places.append((y0, x0, sh, sw, h, w))

    masks = []
    for start in range(0, len(canvases), batch_size):
        results = yolo.predict(
            canvases[start : start + batch_size],
            imgsz=SINGLE_CANVAS,
            conf=conf,
            retina_masks=True,
            verbose=False,
        )
        for result, (y0, x0, sh, sw, h, w) in zip(
            results, places[start : start + batch_size]
        ):
            if result.masks is None or not len(result.masks):
                masks.append(None)
                continue
            best = int(result.boxes.conf.argmax())
            mask = result.masks.data[best].cpu().numpy()[y0 : y0 + sh, x0 : x0 + sw]
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
            masks.append(mask.astype(bool) if mask.any() else None)
    return masks


def brick_mask(rgb: np.ndarray, thr: float = 12.0) -> np.ndarray:
    """RGB image (H, W, 3) -> True for every pixel of the brick, by colour.

    Idea: the image border shows the background, and everything whose colour
    is clearly different from it belongs to the brick.

    1. Colour distance in the Lab colour space instead of grey values:
       a white brick on blue card is easy to separate in colour, but in grey
       both can have almost the same brightness.
    2. "Closing" fills small holes and gaps, e.g. from bright highlights on
       the studs that look like background.
    3. Only the largest connected region counts: a stripe at the image edge,
       crumbs or dust are smaller and are dropped.

    `thr`: minimum colour distance to the background, in OpenCV's 8-bit Lab
    units. OpenCV scales L to 0..255, so a difference in lightness counts about
    2.5 times more than in standard Lab. Returns an all-False mask if nothing
    stands out from the background.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background_colour = np.median(_border_pixels(lab), axis=0)
    colour_distance = np.linalg.norm(lab - background_colour, axis=2)
    mask = (colour_distance > thr).astype(np.uint8)

    kernel = _odd_kernel(int(0.02 * max(mask.shape)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # OpenCV gives label 0 to the background, the regions get 1, 2, ...
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return np.zeros(mask.shape, dtype=bool)

    region_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(region_areas))
    return labels == largest_label


def _to_rgb(image) -> np.ndarray:
    """PIL image or numpy array -> uint8 RGB array (H, W, 3)."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    return array[..., :3].astype(np.uint8)


def _border_pixels(image: np.ndarray) -> np.ndarray:
    """All pixels of the outermost row and column on every side."""
    return np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])


def _odd_kernel(size_in_pixels: int) -> np.ndarray:
    """Round structuring element with an odd size (at least 3), see
    legorec.shape.preprocess."""
    size = max(3, size_in_pixels) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
