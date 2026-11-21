"""Sheet photo in, one BrickCrop per brick out: the segmentation step of the
pipeline.

    from legorec.segment import bricks_on_sheet

    crops = bricks_on_sheet(cv2.imread("IMG_3482.jpg"))

YOLO-seg (one class "brick") finds the bricks on the brick area, warped flat
with the 4 ArUco markers (brick_area). The bbox of a crop is in pixels of that
area (PX_PER_MM per mm), not of the raw photo.

Needs Ultralytics (pip install -e .[detect]) and a trained model at
models/bricks_yolo.pt (or $LEGOREC_BRICK_MODEL): copy
runs/segment/train/weights/best.pt there. The first model was trained on the
outlines of the old colour-threshold finder plus some corrected by hand
(docs/experiments.md, 2026-11-20); new labels start from its own drafts
(scripts/export_yolo.py).
"""

from pathlib import Path

import cv2
import numpy as np

from legorec.paths import brick_model_path
from legorec.segment.sheet import PX_PER_MM, brick_area
from legorec.types import BrickCrop

IMGSZ = 1024  # as in training
MARGIN = 25  # pixels around each brick (about 3 mm)
INSIDE = 0.5  # a mask more than half inside a surer one is dropped

# single_brick_masks: a brick is put on a canvas at about the size it has on
# a sheet (20 mm at PX_PER_MM), so YOLO sees what it was trained on
SINGLE_LONG_SIDE = 20 * PX_PER_MM
SINGLE_CANVAS = 512

_models = {}  # path -> loaded YOLO model


def _yolo(path: Path):
    if path not in _models:
        from ultralytics import YOLO  # here: optional dependency

        if not path.exists():
            raise FileNotFoundError(
                f"no brick model at {path}: copy runs/segment/train/weights/best.pt "
                "there or set LEGOREC_BRICK_MODEL"
            )
        _models[path] = YOLO(str(path))
    return _models[path]


def brick_model_id(model=None) -> str:
    """Short fingerprint of the YOLO model file, for cache names."""
    import hashlib

    path = Path(model or brick_model_path())
    return hashlib.md5(path.read_bytes()).hexdigest()[:6]


def single_brick_masks(
    images_rgb: list[np.ndarray], model=None, conf: float = 0.25, batch_size: int = 32
) -> list[np.ndarray | None]:
    """Images with one brick each (internet photos, renders) -> YOLO masks.

    For training images that come without a mask. YOLO only knows sheet
    photos, so every image is scaled to a long side of SINGLE_LONG_SIDE
    pixels and put in the middle of a SINGLE_CANVAS square filled with the
    colour of its border. The surest mask is taken and scaled back to the
    image. None where YOLO finds nothing (often a light brick on a light
    background): the caller falls back to preprocess.brick_mask.

    Unlike brick_mask, YOLO leaves the shadow out, as on the sheet photos.
    """
    yolo = _yolo(Path(model or brick_model_path()))
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


def brick_masks(area: np.ndarray, model=None, conf: float = 0.25) -> list[np.ndarray]:
    """Warped brick area (BGR) -> one boolean mask (area size) per brick."""
    result = _yolo(Path(model or brick_model_path())).predict(
        area, imgsz=IMGSZ, conf=conf, retina_masks=True, verbose=False
    )[0]
    if result.masks is None:
        return []
    masks = result.masks.data.cpu().numpy().astype(bool)
    return drop_inside(list(masks), result.boxes.conf.cpu().numpy())


def drop_inside(masks: list[np.ndarray], confidences) -> list[np.ndarray]:
    """Surest mask first; drop a mask that lies more than INSIDE in a surer one.

    The model sometimes adds a piece of a long brick (1x6) as a second brick.
    """
    kept = []
    for i in np.argsort(-np.asarray(confidences)):
        mask = masks[i]
        if all((mask & other).sum() <= INSIDE * mask.sum() for other in kept):
            kept.append(mask)
    return kept


def bricks_on_sheet(photo: np.ndarray, model=None) -> list[BrickCrop]:
    """Photo of the capture sheet (BGR, as cv2.imread gives it) -> BrickCrops.

    Raises ValueError if the 4 markers are not found.
    """
    area = brick_area(photo, PX_PER_MM)
    if area is None:
        raise ValueError("not all 4 ArUco markers found: is this a photo of the sheet?")
    rgb = cv2.cvtColor(area, cv2.COLOR_BGR2RGB)
    h, w = mask_shape = area.shape[:2]
    crops = []
    for mask in brick_masks(area, model):
        if mask.shape != mask_shape or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        x0, y0 = max(xs.min() - MARGIN, 0), max(ys.min() - MARGIN, 0)
        x1, y1 = min(xs.max() + MARGIN + 1, w), min(ys.max() + MARGIN + 1, h)
        crops.append(
            BrickCrop(
                image=rgb[y0:y1, x0:x1].copy(),
                mask=mask[y0:y1, x0:x1].copy(),
                bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            )
        )
    return crops
