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

_models = {}  # path -> loaded YOLO model


def load_yolo(model=None):
    """The YOLO brick model (default: models/bricks_yolo.pt), loaded once."""
    path = Path(model or brick_model_path())
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


def brick_masks(area: np.ndarray, model=None, conf: float = 0.25) -> list[np.ndarray]:
    """Warped brick area (BGR) -> one boolean mask (area size) per brick."""
    result = load_yolo(model).predict(
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
