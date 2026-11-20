"""Sheet photo in, one BrickCrop per brick out: the segmentation step of the
pipeline.

    from legorec.segment import bricks_on_sheet

    crops = bricks_on_sheet(cv2.imread("IMG_3482.jpg"))            # YOLO-seg
    crops = bricks_on_sheet(photo, method="threshold")             # find_bricks

Both methods work on the brick area, warped flat with the 4 ArUco markers
(brick_area). The bbox of a crop is in pixels of that area (PX_PER_MM per
mm), not of the raw photo.

"yolo" needs the optional dependency (pip install -e .[detect]) and a
trained model at models/bricks_yolo.pt (or $LEGOREC_BRICK_MODEL): copy
runs/segment/train/weights/best.pt there. It was trained on the outlines of
"threshold" plus some corrected by hand; whether it separates touching
bricks better is not measured yet (docs/experiments.md, 2026-11-20).
"""

from pathlib import Path

import cv2
import numpy as np

from legorec.paths import brick_model_path
from legorec.segment.sheet import PX_PER_MM, brick_area, find_bricks, is_frame
from legorec.types import BrickCrop

IMGSZ = 1024  # as in training
MARGIN = 25  # pixels around each brick, as in cut_bricks
INSIDE = 0.5  # a mask more than half inside a surer one is dropped

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


def brick_masks(area: np.ndarray, method: str = "yolo", model=None, conf: float = 0.25):
    """Warped brick area (BGR) -> one boolean mask (area size) per brick."""
    if method == "threshold":
        return [m for m in find_bricks(area) if not is_frame(m)]
    if method != "yolo":
        raise ValueError(f"method must be 'yolo' or 'threshold', not {method!r}")
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


def bricks_on_sheet(
    photo: np.ndarray, method: str = "yolo", model=None
) -> list[BrickCrop]:
    """Photo of the capture sheet (BGR, as cv2.imread gives it) -> BrickCrops.

    Raises ValueError if the 4 markers are not found.
    """
    area = brick_area(photo, PX_PER_MM)
    if area is None:
        raise ValueError("not all 4 ArUco markers found: is this a photo of the sheet?")
    rgb = cv2.cvtColor(area, cv2.COLOR_BGR2RGB)
    h, w = mask_shape = area.shape[:2]
    crops = []
    for mask in brick_masks(area, method, model):
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
