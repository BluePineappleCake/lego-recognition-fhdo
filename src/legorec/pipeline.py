"""The whole pipeline on one photo of the capture sheet. Start reading here.

    import cv2
    from legorec.pipeline import find_and_name

    for brick in find_and_name(cv2.imread("IMG_8844.jpg")):
        print(brick.part, f"{brick.confidence:.0%}", brick.crop.bbox)

The steps, each in its own module:

    1. warp the sheet flat with its 4 ArUco markers   legorec.segment.sheet
    2. find every brick (YOLO, one class "brick")     legorec.segment.bricks
    3. name its part: a CNN on the grey crop, plus    legorec.shape.classifier
       the size of the brick in mm (size prior)
    4. colour, inventory, "what can I build?"         planned: add them here

scripts/find_bricks.py runs this on photos and draws the result.
"""

from dataclasses import dataclass

import numpy as np

from legorec.segment import bricks_on_sheet
from legorec.types import BrickCrop


@dataclass
class FoundBrick:
    """One brick on the sheet and what the pipeline says about it."""

    crop: BrickCrop  # image, mask, bbox in pixels of the warped sheet (8 px/mm)
    part: str  # part number, e.g. "3001"
    confidence: float  # softmax probability of `part` (not calibrated yet)


_default_classifier = None  # loaded on the first call without a classifier


def find_and_name(
    photo: np.ndarray, classifier=None, brick_model=None
) -> list[FoundBrick]:
    """Sheet photo (BGR, as cv2.imread gives it) -> one FoundBrick per brick.

    classifier: a legorec.shape.classifier.ShapeClassifier; default the team's
        model (models/shape.pt) with the size prior, loaded once.
    brick_model: the YOLO model file; default models/bricks_yolo.pt.

    Raises ValueError if the 4 markers are not found.
    """
    if classifier is None:
        classifier = _team_classifier()
    crops = bricks_on_sheet(photo, brick_model)  # steps 1 and 2
    parts = classifier.predict(crops)  # step 3
    return [FoundBrick(crop, part, conf) for crop, (part, conf) in zip(crops, parts)]


def _team_classifier():
    global _default_classifier
    if _default_classifier is None:
        from legorec.shape.classifier import load_classifier  # needs torch

        _default_classifier = load_classifier()
    return _default_classifier
