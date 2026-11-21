"""Shape step of the pipeline: which part is this brick?

This is the interface the other steps use (issue #7):

    from legorec.shape.predict import predict

    part, confidence = predict(crop)  # e.g. ("3001", 0.93)

`crop` is a BrickCrop from the segmentation step. `confidence` is the softmax
probability of the best part (0..1). It is not calibrated yet: on real photos
the model can be wrong with 70-80 %.

The size prior is on by default: the measured size of the brick in mm (long
and short side, data/reference/part_sizes.csv) weighs the answer, as in
scripts/find_bricks.py (docs/experiments.md, 2026-11-21: ~56 % -> ~84 % on
our sheet photos). It assumes crops of a sheet photo (bricks_on_sheet, 8
px/mm). For crops with another scale: load_model(size_prior=None).

Which model is used?
    By default models/shape.pt. Copy the checkpoint the team agreed on there,
    or set the environment variable LEGOREC_SHAPE_MODEL to another file.
    The model is loaded once, on the first call to predict().

Another model: load_model("models/shape_xyz.pt") once, then predict() as usual.

For many crops at once (faster on the GPU) use the classifier directly:
    load_model().predict(crops) -> [(part, confidence), ...]
"""

from pathlib import Path

import torch

from legorec.paths import shape_model_path
from legorec.shape.model import ShapeClassifier
from legorec.shape.size_prior import TABLE, SizePrior
from legorec.types import BrickCrop

# The loaded model. None until load_model() or the first predict() call.
_classifier = None


def load_model(
    checkpoint: str | Path | None = None,
    device: str | None = None,
    size_prior="auto",
) -> ShapeClassifier:
    """Load a checkpoint and use it for all following predict() calls.

    checkpoint: path to a .pt file from scripts/train_shape.py,
        default: models/shape.pt (or $LEGOREC_SHAPE_MODEL)
    device: "cuda" or "cpu", default: the GPU if there is one
    size_prior: legorec.shape.size_prior.SizePrior, only for crops of a sheet
        photo. "auto" (default): the table in data/reference/ if it exists.
        None: off.
    """
    global _classifier  # predict() uses the model loaded here

    checkpoint = Path(checkpoint or shape_model_path())
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"no shape model at {checkpoint}. Download one from W&B "
            "(artifact 'shape-cnn') or copy one from models/ to models/shape.pt, "
            "or set LEGOREC_SHAPE_MODEL."
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if size_prior == "auto":
        size_prior = SizePrior.load() if TABLE.exists() else None

    _classifier = ShapeClassifier(checkpoint, device=device, size_prior=size_prior)
    return _classifier


def predict(crop: BrickCrop) -> tuple[str, float]:
    """One brick crop -> (part number, confidence), e.g. ("3001", 0.93)."""
    if _classifier is None:
        load_model()
    return _classifier.predict([crop])[0]
