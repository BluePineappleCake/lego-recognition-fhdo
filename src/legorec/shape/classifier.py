"""Use a trained shape model on new bricks.

    from legorec.shape.classifier import load_classifier

    classifier = load_classifier()           # models/shape.pt + size prior
    classifier.predict(crops)                # [("3001", 0.93), ("3022", 0.71)]
    classifier.top_k(crops, k=3)             # the 3 most likely parts per crop
    classifier.scores(crops)                 # raw scores, N crops x parts

`crops` are BrickCrops: from a sheet photo (legorec.segment.bricks_on_sheet)
or from images with one brick each (legorec.segment.single). Every crop goes
through legorec.shape.preprocess.preprocess_crop, as in training.

The size prior only fits crops of a sheet photo (8 px/mm): the size of the
brick in mm weighs the answer (docs/experiments.md, 2026-11-21: ~56 % ->
~83 % on our test photos). For single images (unknown scale) use a classifier
without it: load_classifier(size_prior=None).

Confidences are softmax probabilities (0..1). They are not calibrated yet: on
real photos the model can be wrong with 70-80 %.
"""

from pathlib import Path

import numpy as np
import torch

from legorec.paths import shape_model_path
from legorec.shape.augment import normalize
from legorec.shape.model import SmallCNN, check_supported, load_checkpoint
from legorec.shape.preprocess import preprocess_crop
from legorec.shape.size_prior import TABLE, SizePrior
from legorec.types import BrickCrop


def load_classifier(
    checkpoint: str | Path | None = None,
    device: str | None = None,
    size_prior="auto",
) -> "ShapeClassifier":
    """The team's shape model, ready to use.

    checkpoint: a .pt file from scripts/train_shape.py, default models/shape.pt
        (or $LEGOREC_SHAPE_MODEL). Copy the checkpoint the team agreed on there.
    device: "cuda" or "cpu", default: the GPU if there is one
    size_prior: "auto" (default): the table data/reference/part_sizes.csv if it
        exists. None: off (for images that are not from a sheet photo).
    """
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
    return ShapeClassifier(checkpoint, device=device, size_prior=size_prior)


class ShapeClassifier:
    """A trained model plus everything needed to use it on new bricks.

    Loads a checkpoint written by legorec.shape.model.save_checkpoint and
    preprocesses every crop as in training.

    size_prior: legorec.shape.size_prior.SizePrior or None. If given, every
        answer also weighs how well the measured size fits each part. Only for
        crops of a sheet photo (known px per mm).
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu", size_prior=None):
        saved = load_checkpoint(checkpoint, device)
        check_supported(saved)
        config = saved["cfg"]

        self.parts = list(saved["parts"])  # part numbers in the order of the outputs
        self.names = saved.get("names", {})  # part number -> readable name
        self.size = config["size"]

        self.model = SmallCNN(len(self.parts), config["width"], config["dropout"])
        self.model.load_state_dict(saved["model"])
        self.model.to(device).eval()
        self.device = device
        self.size_prior = size_prior

    def predict(self, crops: list[BrickCrop]) -> list[tuple[str, float]]:
        """The most likely part of every crop, e.g. [("3001", 0.93), ("3022", 0.71)]."""
        return [best[0] for best in self.top_k(crops, k=1)]

    def top_k(
        self, crops: list[BrickCrop], k: int = 3
    ) -> list[list[tuple[str, float]]]:
        """The k most likely parts of every crop, with their probabilities."""
        probabilities = torch.from_numpy(self.scores(crops)).softmax(dim=1)
        top_probs, top_indices = probabilities.topk(min(k, len(self.parts)), dim=1)
        return [
            [(self.parts[i], float(p)) for i, p in zip(indices, probs)]
            for indices, probs in zip(top_indices.tolist(), top_probs.tolist())
        ]

    def scores(self, crops: list[BrickCrop]) -> np.ndarray:
        """Raw scores (N crops x parts), with the size prior if there is one.
        For evaluations that need every score; softmax gives probabilities."""
        scores = self._model_scores([preprocess_crop(c, self.size) for c in crops])
        if self.size_prior is not None and crops:
            scores = scores + self.size_prior.log_likelihood(crops, self.parts)
        return scores

    @torch.no_grad()
    def _model_scores(self, arrays, batch_size=256) -> np.ndarray:
        """Preprocessed arrays -> scores of the network (N x parts)."""
        if not arrays:  # e.g. no brick found on a photo
            return np.empty((0, len(self.parts)), np.float32)
        out = []
        for start in range(0, len(arrays), batch_size):
            batch = np.stack(arrays[start : start + batch_size])
            images = normalize(torch.from_numpy(batch).unsqueeze(1))
            out.append(self.model(images.to(self.device)).cpu())
        return torch.cat(out).numpy()
