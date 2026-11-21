"""Shape classifier: a small CNN, trained from scratch (no pretrained weights).

Input:  grey image of one brick, 1 x 96 x 96 (legorec.shape.preprocess)
Output: one score per part -> softmax -> probability per part

    block 1:  32 channels, 96 x 96 -> 48 x 48
    block 2:  64 channels, 48 x 48 -> 24 x 24
    block 3: 128 channels, 24 x 24 -> 12 x 12
    block 4: 256 channels, 12 x 12 ->  6 x  6
    global average pooling -> 256 numbers ("which features are in the image")
    dropout -> linear layer -> one score per part

About 1.2 million parameters (ResNet-18 for comparison: about 11 million).

The image alone does not show how big a brick is (every brick is resized to
96 x 96). On a sheet photo its size in mm is known: the size prior
(legorec.shape.size_prior) adds that to the scores.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from legorec.shape.augment import normalize
from legorec.shape.preprocess import preprocess_crop, preprocess_images
from legorec.types import BrickCrop

# 2: grey + contrast stretch + YOLO masks, no size input. Checkpoints without
# "format" are older, see check_supported.
CHECKPOINT_FORMAT = 2


def _conv_block(in_channels, out_channels):
    """Two 3x3 convolutions (each followed by BatchNorm and ReLU), then halve
    the image size with max pooling."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """See the module docstring. `width` = channels of the first block."""

    def __init__(self, n_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()

        # Do not reorder these layers: checkpoints store the weights by
        # position (features.0, features.1, ...) and could no longer be loaded.
        self.features = nn.Sequential(
            _conv_block(1, width),
            _conv_block(width, 2 * width),
            _conv_block(2 * width, 4 * width),
            _conv_block(4 * width, 8 * width),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(8 * width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: images (B, 1, 96, 96) -> scores (B, n_classes)."""
        return self.fc(self.features(x))


def parts_known_by_all(checkpoints) -> set[str]:
    """Part numbers that every one of these checkpoints can answer.

    Models trained on other data know other parts. Comparing them is only fair
    on the parts they all know.
    """
    known = None
    for path in checkpoints:
        parts = set(load_checkpoint(path)["parts"])
        known = parts if known is None else known & parts
    return known or set()


def save_checkpoint(path, model, parts, names, config, split_hash) -> None:
    """Save a trained model with everything ShapeClassifier needs to use it.

    config: the training settings as a dict; ShapeClassifier reads size, width
    and dropout from it.
    """
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "parts": parts,
            "names": names,
            "cfg": config,
            "split_hash": split_hash,
        },
        path,
    )


def check_supported(saved: dict) -> None:
    """ValueError for a checkpoint of an option that is gone.

    Older checkpoints have no "format". They still work if they were trained
    the way every model is trained now; the others need the code of the Git
    tag v1-legacy (Kaggle renders, size feature, no contrast stretch, the
    brick_mask of the training images).
    """
    if saved.get("format") == CHECKPOINT_FORMAT:
        return
    config = saved["cfg"]
    if (
        config.get("size_feature")
        or config.get("contrast", "none") != "stretch"
        or config.get("mask", "brick") != "yolo"
    ):
        raise ValueError(
            "checkpoint of an older model (Kaggle, size feature, no contrast "
            "stretch or brick_mask): use the code of the Git tag v1-legacy"
        )


def usable_checkpoints(checkpoints) -> list[Path]:
    """The checkpoints of `checkpoints` that ShapeClassifier can load; prints
    which older ones are left out."""
    usable = []
    for path in map(Path, checkpoints):
        try:
            check_supported(load_checkpoint(path))
            usable.append(path)
        except ValueError:
            print(f"[skip] {path.name}: older model, needs the Git tag v1-legacy")
    return usable


def load_checkpoint(path, device: str = "cpu") -> dict:
    """Read a checkpoint written by save_checkpoint.

    weights_only=True: only tensors and plain Python values (dicts, lists,
    numbers, strings) are unpickled, never code. A downloaded checkpoint cannot
    run anything on our machines, and save_checkpoint stores nothing else.
    """
    return torch.load(path, map_location=device, weights_only=True)


class ShapeClassifier:
    """A trained model plus everything needed to use it on new images.

    Loads a checkpoint written by save_checkpoint and applies the same
    preprocessing that was used in training.

    size_prior: legorec.shape.size_prior.SizePrior. If given, predict() and
        crop_scores() also weigh how well the measured size fits each part.
        Only for crops from a sheet photo (known px per mm).
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        size_prior=None,
    ):
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

    @torch.no_grad()
    def _scores(self, arrays, batch_size=256) -> torch.Tensor:
        """Preprocessed arrays -> raw model scores (N x parts), on the CPU."""
        if not arrays:
            # e.g. the segmentation found no brick on a photo
            return torch.empty(0, len(self.parts))
        out = []
        for start in range(0, len(arrays), batch_size):
            batch = arrays[start : start + batch_size]
            images = normalize(torch.from_numpy(np.stack(batch)).unsqueeze(1))
            out.append(self.model(images.to(self.device)).cpu())
        return torch.cat(out)

    def _sheet_scores(self, crops) -> torch.Tensor:
        """Scores of crops from a sheet photo: + the size prior, if there is one."""
        scores = self._scores([preprocess_crop(c, self.size) for c in crops])
        if self.size_prior is not None:
            prior = self.size_prior.log_likelihood(crops, self.parts)
            scores = scores + torch.from_numpy(prior)
        return scores

    def _top_k(self, scores: torch.Tensor, k):
        """Scores -> for every image the k most likely parts."""
        probabilities = scores.softmax(dim=1)
        top_probs, top_indices = probabilities.topk(
            min(k, probabilities.shape[1]), dim=1
        )

        results = []
        for indices, probs in zip(top_indices.tolist(), top_probs.tolist()):
            results.append([(self.parts[i], float(p)) for i, p in zip(indices, probs)])
        return results

    def predict(self, crops: list[BrickCrop]) -> list[tuple[str, float]]:
        """Bricks cut out by the segmentation -> [(part number, probability)].

        Example: [("3001", 0.93), ("3022", 0.71)]
        """
        top_1 = self._top_k(self._sheet_scores(crops), k=1)
        return [best[0] for best in top_1]

    def predict_images(self, images, k: int = 3):
        """Photos with one brick each on a plain background.

        Returns (top-k [(part number, probability)] per image,
                 the preprocessed images the model saw).
        """
        arrays = self._from_images(images)
        return self._top_k(self._scores(arrays), k), arrays

    def scores(self, images) -> np.ndarray:
        """Photos with one brick each -> raw scores (N images x parts), the same
        preprocessing as in training. For evaluations that need every score."""
        return self._scores(self._from_images(images)).numpy()

    def crop_scores(self, crops: list[BrickCrop]) -> np.ndarray:
        """Like scores(), for BrickCrops of the segmentation (as predict(),
        with the size prior if there is one)."""
        return self._sheet_scores(crops).numpy()

    def _from_images(self, images):
        """Photos -> preprocessed arrays, as in training (YOLO mask)."""
        return preprocess_images(images, self.size)[0]
