"""Training the shape classifier: the settings and one epoch.

scripts/train_shape.py puts the pieces together and adds the W&B logging.
They live here, without W&B, so they can be tested and reused later (e.g.
for fine-tuning on our own photos). Where the images come from:
legorec.shape.training_data.
"""

import random
from dataclasses import dataclass, field, fields

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from legorec.shape.augment import augment_batch
from legorec.shape.training_data import SOURCES

# settings of older runs (Kaggle renders, size feature, contrast, mask):
# their code is in the Git tag v1-legacy
REMOVED_SETTINGS = {"size_feature", "size_noise", "contrast", "mask", "classes_from"}


@dataclass
class TrainConfig:
    """All settings of a training run. A file in configs/ only lists what it changes."""

    size: int = 96  # side length of the brick images in pixels
    width: int = 32  # channels of the first conv block (see model.py)
    dropout: float = 0.3
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-3  # peak learning rate of the OneCycle schedule
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1  # target ~0.9 instead of 1 for the right class
    hflip: bool = False  # mirror images; only if no part has a mirror twin
    # sources of the training images, see SOURCES; the classes are every part
    # with training images
    train_on: list[str] = field(
        default_factory=lambda: ["internet_photos", "internet_renders", "own_renders"]
    )
    own_photo_parts: list[str] = field(default_factory=list)  # for own_photos
    # val split that picks the best epoch (default: the first of train_on)
    select_on: str | None = None
    real_repeat: int = 8  # how often every real photo is used per epoch
    seed: int = 0
    project: str = "lego-recognition"  # W&B project
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path, **overrides) -> "TrainConfig":
        """Read a config file; overrides that are None (option not given) are ignored.

        Unknown keys are an error. Otherwise a typo like "epoch: 5" would be
        ignored without a warning and the run would quietly use the default.
        """
        with open(path, encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}
        values |= {key: value for key, value in overrides.items() if value is not None}

        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown & REMOVED_SETTINGS:
            raise ValueError(
                f"{path} uses removed settings {sorted(unknown & REMOVED_SETTINGS)}: "
                "use the code of the Git tag v1-legacy for it"
            )
        if unknown:
            raise ValueError(f"unknown settings in {path}: {sorted(unknown)}")
        config = cls(**values)
        config.check()
        return config

    def __post_init__(self):
        # YAML reads 4073 as a number, part numbers are text
        self.own_photo_parts = [str(part) for part in self.own_photo_parts]
        if self.select_on is None:
            self.select_on = self.train_on[0] if self.train_on else None

    def check(self) -> None:
        """Settings that cannot work together are an error before any data loads."""
        unknown = set(self.train_on) - set(SOURCES)
        if unknown or not self.train_on:
            raise ValueError(f"train_on: {self.train_on}; sources are {SOURCES}")
        if ("own_photos" in self.train_on) != bool(self.own_photo_parts):
            raise ValueError("own_photos in train_on needs own_photo_parts, and back")
        if self.select_on not in [*self.train_on, "internet_photos"]:
            raise ValueError(f"select_on {self.select_on} is not loaded")


def set_seed(seed: int) -> None:
    """Same seed + same code -> same result (on the GPU not fully guaranteed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_epoch(
    model,
    batches,
    optimizer,
    scheduler,
    loss_fn,
    config: TrainConfig,
    augment: bool = True,
) -> tuple[float, float]:
    """One pass over the training batches. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, n_correct, n_images = 0.0, 0, 0

    for x, labels in batches:
        if augment:
            x = augment_batch(x, config.hflip)

        scores = model(x)
        loss = loss_fn(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # OneCycle changes the learning rate after every batch

        total_loss += loss.item() * len(labels)
        n_correct += (scores.argmax(dim=1) == labels).sum().item()
        n_images += len(labels)

    return total_loss / n_images, n_correct / n_images


@torch.no_grad()
def evaluate(model, batches) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the model without training. Returns (scores (N, C), labels (N,), loss)."""
    model.eval()
    all_scores, all_labels = [], []
    for x, labels in batches:
        all_scores.append(model(x).cpu())
        all_labels.append(labels.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    # plain cross-entropy: no label smoothing, no augmentation, no dropout,
    # so this loss is lower than the training loss and not directly comparable
    loss = F.cross_entropy(scores, labels).item()
    return scores.numpy(), labels.numpy(), loss


def overfit_one_batch(
    model, batches, optimizer, scheduler, loss_fn, config: TrainConfig
) -> float:
    """Sanity check: a working model and training loop can memorise one batch.

    `batches` is a list with the same batch many times. Returns the loss on
    that batch after training. If it is not close to 0, something is broken
    (labels, model, optimizer) and a full run would be wasted time.
    """
    train_epoch(model, batches, optimizer, scheduler, loss_fn, config, augment=False)
    x, labels = batches[0]
    with torch.no_grad():
        return loss_fn(model(x), labels).item()
