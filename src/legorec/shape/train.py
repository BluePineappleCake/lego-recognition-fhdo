"""Building blocks for training the shape classifier.

scripts/train_shape.py puts them together and adds the W&B logging. They live
here, without W&B, so they can be tested and reused later (e.g. for
fine-tuning on our own photos).

Real photos (legorec.data.internet) are used in two ways:
    - always, if the folder exists: internet/val is evaluated after every
      epoch, so every run reports how well it transfers to real photos
    - with real_train: internet, the internet/train photos are also added to
      the training set (each one real_repeat times per epoch)
"""

import random
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from legorec.data.internet import index_photos, load_or_create_photo_split, only_parts
from legorec.data.renders import build_cache, index_renders, load_or_create_split
from legorec.paths import internet_dir, processed_dir, renders_dir
from legorec.shape.augment import augment_batch
from legorec.shape.model import SizeNorm
from legorec.shape.preprocess import PREPROCESS_VERSION


@dataclass
class TrainConfig:
    """All settings of a training run. A file in configs/ only lists what it changes."""

    size: int = 96  # side length of the brick images in pixels
    width: int = 32  # channels of the first conv block (see model.py)
    dropout: float = 0.3
    size_feature: bool = False  # also give the network the measured brick size
    contrast: str = "none"  # "stretch": stretch the contrast per brick (preprocess.py)
    size_noise: float = 0.05  # relative noise on the brick size during training
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-3  # peak learning rate of the OneCycle schedule
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1  # target ~0.9 instead of 1 for the right class
    hflip: bool = False  # mirror images; only if no part has a mirror twin
    real_train: str = "none"  # "internet": also train on the internet/train photos
    real_repeat: int = 8  # how often every real photo is used per epoch
    seed: int = 0
    project: str = "lego-recognition"  # W&B project
    tags: list[str] = field(default_factory=lambda: ["renders-only"])

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
        if unknown:
            raise ValueError(f"unknown settings in {path}: {sorted(unknown)}")
        return cls(**values)


@dataclass
class RenderData:
    """The preprocessed renders, ready for training."""

    # "train" / "val" / "test" -> (images (N, size, size), size feature z (N,), labels)
    splits: dict[str, tuple[np.ndarray, np.ndarray, list[int]]]
    parts: list[str]  # label i means parts[i]
    names: dict[str, str]  # part number -> readable name, e.g. "3001 brick 2x4"
    size_norm: SizeNorm  # turns brick sizes in pixels into z
    split_hash: str  # fingerprint of the split (see legorec.data.renders)


def set_seed(seed: int) -> None:
    """Same seed + same code -> same result (on the GPU not fully guaranteed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_render_data(
    config: TrainConfig, data_dir: str | Path | None = None
) -> RenderData:
    """Renders -> split by pose -> preprocessed arrays (cached on disk)."""
    images = index_renders(data_dir or renders_dir())
    parts = sorted({image.part for image in images})
    label_of_part = {part: label for label, part in enumerate(parts)}
    names = {image.part: image.name for image in images}
    images_per_split, split_hash = load_or_create_split(images)

    # Every setting that changes the preprocessed images must be part of the
    # cache file name, otherwise an old cache would be reused without warning.
    cache_tag = f"{config.size}_{PREPROCESS_VERSION}_{config.contrast}_{split_hash}"

    raw = {}
    for split in ("train", "val", "test"):
        cache_file = processed_dir() / f"renders_{split}_{cache_tag}.npz"
        x, sizes_px = build_cache(
            images_per_split[split], config.size, cache_file, config.contrast
        )
        labels = [label_of_part[image.part] for image in images_per_split[split]]
        raw[split] = (x, sizes_px, labels)
        print(f"[data] {split}: {len(labels)} images")

    # Standardise the size with the training set only: val or test statistics
    # would leak information about them into the model.
    size_norm = SizeNorm.fit(raw["train"][1])
    splits = {
        split: (x, size_norm(sizes_px), labels)
        for split, (x, sizes_px, labels) in raw.items()
    }
    return RenderData(splits, parts, names, size_norm, split_hash)


@dataclass
class RealData:
    """Internet photos of the parts the model knows, preprocessed like the renders."""

    # "train" / "val" / "test" -> (images, z, labels), same layout as RenderData
    splits: dict[str, tuple[np.ndarray, np.ndarray, list[int]]]
    split_hash: str  # fingerprint of data/splits/internet.csv


def load_internet_data(
    config: TrainConfig, parts: list[str], data_dir: str | Path | None = None
) -> RealData | None:
    """Internet photos -> split by capture -> preprocessed arrays (cached on disk).

    Only photos of `parts` are kept, labelled with the same class index as the
    renders. Returns None if the folder does not exist (e.g. on Kaggle without
    the dataset), so training on renders still works without it.

    The size feature z is always 0 ("average size"): the photos are tightly
    cropped, so the measured size in pixels says nothing about the real size.
    """
    root = Path(data_dir or internet_dir())
    if not (root / "photos").is_dir():
        print(f"[data] no internet photos in {root}, skipping them")
        return None

    label_of_part = {part: label for label, part in enumerate(parts)}
    photos_per_split, split_hash = load_or_create_photo_split(index_photos(root))
    cache_tag = f"{config.size}_{PREPROCESS_VERSION}_{config.contrast}_{split_hash}"

    splits = {}
    for split in ("train", "val", "test"):
        photos = only_parts(photos_per_split[split], set(parts))
        cache_file = processed_dir() / f"internet_{split}_{cache_tag}.npz"
        x, _ = build_cache(photos, config.size, cache_file, config.contrast)
        z = np.zeros(len(photos), dtype=np.float32)
        splits[split] = (x, z, [label_of_part[photo.part] for photo in photos])
        print(f"[data] internet {split}: {len(photos)} photos")
    return RealData(splits, split_hash)


def add_real_photos(render_train, real_train, repeat: int):
    """Training arrays of the renders + every real photo `repeat` times.

    Without repeating, ~1,900 photos next to 32,000 renders would hardly
    matter. With repeat 8 they are about a third of every epoch. Repeated
    copies are not identical for the network: every batch is augmented anew.
    """
    x_r, z_r, y_r = render_train
    x_p, z_p, y_p = real_train
    x = np.concatenate([x_r] + [x_p] * repeat)
    z = np.concatenate([z_r] + [z_p] * repeat)
    y = list(y_r) + list(y_p) * repeat
    return x, z.astype(np.float32), y


def train_epoch(
    model,
    batches,
    optimizer,
    scheduler,
    loss_fn,
    config: TrainConfig,
    size_norm: SizeNorm,
    augment: bool = True,
) -> tuple[float, float]:
    """One pass over the training batches. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, n_correct, n_images = 0.0, 0, 0

    for x, z, labels in batches:
        if augment:
            x = augment_batch(x, config.hflip)
            # The size will never be measured exactly on photos, so the model
            # should not rely on it too much. size_noise is relative (log
            # scale); dividing by sd converts it to the scale of z.
            z = z + torch.randn_like(z) * config.size_noise / size_norm.sd

        scores = model(x, z)
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
    for x, z, labels in batches:
        all_scores.append(model(x, z).cpu())
        all_labels.append(labels.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    # plain cross-entropy: no label smoothing, no augmentation, no dropout,
    # so this loss is lower than the training loss and not directly comparable
    loss = F.cross_entropy(scores, labels).item()
    return scores.numpy(), labels.numpy(), loss


def overfit_one_batch(
    model, batches, optimizer, scheduler, loss_fn, config: TrainConfig, size_norm
) -> float:
    """Sanity check: a working model and training loop can memorise one batch.

    `batches` is a list with the same batch many times. Returns the loss on
    that batch after training. If it is not close to 0, something is broken
    (labels, model, optimizer) and a full run would be wasted time.
    """
    train_epoch(
        model, batches, optimizer, scheduler, loss_fn, config, size_norm, augment=False
    )
    x, z, labels = batches[0]
    with torch.no_grad():
        return loss_fn(model(x, z), labels).item()
