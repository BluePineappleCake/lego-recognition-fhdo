"""Building blocks for training the shape classifier.

scripts/train_shape.py puts them together and adds the W&B logging. They live
here, without W&B, so they can be tested and reused later (e.g. for
fine-tuning on our own photos).

Training images come from the sources in config.train_on (see SOURCES):
internet photos and renders, our renders and, for parts without internet
data, our sheet photos. The classes are the parts that have
training images. Every source keeps its own split file, so a test image of
one source never gets into training.

The internet photos are always loaded, if the folder exists: their val split
is evaluated after every epoch, so every run reports how well it transfers to
real photos (also when they are not in train_on).
"""

import hashlib
import json
import random
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from legorec.data.internet import index_photos, load_or_create_photo_split
from legorec.data.renders import index_renders, load_or_create_render_split
from legorec.paths import internet_dir, own_photos_dir, own_renders_dir, processed_dir
from legorec.shape.augment import augment_batch
from legorec.shape.preprocess import (
    PREPROCESS_VERSION,
    preprocess_crop,
    preprocess_images,
)

# where training images can come from (config: train_on)
SOURCES = (
    "internet_photos",  # data/raw/internet/photos
    "internet_renders",  # data/raw/internet/renders
    "own_renders",  # data/raw/own_renders (scripts/render_ldraw.py)
    "own_photos",  # bricks on our sheet photos, only config.own_photo_parts
)
# settings of older runs (Kaggle renders, size feature, contrast, mask):
# their code is in the Git tag v1-legacy
REMOVED_SETTINGS = {"size_feature", "size_noise", "contrast", "mask", "classes_from"}
# real photos: few, so every one is used real_repeat times per epoch
PHOTO_SOURCES = {"internet_photos", "own_photos"}


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


@dataclass
class SourceData:
    """One data source, preprocessed, all of its parts."""

    # "train" / "val" / "test" -> (images (N, size, size) uint8, their parts)
    splits: dict[str, tuple[np.ndarray, list[str]]]
    names: dict[str, str]  # part number -> readable name, e.g. "3001 brick 2x4"
    split_hash: str  # fingerprint of its split file


@dataclass
class TrainData:
    """All sources, cut down to the classes of the model, ready for training."""

    # source -> "train" / "val" / "test" -> (images, labels)
    sources: dict[str, dict[str, tuple[np.ndarray, list[int]]]]
    parts: list[str]  # label i means parts[i]
    names: dict[str, str]
    split_hashes: dict[str, str]  # source -> fingerprint of its split

    @property
    def split_hash(self) -> str:
        """One fingerprint for the run, over the splits of all sources."""
        text = json.dumps(self.split_hashes, sort_keys=True)
        return hashlib.md5(text.encode()).hexdigest()[:8]


def set_seed(seed: int) -> None:
    """Same seed + same code -> same result (on the GPU not fully guaranteed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _preprocess_cached(items_per_split, prefix, split_hash, config) -> SourceData:
    """Items with .path / .part / .name per split -> SourceData (cached on disk).

    Every setting that changes the preprocessed images must be part of the
    cache file name, otherwise an old cache would be reused without warning.
    The cache holds all parts of the source; the model's classes are picked
    later (load_training_data), so one cache serves every class list.
    """
    from legorec.segment.bricks import brick_model_id  # another YOLO, other masks

    cache_tag = (
        f"{config.size}_{PREPROCESS_VERSION}_{split_hash}_yolo{brick_model_id()}"
    )
    splits, names = {}, {}
    for split in ("train", "val", "test"):
        items = items_per_split.get(split, [])
        cache_file = processed_dir() / f"{prefix}_{split}_{cache_tag}.npz"
        x = _build_cache(items, config.size, cache_file)
        splits[split] = (x, [item.part for item in items])
        names |= {item.part: item.name for item in items}
    return SourceData(splits, names, split_hash)


def _build_cache(items, size: int, cache_file, chunk: int = 512) -> np.ndarray:
    """Preprocess all images once and store them on disk; later calls load the file.

    Preprocessing 40,000 images takes several minutes, loading the cache a few
    seconds. Returns the images (N, size, size) uint8. The cache is only
    recognised by its file name (see _preprocess_cached).
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        return np.load(cache_file)["x"]

    arrays, fallbacks = [], 0
    for start in range(0, len(items), chunk):  # not all images in memory at once
        loaded = [Image.open(item.path) for item in items[start : start + chunk]]
        done, n = preprocess_images(loaded, size)
        arrays += done
        fallbacks += n
    print(f"[cache] {cache_file.name}: YOLO found no brick in {fallbacks} of "
          f"{len(items)} images, brick_mask used there")  # fmt: skip

    x = np.stack(arrays) if arrays else np.zeros((0, size, size), np.uint8)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, x=x)
    return x


def _own_photo_crops(config: "TrainConfig") -> SourceData:
    """Bricks on our sheet photos of config.own_photo_parts, cut by YOLO.

    The split of data/splits/sheets.csv is kept, so the test photos that
    scripts/eval_own.py uses never get into training. Not cached: these are
    a few photos, and the brick finder may change.
    """
    import cv2

    from legorec.data.sheets import index_sheet_photos, load_or_create_sheet_split
    from legorec.segment import bricks_on_sheet

    photos = index_sheet_photos(own_photos_dir())
    photos_per_split, split_hash = load_or_create_sheet_split(photos)
    wanted = set(config.own_photo_parts)
    splits, names = {}, {}
    for split in ("train", "val", "test"):
        arrays, parts = [], []
        for photo in photos_per_split.get(split, []):
            if photo.part not in wanted:
                continue
            names[photo.part] = photo.folder
            for crop in bricks_on_sheet(cv2.imread(photo.path)):
                arrays.append(preprocess_crop(crop, config.size))
                parts.append(photo.part)
        x = (
            np.stack(arrays)
            if arrays
            else np.zeros((0, config.size, config.size), np.uint8)
        )
        splits[split] = (x, parts)
    missing = wanted - set(splits["train"][1])
    if missing:
        raise ValueError(f"own_photo_parts without train photos: {sorted(missing)}")
    return SourceData(splits, names, split_hash)


def load_source(source: str, config: "TrainConfig", root=None) -> SourceData | None:
    """Load one source (see SOURCES). None if its folder does not exist.

    root: folder of the source instead of the default from legorec.paths
    (internet_*: data/raw/internet, with photos/ and renders/).
    """
    if source == "internet_photos":
        root = Path(root or internet_dir())
        if not (root / "photos").is_dir():
            return None
        per_split, split_hash = load_or_create_photo_split(index_photos(root))
        return _preprocess_cached(per_split, source, split_hash, config)
    if source in ("internet_renders", "own_renders"):
        if root is None:
            root = internet_dir() if source == "internet_renders" else own_renders_dir()
        folder = Path(root) / "renders" if source == "internet_renders" else Path(root)
        if not folder.is_dir() or not any(d.is_dir() for d in folder.iterdir()):
            return None
        renders = index_renders(folder, strict=source == "own_renders")
        per_split, split_hash = load_or_create_render_split(renders, source)
        return _preprocess_cached(per_split, source, split_hash, config)
    if source == "own_photos":
        return _own_photo_crops(config)
    raise ValueError(f"unknown source {source!r}, known: {SOURCES}")


def load_training_data(config: "TrainConfig", roots: dict | None = None) -> TrainData:
    """Load config.train_on (+ the internet photos for evaluation, if there).

    The classes are every part with training images in one of the train_on
    sources. Images of other parts are dropped, e.g. internet photos of a part
    the model does not know (when they are only loaded for evaluation).
    """
    roots = roots or {}
    loaded = {}
    for source in dict.fromkeys([*config.train_on, "internet_photos"]):
        data = load_source(source, config, roots.get(source))
        if data is None and source in config.train_on:
            raise FileNotFoundError(f"train_on has {source}, but its folder is missing")
        if data is None:
            print(f"[data] no {source}, skipping it")
            continue
        loaded[source] = data

    parts = sorted({p for s in config.train_on for p in loaded[s].splits["train"][1]})
    label_of_part = {part: label for label, part in enumerate(parts)}
    names = {}
    for source in reversed(list(loaded)):  # the first source wins
        names |= loaded[source].names

    sources = {}
    for source, data in loaded.items():
        sources[source] = {}
        for split, (x, part_of) in data.splits.items():
            keep = np.array([p in label_of_part for p in part_of], bool)
            labels = [label_of_part[p] for p, k in zip(part_of, keep) if k]
            sources[source][split] = (x[keep] if len(x) else x, labels)
            print(f"[data] {source} {split}: {len(labels)} images")
    hashes = {source: data.split_hash for source, data in loaded.items()}
    return TrainData(sources, parts, names, hashes)


def combine_train(data: TrainData, train_on, repeat: int):
    """Training arrays of all train_on sources; photos `repeat` times each.

    Without repeating, a few thousand photos next to tens of thousands of
    renders would hardly matter. Repeated copies are not identical for the
    network: every batch is augmented anew.
    """
    xs, ys = [], []
    for source in train_on:
        x, y = data.sources[source]["train"]
        times = repeat if source in PHOTO_SOURCES else 1
        xs += [x] * times
        ys += list(y) * times
    return np.concatenate(xs), ys


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
