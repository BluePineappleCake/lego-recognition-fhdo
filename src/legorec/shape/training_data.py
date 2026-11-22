"""Where the training images come from, preprocessed and cached.

    data = load_training_data(config)   # every source of config.train_on
    x, y = combine_train(data, config.train_on, config.real_repeat)

Training images come from the sources in config.train_on (see SOURCES):
internet photos and renders, our renders and, for parts without internet
data, our sheet photos. The classes are the parts that have training images.
Every source keeps its own split file, so a test image of one source never
gets into training.

Every image is cut out like a brick on a sheet photo (legorec.segment.single:
YOLO, brick_mask as fallback) and preprocessed by preprocess_crop. That takes
minutes, so the result is cached in $LEGOREC_PROCESSED.

The internet photos are always loaded, if the folder exists: their val split
is evaluated after every epoch, so every run reports how well it transfers to
real photos (also when they are not in train_on).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from legorec.data.internet import index_photos, load_or_create_photo_split
from legorec.data.renders import index_renders, load_or_create_render_split
from legorec.paths import internet_dir, own_photos_dir, own_renders_dir, processed_dir
from legorec.shape.preprocess import PREPROCESS_VERSION, preprocess_crop

if TYPE_CHECKING:
    from legorec.shape.training import TrainConfig

# where training images can come from (config: train_on)
SOURCES = (
    "internet_photos",  # data/raw/internet/photos
    "internet_renders",  # data/raw/internet/renders
    "own_renders",  # data/raw/own_renders (scripts/render_ldraw.py)
    "own_photos",  # bricks on our sheet photos, only config.own_photo_parts
)
# real photos: few, so every one is used real_repeat times per epoch
PHOTO_SOURCES = {"internet_photos", "own_photos"}


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


def _preprocess_cached(items_per_split, prefix, split_hash, config) -> SourceData:
    """Items with .path / .part / .name per split -> SourceData (cached on disk).

    Every setting that changes the preprocessed images must be part of the
    cache file name, otherwise an old cache would be reused without warning.
    So are the image files themselves (_files_id).
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
        cache_file = (
            processed_dir() / f"{prefix}_{split}_{_files_id(items)}_{cache_tag}.npz"
        )
        x = _build_cache(items, config.size, cache_file)
        splits[split] = (x, [item.part for item in items])
        names |= {item.part: item.name for item in items}
    return SourceData(splits, names, split_hash)


def _files_id(items) -> str:
    """Fingerprint of the image files of one split: a deleted, added or
    re-rendered image gives a new cache name. File name and time of change
    only, so moving the data folder does not rebuild the cache."""
    text = "\n".join(
        f"{item.part}/{Path(item.path).name} {Path(item.path).stat().st_mtime_ns}"
        for item in items
    )
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _build_cache(items, size: int, cache_file, chunk: int = 512) -> np.ndarray:
    """Preprocess all images once and store them on disk; later calls load the file.

    Preprocessing 40,000 images takes several minutes, loading the cache a few
    seconds. Returns the images (N, size, size) uint8. The cache is only
    recognised by its file name (see _preprocess_cached).
    """
    from legorec.segment.single import single_brick_crops  # needs ultralytics

    cache_file = Path(cache_file)
    if cache_file.exists():
        return np.load(cache_file)["x"]

    arrays, fallbacks = [], 0
    for start in range(0, len(items), chunk):  # not all images in memory at once
        loaded = [Image.open(item.path) for item in items[start : start + chunk]]
        crops, n = single_brick_crops(loaded)
        arrays += [preprocess_crop(crop, size) for crop in crops]
        fallbacks += n
    print(f"[cache] {cache_file.name}: YOLO found no brick in {fallbacks} of "
          f"{len(items)} images, brick_mask used there")  # fmt: skip

    x = np.stack(arrays) if arrays else np.zeros((0, size, size), np.uint8)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, x=x)
    return x


def _own_photo_crops(config: TrainConfig) -> SourceData:
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


def load_source(source: str, config: TrainConfig, root=None) -> SourceData | None:
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


def load_training_data(config: TrainConfig, roots: dict | None = None) -> TrainData:
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
