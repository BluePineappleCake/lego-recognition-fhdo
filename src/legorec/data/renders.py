"""Kaggle render dataset: find the images, split them, cache the preprocessed images.

File names look like "3001 brick 2x4 123L.png":

    3001 brick 2x4   name of the part, starting with the part number
    123              pose number (how the brick is turned)
    L / R            camera (left / right)

Every pose was rendered by two cameras, and the two images are almost
identical. If one were in training and the other in the test set, the test
would be far too easy. So we split by pose ("group" below), not by image.
"""

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from legorec.paths import SPLITS_DIR
from legorec.shape.preprocess import preprocess_image

# "3001 brick 2x4 123L" -> name "3001 brick 2x4", pose "123", camera "L"
FILE_NAME_PATTERN = re.compile(r"^(?P<name>.+?)\s+(?P<pose>\d+)(?P<camera>[LR]?)$")

SPLIT_FILE = SPLITS_DIR / "renders.csv"


@dataclass(frozen=True)
class RenderImage:
    path: str  # full path of the PNG file
    part: str  # part number = the label, e.g. "3001"
    name: str  # part number + name, e.g. "3001 brick 2x4"
    group: str  # part + pose, e.g. "3001_123"; the same for camera L and R


def index_renders(root: str | Path) -> list[RenderImage]:
    """Find all render PNGs below `root` and read part and pose from the names."""
    files = sorted(Path(root).rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"no .png under {root}")

    images = []
    unreadable_names = []
    for file in files:
        match = FILE_NAME_PATTERN.match(file.stem)
        if not match:
            unreadable_names.append(file.name)
            continue

        name = match["name"].strip()  # "3001 brick 2x4"
        part = name.split()[0]  # "3001"
        group = f"{part}_{match['pose']}"  # "3001_123"
        images.append(RenderImage(str(file), part, name, group))

    if unreadable_names:
        print(
            f"[warn] {len(unreadable_names)} files do not match the name pattern, "
            f"e.g. {unreadable_names[:3]}"
        )
    return images


def make_split(
    items: list[RenderImage], seed: int = 42, frac=(0.8, 0.1, 0.1)
) -> dict[str, str]:
    """Assign every pose to "train", "val" or "test", e.g. {"3001_123": "train"}.

    Per part: shuffle its poses with a fixed seed, give the first 80 % to
    train, the next 10 % to val and the rest to test. So every part has the
    same share in each split, and the result is the same on every computer.
    """
    train_share, val_share, _ = frac
    random_generator = random.Random(seed)

    poses_per_part = defaultdict(set)
    for image in items:
        poses_per_part[image.part].add(image.group)

    split_of_group = {}
    for part in sorted(poses_per_part):
        # a set has no fixed order, so sort before shuffling
        groups = sorted(poses_per_part[part])
        random_generator.shuffle(groups)

        n_train = int(train_share * len(groups))
        n_val = int(val_share * len(groups))
        for position, group in enumerate(groups):
            if position < n_train:
                split_of_group[group] = "train"
            elif position < n_train + n_val:
                split_of_group[group] = "val"
            else:
                split_of_group[group] = "test"
    return split_of_group


def save_split(assign: dict[str, str], path=SPLIT_FILE) -> None:
    """Write {group: split} as a CSV file with the columns group,split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["group", "split"])
        writer.writerows(sorted(assign.items()))


def load_split(path=SPLIT_FILE) -> dict[str, str]:
    """Read the CSV written by save_split back into {group: split}."""
    with open(path, newline="", encoding="utf-8") as file:
        return {row["group"]: row["split"] for row in csv.DictReader(file)}


def split_hash(assign: dict[str, str]) -> str:
    """Short fingerprint of a split, e.g. "28030e6c".

    It is part of the cache file names and logged to W&B, so we can always
    tell which split a cache file or a training run used.
    """
    as_text = json.dumps(assign, sort_keys=True)
    return hashlib.md5(as_text.encode()).hexdigest()[:8]


def load_or_create_split(
    items: list[RenderImage], path=SPLIT_FILE
) -> tuple[dict[str, list[RenderImage]], str]:
    """Load the split from the repo, or create it if it does not exist yet.

    The split file is in Git so the whole team uses the same split: results
    are only comparable on the same split.

    Returns ({"train": [...], "val": [...], "test": [...]}, fingerprint).
    """
    path = Path(path)
    if path.exists():
        split_of_group = load_split(path)
        print(f"[split] loaded {path}")
    else:
        split_of_group = make_split(items)
        save_split(split_of_group, path)
        print(f"[split] created {path} -> commit it, the whole team must use it")

    # every pose must be in the file, otherwise images would silently be lost
    unknown_groups = {image.group for image in items} - split_of_group.keys()
    if unknown_groups:
        raise ValueError(
            f"{len(unknown_groups)} groups are not in {path}, "
            f"e.g. {sorted(unknown_groups)[:3]}"
        )

    images_per_split = defaultdict(list)
    for image in items:
        images_per_split[split_of_group[image.group]].append(image)
    return dict(images_per_split), split_hash(split_of_group)


def build_cache(
    items: list[RenderImage], size: int, cache_file, contrast: str = "none"
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess all images once and store them on disk; later calls load the file.

    Preprocessing 40,000 images takes several minutes, loading the cache a few
    seconds. Returns x: images (N, size, size) uint8 and px: the measured
    brick sizes (N,) in pixels.

    The cache is only recognised by its file name, so the name must change
    whenever the preprocessing changes (see load_render_data in
    legorec.shape.train), otherwise an old cache is reused.
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        cached = np.load(cache_file)
        return cached["x"], cached["px"]

    images, sizes = [], []
    for image in items:
        array, brick_size = preprocess_image(
            Image.open(image.path), size, contrast=contrast
        )
        images.append(array)
        sizes.append(brick_size)

    x = np.stack(images)
    px = np.array(sizes, dtype=np.float32)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, x=x, px=px)
    return x, px
