"""Kaggle render dataset: index the files, split by pose, cache the preprocessed images.

File names look like "3001 brick 2x4 123L.png": part number, name, pose index,
camera (L/R). L and R are two views of the SAME pose, so they share a group and
always land in the same split. Splitting by single image would leak near-copies
into the test set.
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

NAME_RE = re.compile(r"^(?P<cls>.+?)\s+(?P<idx>\d+)(?P<view>[LR]?)$")
SPLIT_FILE = SPLITS_DIR / "renders.csv"


@dataclass(frozen=True)
class RenderImage:
    path: str
    part: str  # part number, used as label, e.g. "3001"
    name: str  # e.g. "3001 brick 2x4"
    group: str  # pose id, e.g. "3001_123" (shared by L and R)


def index_renders(root) -> list[RenderImage]:
    files = sorted(Path(root).rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"no .png under {root}")
    items, bad = [], []
    for p in files:
        m = NAME_RE.match(p.stem)
        if not m:
            bad.append(p.name)
            continue
        name = m["cls"].strip()
        part = name.split()[0]
        items.append(RenderImage(str(p), part, name, f"{part}_{m['idx']}"))
    if bad:
        print(f"[warn] {len(bad)} files do not match the name pattern, e.g. {bad[:3]}")
    return items


def make_split(items, seed=42, frac=(0.8, 0.1, 0.1)) -> dict[str, str]:
    """group -> 'train' | 'val' | 'test', stratified per part."""
    rng = random.Random(seed)
    groups_by_part = defaultdict(set)
    for it in items:
        groups_by_part[it.part].add(it.group)
    assign = {}
    for part in sorted(groups_by_part):
        gs = sorted(groups_by_part[part])
        rng.shuffle(gs)
        n_tr, n_va = int(frac[0] * len(gs)), int(frac[1] * len(gs))
        for i, g in enumerate(gs):
            assign[g] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return assign


def save_split(assign: dict[str, str], path=SPLIT_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["group", "split"])
        w.writerows(sorted(assign.items()))


def load_split(path=SPLIT_FILE) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["group"]: row["split"] for row in csv.DictReader(f)}


def split_hash(assign: dict[str, str]) -> str:
    return hashlib.md5(json.dumps(assign, sort_keys=True).encode()).hexdigest()[:8]


def load_or_create_split(items, path=SPLIT_FILE):
    """The split file is versioned in Git. Only create it if it does not exist yet."""
    path = Path(path)
    if path.exists():
        assign = load_split(path)
        print(f"[split] loaded {path}")
    else:
        assign = make_split(items)
        save_split(assign, path)
        print(
            f"[split] created {path} -> commit it, the whole team must use the same one"
        )
    missing = {it.group for it in items} - assign.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} groups are not in {path}, e.g. {sorted(missing)[:3]}"
        )
    out = defaultdict(list)
    for it in items:
        out[assign[it.group]].append(it)
    return dict(out), split_hash(assign)


def build_cache(items, size, cache_file):
    """Preprocess once and keep images (uint8) + brick sizes (px) on disk."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        c = np.load(cache_file)
        return c["x"], c["px"]
    res = [preprocess_image(Image.open(it.path), size) for it in items]
    x = np.stack([r[0] for r in res])
    px = np.array([r[1] for r in res], np.float32)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, x=x, px=px)
    return x, px
