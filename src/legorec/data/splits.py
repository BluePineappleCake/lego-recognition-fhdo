"""Train / val / test splits, the same for the whole team.

Every data source has a split file in data/splits/ (in Git) that maps each
group to "train", "val" or "test". A group is a set of images that must stay
together: two cameras of one capture, two colours of one render pose. If one
were in train and the other in test, the test would be far too easy.

    load_or_create_split(items, path)   the split of a list of items
    split_hash(...)                     short fingerprint, part of cache names
"""

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def make_split(items, seed: int = 42, frac=(0.8, 0.1, 0.1)) -> dict[str, str]:
    """Assign every group to "train", "val" or "test", e.g. {"3001_123": "train"}.

    Per part: shuffle its groups with a fixed seed, give the first 80 % to
    train, the next 10 % to val and the rest to test. So every part has the
    same share in each split, and the result is the same on every computer.
    """
    train_share, val_share, _ = frac
    random_generator = random.Random(seed)

    groups_per_part = defaultdict(set)
    for item in items:
        groups_per_part[item.part].add(item.group)

    split_of_group = {}
    for part in sorted(groups_per_part):
        # a set has no fixed order, so sort before shuffling
        groups = sorted(groups_per_part[part])
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


def save_split(assign: dict[str, str], path) -> None:
    """Write {group: split} as a CSV file with the columns group,split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["group", "split"])
        writer.writerows(sorted(assign.items()))


def load_split(path) -> dict[str, str]:
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


def split_by_hash(group: str, frac=(0.8, 0.1, 0.1)) -> str:
    """ "train", "val" or "test" for one group, from a hash of its name.

    For groups added to an existing split: the answer depends only on the
    name, not on which other groups exist, so adding data never moves a group
    that is already in the file.
    """
    position = int(hashlib.md5(group.encode()).hexdigest()[:8], 16) / 2**32
    train_share, val_share, _ = frac
    if position < train_share:
        return "train"
    return "val" if position < train_share + val_share else "test"


def load_or_create_split(
    items, path, frac=(0.8, 0.1, 0.1), extend=False
) -> tuple[dict[str, list], str]:
    """Load the split from the repo, or create it if it does not exist yet.

    The split file is in Git so the whole team uses the same split: results
    are only comparable on the same split.

    items: anything with .part and .group (renders, photos). `frac` is only
    used when the file is created or extended.

    extend=False: a group missing in the file is an error (fixed datasets).
    extend=True: missing groups are added (split_by_hash) and the file is
    written again; groups already in the file keep their split. For data we
    keep adding to (our renders, our photos).

    Returns ({"train": [...], "val": [...], "test": [...]}, fingerprint).
    """
    path = Path(path)
    if path.exists():
        split_of_group = load_split(path)
        print(f"[split] loaded {path}")
    else:
        split_of_group = make_split(items, frac=frac)
        save_split(split_of_group, path)
        print(f"[split] created {path} -> commit it, the whole team must use it")

    # every group must be in the file, otherwise images would silently be lost
    unknown_groups = {image.group for image in items} - split_of_group.keys()
    if unknown_groups and extend:
        split_of_group |= {g: split_by_hash(g, frac) for g in unknown_groups}
        save_split(split_of_group, path)
        print(f"[split] added {len(unknown_groups)} new groups to {path} -> commit it")
    elif unknown_groups:
        raise ValueError(
            f"{len(unknown_groups)} groups are not in {path}, "
            f"e.g. {sorted(unknown_groups)[:3]}"
        )

    items_per_split = defaultdict(list)
    for item in items:
        items_per_split[split_of_group[item.group]].append(item)
    return dict(items_per_split), split_hash(split_of_group)
