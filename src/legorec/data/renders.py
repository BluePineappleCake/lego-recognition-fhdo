"""Renders of single bricks, cut to the brick, one folder per part:

    data/raw/internet/renders/      ~44,000 renders of 17 parts (internet set)
    data/raw/own_renders/           ours (scripts/render_ldraw.py), can grow
    └── 3005 brick 1x1/
        ├── 3005_Aqua_0_3000100000.jpeg     part_colour_view_id
        └── ...

Both use the same file names. Images with the same part and id show the same
pose (other colour or camera), so they form one group and stay in one split.

The label is the folder. The internet set puts mould variants of a brick into
its folder, under their own number: 3065 and 93792 in "3004 brick 1x2", 30071
and 35382 in "3005 brick 1x1", 6223 in "3003 brick 2x2" and others. They look
the same from outside, so that is fine. In our renders (strict=True) the
number in the file name must match the folder: a render copied into the wrong
folder would otherwise get the wrong label.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from legorec.data.splits import load_or_create_split
from legorec.paths import SPLITS_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# "3005_Aqua_0_3000100000", "4073_Transparent Yellow_0_3000000000", "2654a_Aqua_0_1"
FILE_NAME_PATTERN = re.compile(
    r"^(?P<part>[^_]+)_(?P<colour>.+)_(?P<view>\d+)_(?P<id>\d+)$"
)

SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_FILES = {
    "internet_renders": SPLITS_DIR / "internet_renders.csv",
    "own_renders": SPLITS_DIR / "own_renders.csv",
}


@dataclass(frozen=True)
class Render:
    path: str  # full path of the image file
    part: str  # part number = the label, e.g. "3005"
    name: str  # folder name, e.g. "3005 brick 1x1"
    group: str  # part number in the file name + id, e.g. "3005_3000100000"


def index_renders(root: str | Path, strict: bool = True) -> list[Render]:
    """Find all renders below `root`; every subfolder is one part.

    strict: the part number in every file name must match its folder.
    """
    root = Path(root)
    part_dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    if not part_dirs:
        raise FileNotFoundError(f"no part folders in {root}")

    renders, bad_names, wrong_folder = [], [], []
    for part_dir in part_dirs:
        name = part_dir.name.strip()
        part = name.split()[0]
        files = sorted(
            (f for f in part_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda f: f.name,
        )
        for file in files:
            match = FILE_NAME_PATTERN.match(file.stem)
            if not match:
                bad_names.append(file.name)
            elif strict and match["part"] != part:
                wrong_folder.append(f"{part_dir.name}/{file.name}")
            else:
                group = f"{match['part']}_{match['id']}"
                renders.append(Render(str(file), part, name, group))

    # never silently drop or mislabel images
    if bad_names:
        raise ValueError(
            f"{len(bad_names)} files in {root} do not match part_colour_view_id, "
            f"e.g. {bad_names[:3]}"
        )
    if wrong_folder:
        raise ValueError(
            f"{len(wrong_folder)} files are in the folder of another part, "
            f"e.g. {wrong_folder[:3]}"
        )
    if not renders:
        raise FileNotFoundError(f"no images in {root}")
    return renders


def load_or_create_render_split(renders: list[Render], source: str):
    """Split by pose, per part 80 / 10 / 10 (data/splits/<source>.csv).

    own_renders can grow: new renders are added to the file, renders already
    in it keep their split. The internet renders are a fixed set: a render
    missing in the file is an error.
    Returns ({"train": [...], "val": [...], "test": [...]}, fingerprint).
    """
    extend = source == "own_renders"
    return load_or_create_split(
        renders, SPLIT_FILES[source], frac=SPLIT_FRACTIONS, extend=extend
    )
