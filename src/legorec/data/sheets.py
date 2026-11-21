"""Our photos of the capture sheet (template v2): find them and split them.

The folder (data/raw/own_photos, see data/README.md) has one subfolder per part:

    own_photos/
    ├── 3001 brick 2x4/
    │   ├── IMG_3415.jpg
    │   └── ...
    ├── MIXED 3009+3710+3023/      several parts on one photo
    └── ...

Images in deeper subfolders count for the top folder. A folder name must
start with the part number (or MIXED) and a space: the part number is read
from it, so any other name with photos of its own is an error.

A photo that is a byte for byte copy of one found before is skipped (e.g.
lego-real-photos/, a sorted copy of all photos): otherwise the same photo
could end up in train and test. Folders are read in sorted order, so the
original part folders come first.

The "part" of a photo is the first word of its top folder ("3001", "MIXED");
it is only used to spread every part evenly over train / val / test. For
finding bricks (YOLO, one class) the part does not matter.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from legorec.data.renders import load_or_create_split
from legorec.paths import SPLITS_DIR

SPLIT_FILE = SPLITS_DIR / "sheets.csv"
SPLIT_FRACTIONS = (0.7, 0.15, 0.15)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# "3001 brick 2x4", "3062b round brick 1x1", "MIXED 3009+3710+3023"
FOLDER_NAME = re.compile(r"^(\d+[a-z]?|MIXED) \S")


@dataclass(frozen=True)
class SheetPhoto:
    path: str
    part: str  # first word of the top folder, e.g. "3001" or "MIXED"
    folder: str  # top folder, e.g. "3001 brick 2x4"
    group: str  # unique name of the photo, e.g. "3001_IMG_3415"; used as file name


def index_sheet_photos(root: str | Path) -> list[SheetPhoto]:
    root = Path(root)
    # sort by the plain name: Windows sorts paths case-insensitively, Linux not,
    # and which copy of a photo is kept must not depend on the computer
    folders = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    if not folders:
        raise FileNotFoundError(f"no folders in {root}")
    photos, groups, seen, copies, bad_names = [], set(), {}, [], set()
    for folder in folders:
        part = folder.name.split()[0]
        for file in sorted(folder.rglob("*"), key=lambda f: f.as_posix()):
            if file.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = (file.name.lower(), file.stat().st_size)  # cheap first check
            if key in seen and _same_content(seen[key], file):
                copies.append(file)
                continue
            seen.setdefault(key, file)
            if not FOLDER_NAME.match(folder.name):
                bad_names.add(folder.name)
                continue
            group = f"{part}_{file.stem}"
            if group in groups:
                raise ValueError(f"two photos named {group}: rename one of them")
            groups.add(group)
            photos.append(SheetPhoto(str(file), part, folder.name, group))
    if bad_names:
        # the part number comes from the folder name: a typo must not become a part
        raise ValueError(
            f"folder names must look like '3001 brick 2x4' or 'MIXED 3009+3710': "
            f"{sorted(bad_names)}"
        )
    if copies:
        print(
            f"[photos] skipped {len(copies)} copies of other photos, e.g. {copies[0]}"
        )
    if not photos:
        raise FileNotFoundError(f"no images below {root}")
    return photos


def _same_content(a: Path, b: Path) -> bool:
    def digest(path):
        return hashlib.md5(path.read_bytes()).hexdigest()

    return digest(a) == digest(b)


def load_or_create_sheet_split(photos: list[SheetPhoto], path=SPLIT_FILE):
    """Split by photo, per part 70 / 15 / 15. Same rules as for the renders:
    the file is created once and committed. New photos are added to it (by a
    hash of their name), photos already in it never change their split."""
    return load_or_create_split(photos, path, frac=SPLIT_FRACTIONS, extend=True)
