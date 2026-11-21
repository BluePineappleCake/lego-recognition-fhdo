"""Real photos from the internet dataset: find them, group them, split them.

The folder (see data/README.md, section 2b) looks like this:

    internet/
    ├── photos/
    │   ├── 3001 brick 2x4/
    │   │   ├── c0_5_pcsp_original_3001_1609710513579.jpg
    │   │   ├── c1_5_pcsp_original_3001_1609710513579.jpg
    │   │   └── ...
    │   └── ...
    └── renders/        (not used here yet)

The label comes from the folder name ("3001 brick 2x4" -> part "3001"), not
from the file name: some file names contain no part number at all.

The photos were taken with a rig of several cameras. "c0_..." and "c1_..."
with the same rest of the name show the same brick at the same moment from
different angles. If one were in training and the other in the test set, the
test would be far too easy. So we split by capture ("group" below), not by
image - the same idea as camera L / R for the renders.

The split covers all 18 parts of the dataset, also the 5 that are not in our
50 render classes. Filter with only_parts() when loading, so the split file
never has to change when we later use more parts.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from legorec.data.renders import load_or_create_split
from legorec.paths import SPLITS_DIR

SPLIT_FILE = SPLITS_DIR / "internet.csv"

# 70 / 15 / 15: val and test get more than for the renders, because there are
# only 16 to 152 captures per part (renders: 800 poses per part)
SPLIT_FRACTIONS = (0.7, 0.15, 0.15)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# "c0_5_pcsp_original_3001_1609710513579" -> camera "0",
# capture "5_pcsp_original_3001_1609710513579"
FILE_NAME_PATTERN = re.compile(r"^c(?P<camera>\d+)_(?P<capture>.+)$")


@dataclass(frozen=True)
class InternetPhoto:
    path: str  # full path of the image file
    part: str  # part number = the label, e.g. "3001"
    name: str  # folder name, e.g. "3001 brick 2x4"
    group: str  # part + capture, e.g. "3001_5_pcsp_original_3001_1609710513579";
    # the same for all cameras of one capture
    camera: str  # "0", "1", ...


def index_photos(root: str | Path) -> list[InternetPhoto]:
    """Find all photos below `root`/photos (or `root` itself, if it has no photos/).

    Every subfolder is one part; its name starts with the part number.
    """
    root = Path(root)
    photos_dir = root / "photos" if (root / "photos").is_dir() else root

    part_dirs = sorted(d for d in photos_dir.iterdir() if d.is_dir())
    if not part_dirs:
        raise FileNotFoundError(f"no part folders in {photos_dir}")

    photos = []
    unreadable_names = []
    for part_dir in part_dirs:
        name = part_dir.name.strip()  # "3001 brick 2x4"
        part = name.split()[0]  # "3001"
        files = sorted(
            f for f in part_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
        )
        for file in files:
            match = FILE_NAME_PATTERN.match(file.stem)
            if not match:
                unreadable_names.append(file.name)
                continue
            group = f"{part}_{match['capture']}"
            photos.append(InternetPhoto(str(file), part, name, group, match["camera"]))

    if unreadable_names:
        # never silently drop images: a new naming scheme must be looked at
        raise ValueError(
            f"{len(unreadable_names)} files do not match the name pattern, "
            f"e.g. {unreadable_names[:3]}"
        )
    if not photos:
        raise FileNotFoundError(f"no images in {photos_dir}")
    return photos


def load_or_create_photo_split(
    photos: list[InternetPhoto], path=SPLIT_FILE
) -> tuple[dict[str, list[InternetPhoto]], str]:
    """Load data/splits/internet.csv, or create it (70/15/15, per part, by capture).

    Returns ({"train": [...], "val": [...], "test": [...]}, fingerprint).
    Same rules as for the render split: commit the file, never regenerate it.
    """
    return load_or_create_split(photos, path, frac=SPLIT_FRACTIONS)


def only_parts(photos: list[InternetPhoto], parts: set[str]) -> list[InternetPhoto]:
    """Keep the photos whose part is in `parts`, e.g. the parts every model knows."""
    return [photo for photo in photos if photo.part in parts]
