"""Our own photos: single-brick crops, one subfolder per part.

The folder looks like this (one crop per brick; the label is the folder):

    photos_own_cutout/
    ├── 3001 brick 2x4/
    │   ├── IMG_3415_b01.png     crop 1 of photo IMG_3415 (segment.sheet)
    │   ├── IMG_3415_b02.png
    │   └── ...
    ├── 3009 brick 1x6/
    │   ├── IMG_3480.jpg         a whole photo cropped by hand: one brick
    │   └── ...
    └── labels.csv               (not read here)

All crops of one photo show bricks of the same part, taken at the same moment
with the same light. So they are not independent: we split and count by
photo ("group"), never by crop.

Session: the phones number their photos consecutively. A jump of more than
SESSION_GAP numbers starts a new session (so far: IMG_3415-3577 and
IMG_8817-8922, two phones). Most parts were photographed in only one session,
so a split by session is not possible per part yet.
"""

import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SESSION_GAP = 20

# "IMG_3415_b01" -> photo "IMG_3415", number 3415
FILE_NAME_PATTERN = re.compile(r"^(?P<photo>IMG_(?P<number>\d+))(_b\d+)?$")


@dataclass(frozen=True)
class OwnCrop:
    path: str  # full path of the crop
    part: str  # part number = the label, e.g. "3001"
    name: str  # folder name, e.g. "3001 brick 2x4"
    group: str  # the photo the crop was cut from, e.g. "IMG_3415"
    session: str  # e.g. "IMG_3415-3577"


def index_own_crops(root: str | Path) -> list[OwnCrop]:
    """Find all crops below `root`; every subfolder is one part."""
    root = Path(root)
    part_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not part_dirs:
        raise FileNotFoundError(f"no part folders in {root}")

    found = []  # (path, part, name, photo, number)
    unreadable_names = []
    for part_dir in part_dirs:
        name = part_dir.name.strip()
        part = name.split()[0]
        for file in sorted(part_dir.iterdir()):
            if file.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            match = FILE_NAME_PATTERN.match(file.stem)
            if not match:
                unreadable_names.append(file.name)
                continue
            found.append((str(file), part, name, match["photo"], int(match["number"])))

    if unreadable_names:
        # never silently drop images: a new naming scheme must be looked at
        raise ValueError(
            f"{len(unreadable_names)} files do not match the name pattern, "
            f"e.g. {unreadable_names[:3]}"
        )
    if not found:
        raise FileNotFoundError(f"no images in {root}")

    session_of = _sessions(sorted({number for *_, number in found}))
    return [
        OwnCrop(path, part, name, photo, session_of[number])
        for path, part, name, photo, number in found
    ]


def _sessions(numbers: list[int]) -> dict[int, str]:
    """Consecutive photo numbers -> session name "IMG_<first>-<last>"."""
    blocks = [[numbers[0]]]
    for number in numbers[1:]:
        if number - blocks[-1][-1] > SESSION_GAP:
            blocks.append([])
        blocks[-1].append(number)
    return {
        number: f"IMG_{block[0]}-{block[-1]}" for block in blocks for number in block
    }
