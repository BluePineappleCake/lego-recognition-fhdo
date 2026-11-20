"""Export our sheet photos as a YOLO segmentation data set (one class: brick).

    python scripts/export_yolo.py
    python scripts/export_yolo.py --photos data/raw/own_photos --background border

For every photo with all 4 ArUco markers:

    1. warp the brick area flat (segment.brick_area, PX_PER_MM pixels per mm)
    2. find the bricks with the colour threshold (segment.find_bricks,
       local background colour)
    3. write the area as images/<split>/<photo>.jpg and one polygon per
       brick to labels/<split>/<photo>.txt

These labels are a first draft: bricks that touch come back as one polygon,
and a brick in the colour of the paper (white on white) is only found by its
shadow. Correct them in CVAT with scripts/cvat_labels.py, photos from
review.csv first. Corrected labels live in data/labels/sheets/ and are
used instead of the automatic ones (and those photos leave review.csv).

review.csv lists every photo with a reason to look at it, as polygon numbers
(1 = first line of the label file):

    big        much larger than the median brick of the photo: probably
               bricks that touch
    ragged     not compact (area / convex hull area < RAGGED): a shadow
               without its brick, or a brick cut in pieces
    edge       touches the edge of the brick area: maybe cut off
    dropped_frame  as wide/high as the whole area: the paper edge around
               the coloured area, not a brick; left out of the labels

Split by photo, per part 70 / 15 / 15, stored in data/splits/sheets.csv.
The output goes to $LEGOREC_PROCESSED/yolo_bricks unless --out is given;
an old export there is deleted first.
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from legorec.data.sheets import index_sheet_photos, load_or_create_sheet_split
from legorec.paths import LABELS_DIR, own_photos_dir, processed_dir
from legorec.segment import brick_area, find_bricks
from legorec.segment.sheet import PX_PER_MM, is_frame
from legorec.segment.yolo import NAMES, label_line, mask_to_polygon, read_label_file

BIG = 1.8  # times the median brick of the photo (two touching bricks: about 2)
RAGGED = 0.75  # real bricks: 95 % have area / hull area >= 0.78
EDGE = 3  # pixels


def solidity(polygon: np.ndarray) -> float:
    hull = cv2.contourArea(cv2.convexHull(polygon.astype(np.int32)))
    return cv2.contourArea(polygon.astype(np.int32)) / hull if hull else 0.0


def touches_edge(polygon: np.ndarray, width: int, height: int) -> bool:
    xs, ys = polygon[:, 0], polygon[:, 1]
    return (
        xs.min() < EDGE
        or ys.min() < EDGE
        or xs.max() > width - 1 - EDGE
        or ys.max() > height - 1 - EDGE
    )


def write_data_yaml(out: Path) -> None:
    names = "\n".join(f"  {k}: {v}" for k, v in NAMES.items())
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def numbers(flags: list[int]) -> str:
    return " ".join(map(str, flags))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", default=str(own_photos_dir()))
    parser.add_argument("--out", default=str(processed_dir() / "yolo_bricks"))
    parser.add_argument("--background", default="local", choices=["local", "border"])
    parser.add_argument(
        "--fixed",
        default=str(LABELS_DIR / "sheets"),
        help="folder with labels corrected by hand (<photo>.txt)",
    )
    args = parser.parse_args()

    photos = index_sheet_photos(args.photos)
    split_of, fingerprint = load_or_create_sheet_split(photos)
    split_of_group = {p.group: s for s, ps in split_of.items() for p in ps}
    out = Path(args.out)
    if out.exists():  # never mix a new export with files of an old one
        shutil.rmtree(out)
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / kind / split).mkdir(parents=True, exist_ok=True)

    fixed_dir = Path(args.fixed)
    review, skipped, counts = [], [], defaultdict(lambda: [0, 0])
    n_fixed = 0
    for photo in photos:
        image = cv2.imread(photo.path)
        area = brick_area(image, PX_PER_MM) if image is not None else None
        if area is None:
            skipped.append(photo.path)
            continue
        h, w = area.shape[:2]
        polygons, frames = [], 0
        fixed = fixed_dir / f"{photo.group}.txt"
        if fixed.exists():  # corrected by hand: use it as it is, no review
            polygons = [p * (w, h) for p in read_label_file(fixed)]
            n_fixed += 1
        for mask in (
            [] if fixed.exists() else find_bricks(area, background=args.background)
        ):
            if is_frame(mask):
                frames += 1
                continue
            polygon = mask_to_polygon(mask)
            if polygon is not None:
                polygons.append(polygon)

        split = split_of_group[photo.group]
        cv2.imwrite(
            str(out / "images" / split / f"{photo.group}.jpg"),
            area,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        lines = [label_line(p, w, h) for p in polygons]
        (out / "labels" / split / f"{photo.group}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        counts[split][0] += 1
        counts[split][1] += len(lines)

        areas = np.array([cv2.contourArea(p.astype(np.int32)) for p in polygons])
        median = np.median(areas) if len(areas) else 0
        big = [i + 1 for i, a in enumerate(areas) if a > BIG * median]
        ragged = [i + 1 for i, p in enumerate(polygons) if solidity(p) < RAGGED]
        edge = [i + 1 for i, p in enumerate(polygons) if touches_edge(p, w, h)]
        if (big or ragged or edge or frames) and not fixed.exists():
            review.append(
                [photo.group, split, len(lines), numbers(big), numbers(ragged),
                 numbers(edge), frames or ""]
            )  # fmt: skip

    write_data_yaml(out)
    review.sort(key=lambda r: (-sum(len(c.split()) for c in r[3:6]), r[0]))
    with open(out / "review.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ["photo", "split", "n_bricks", "big", "ragged", "edge", "dropped_frame"]
        )
        writer.writerows(review)

    print(f"[split] {fingerprint}")
    for split, (n_photos, n_bricks) in counts.items():
        print(f"[{split}] {n_photos} photos, {n_bricks} bricks")
    print(f"[fixed] {n_fixed} photos with labels corrected by hand ({fixed_dir})")
    print(f"[review] {len(review)} photos to check first -> {out / 'review.csv'}")
    if skipped:
        print(f"[skip] {len(skipped)} photos without 4 markers, e.g. {skipped[:3]}")
    print(f"[done] {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
