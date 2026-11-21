"""Export our sheet photos as a YOLO segmentation data set (one class: brick).

    python scripts/export_yolo.py
    python scripts/export_yolo.py --model runs/segment/train2/weights/best.pt

For every photo with all 4 ArUco markers:

    1. warp the brick area flat (segment.brick_area, PX_PER_MM pixels per mm)
    2. take the labels corrected by hand (data/labels/sheets/<photo>.txt) or,
       if there are none, the bricks the current YOLO model finds (a draft)
    3. write the area as images/<split>/<photo>.jpg and one polygon per
       brick to labels/<split>/<photo>.txt

Drafts are not the truth: correct them in CVAT with scripts/cvat_labels.py,
photos from review.csv first. Corrected labels are used instead of the drafts
(and those photos leave review.csv). Commit them: data/labels/sheets/ is in
Git.

Val and test first: a YOLO model trained on this export is measured on the
val / test labels. Where those are uncorrected drafts of the previous model,
its mAP only says how well it agrees with that model, not how good it is. The
export prints how many val / test photos are still uncorrected.

review.csv lists every uncorrected val / test photo (first) and every other
uncorrected photo with a reason to look at it, as polygon numbers (1 = first
line of the label file):

    big        much larger than the median brick of the photo: probably
               bricks that touch
    ragged     not compact (area / convex hull area < RAGGED): a shadow
               without its brick, or a brick cut in pieces
    edge       touches the edge of the brick area: maybe cut off

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
from legorec.paths import LABELS_DIR, brick_model_path, own_photos_dir, processed_dir
from legorec.segment import brick_area
from legorec.segment.bricks import brick_masks
from legorec.segment.sheet import PX_PER_MM
from legorec.segment.yolo import NAMES, label_line, mask_to_polygon, read_label_file

BIG = 1.8  # times the median brick of the photo (two touching bricks: about 2)
RAGGED = 0.75  # real bricks: 95 % have area / hull area >= 0.78
EDGE = 3  # pixels
EVAL_SPLITS = ("val", "test")  # YOLO is measured on these


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
    parser.add_argument(
        "--model",
        default=str(brick_model_path()),
        help="YOLO model for the drafts (default: models/bricks_yolo.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence")
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
    n_fixed, unverified = 0, defaultdict(int)  # eval split -> uncorrected photos
    for photo in photos:
        image = cv2.imread(photo.path)
        area = brick_area(image, PX_PER_MM) if image is not None else None
        if area is None:
            skipped.append(photo.path)
            continue
        h, w = area.shape[:2]
        split = split_of_group[photo.group]
        fixed = fixed_dir / f"{photo.group}.txt"
        if fixed.exists():  # corrected by hand: use it as it is, no review
            polygons = [p * (w, h) for p in read_label_file(fixed)]
            n_fixed += 1
        else:  # a draft of the current model
            masks = brick_masks(area, args.model, args.conf)
            polygons = [p for p in map(mask_to_polygon, masks) if p is not None]
            if split in EVAL_SPLITS:
                unverified[split] += 1

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
        if not fixed.exists() and (big or ragged or edge or split in EVAL_SPLITS):
            review.append(
                [photo.group, split, len(lines), numbers(big), numbers(ragged),
                 numbers(edge)]
            )  # fmt: skip

    write_data_yaml(out)
    # val / test first, then the photos with the most flags
    review.sort(
        key=lambda r: (
            r[1] not in EVAL_SPLITS,
            -sum(len(c.split()) for c in r[3:6]),
            r[0],
        )
    )
    with open(out / "review.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["photo", "split", "n_bricks", "big", "ragged", "edge"])
        writer.writerows(review)

    print(f"[split] {fingerprint}")
    for split, (n_photos, n_bricks) in counts.items():
        print(f"[{split}] {n_photos} photos, {n_bricks} bricks")
    print(f"[fixed] {n_fixed} photos with labels corrected by hand ({fixed_dir})")
    print(f"[draft] the other photos: drafts of {args.model}")
    if unverified:
        counts_text = ", ".join(
            f"{n} {split}" for split, n in sorted(unverified.items())
        )
        print(
            f"[warn] uncorrected {counts_text} photos: a YOLO model trained on this "
            "export is measured against drafts there, not against the truth. "
            "Correct them first (review.csv lists them first)."
        )
    print(f"[review] {len(review)} photos to check first -> {out / 'review.csv'}")
    if skipped:
        print(f"[skip] {len(skipped)} photos without 4 markers, e.g. {skipped[:3]}")
    print(f"[done] {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
