"""Measure the size of every part on our sheet photos: the table of the size
prior (legorec.shape.size_prior).

    python scripts/size_check.py --table      # (--table is optional)

Every brick on the train sheet photos is found with the brick finder; its long
and short side in mm (smallest rotated rectangle around the mask, 8 px/mm)
are collected per part. Median and spread go to data/reference/part_sizes.csv.
Train photos only, so the test photos stay unseen. MIXED photos are left out
(several parts, no label per brick). Run it again after new photos; then commit
the table.
"""

import argparse
from collections import defaultdict

import numpy as np


def train_sheet_crops(method: str) -> dict[str, list]:
    """Every brick on the train sheet photos (BrickCrops), per part."""
    import cv2

    from legorec.data.sheets import index_sheet_photos, load_or_create_sheet_split
    from legorec.paths import own_photos_dir
    from legorec.segment import bricks_on_sheet

    photos = index_sheet_photos(own_photos_dir())
    train = load_or_create_sheet_split(photos)[0]["train"]
    crops_per_part = defaultdict(list)
    for photo in train:
        if photo.part != "MIXED":
            crops_per_part[photo.part] += bricks_on_sheet(
                cv2.imread(photo.path), method
            )
    return crops_per_part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table", action="store_true", help="the only mode (kept for old commands)"
    )
    parser.add_argument("--method", default="yolo", choices=["yolo", "threshold"])
    args = parser.parse_args()

    from legorec.shape.size_prior import PX_PER_MM, TABLE, SizePrior, dimensions_px

    mm = {
        part: [np.array(dimensions_px(crop)) / PX_PER_MM for crop in crops]
        for part, crops in train_sheet_crops(args.method).items()
    }
    prior = SizePrior.fit(mm)
    prior.save()
    print(f"{'part':8}{'n':>5}{'long':>8}{'short':>8}   sd long / short")
    for part, (log_long, log_short, sd_long, sd_short, n) in prior.sizes.items():
        print(
            f"{part:8}{n:>5}{np.exp(log_long):>8.1f}{np.exp(log_short):>8.1f}"
            f"   {sd_long:.3f} / {sd_short:.3f}"
        )
    print(f"[done] -> {TABLE}, commit it")


if __name__ == "__main__":
    main()
