"""Do the renders keep the true brick size? And how do sheet pixels convert to
render pixels?

    python scripts/size_check.py [--data <folder with the render PNGs>]
    python scripts/size_check.py --sheet

Default: prints the median brick size (px) for parts that differ only in
length. If the ratios match the real geometry, the render camera distance is
constant and the size is usable information (it was: see docs/experiments.md).

--sheet: for every render part on the train sheet photos, the median size on
the sheet (bricks_on_sheet, 8 px/mm) and in the test renders, and their ratio.
The median ratio is real_scale for size-feature models on sheet photos
(SHEET_REAL_SCALE in scripts/eval_own.py). Train photos only, so the test
photos stay unseen.
"""

import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

from legorec.data.renders import index_renders, load_or_create_split
from legorec.paths import renders_dir
from legorec.shape.preprocess import preprocess_image

# pairs of parts that differ only in length
PARTS = {"3001", "3002", "3010", "3622", "3069", "3070"}


def render_sizes(images, parts) -> dict[str, list[float]]:
    """Brick size in pixels of every render of `parts`, per part."""
    sizes_per_part = defaultdict(list)
    for image in images:
        if image.part in parts:
            _, brick_size = preprocess_image(Image.open(image.path))
            sizes_per_part[image.part].append(brick_size)
    return sizes_per_part


def sheet_sizes(method: str) -> dict[str, list[float]]:
    """Brick size in pixels of every brick on the train sheet photos, per part."""
    import cv2

    from legorec.data.sheets import index_sheet_photos, load_or_create_sheet_split
    from legorec.paths import own_photos_dir
    from legorec.segment import bricks_on_sheet
    from legorec.shape.preprocess import preprocess_crop

    photos = index_sheet_photos(own_photos_dir())
    train = load_or_create_sheet_split(photos)[0]["train"]
    sizes_per_part = defaultdict(list)
    for photo in train:
        for crop in bricks_on_sheet(cv2.imread(photo.path), method):
            sizes_per_part[photo.part].append(preprocess_crop(crop)[1])
    return sizes_per_part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--sheet", action="store_true")
    parser.add_argument("--method", default="yolo", choices=["yolo", "threshold"])
    args = parser.parse_args()
    images = index_renders(args.data or renders_dir())

    if not args.sheet:
        names = {image.part: image.name for image in images}
        for part, sizes in sorted(render_sizes(images, PARTS).items()):
            low, high = np.percentile(sizes, [10, 90])
            print(
                f"{names[part]:20s} median {np.median(sizes):6.1f} px   "
                f"10-90 %: {low:.0f} to {high:.0f} px"
            )
        return

    on_sheet = sheet_sizes(args.method)
    test_renders = load_or_create_split(images)[0]["test"]
    rendered = render_sizes(test_renders, set(on_sheet))
    ratios = []
    print(f"{'part':8}{'n':>5}{'sheet px':>10}{'mm':>7}{'render px':>11}{'ratio':>8}")
    for part in sorted(set(on_sheet) & set(rendered)):
        sheet, render = np.median(on_sheet[part]), np.median(rendered[part])
        ratios.append(render / sheet)
        print(
            f"{part:8}{len(on_sheet[part]):>5}{sheet:>10.1f}{sheet / 8:>7.1f}"
            f"{render:>11.1f}{ratios[-1]:>8.3f}"
        )
    print(
        f"\nreal_scale = median ratio {np.median(ratios):.3f} "
        f"(range {min(ratios):.3f} to {max(ratios):.3f})"
    )


if __name__ == "__main__":
    main()
