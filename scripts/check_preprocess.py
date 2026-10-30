"""Compare the old (grey threshold) and new (brick_mask) brick detection on renders.

On renders (one brick, plain background) both must give almost the same brick
size. If more than ~1 % of the images differ by more than 2 %, something is
wrong with the new detection.

    python scripts/check_preprocess.py --data C:\\lego\\data [--n 5000]

Writes check_preprocess.png: the 20 largest differences, old crop | new crop.
Result on 2026-10-29: 5,000 renders, 0 differ by more than 2 %, max 1.3 %.
"""

import argparse
import random

import numpy as np
from PIL import Image, ImageDraw

from legorec.data.renders import index_renders
from legorec.paths import renders_dir
from legorec.shape.preprocess import (
    _border_pixels,
    _bounding_box,
    _cut_out_square,
    _longest_side,
    _to_grey,
    preprocess_image,
)

TILE = 96 * 2  # size of one crop in the output image
CAPTION_HEIGHT = 30


def old_preprocess(image, size=96, margin=0.08, thr=20):
    """The detection used for the v0 results: every pixel whose grey value
    differs by more than `thr` from the image border counts as brick."""
    grey = _to_grey(image)
    background = int(np.median(_border_pixels(grey)))
    mask = np.abs(grey - background) > thr
    if not mask.any():
        return None, float(max(grey.shape))

    box = _bounding_box(mask)
    return _cut_out_square(grey, box, background, size, margin), _longest_side(box)


def compare(images):
    """Run old and new preprocessing on every image.

    Returns one row per image: (relative size difference, name,
    old size, new size, old crop, new crop).
    """
    rows = []
    for number, image in enumerate(images, start=1):
        loaded = Image.open(image.path)
        old_crop, old_size = old_preprocess(loaded)
        new_crop, new_size = preprocess_image(loaded)
        difference = abs(new_size - old_size) / old_size
        rows.append((difference, image.name, old_size, new_size, old_crop, new_crop))
        if number % 1000 == 0:
            print(f"{number}/{len(images)}")
    return rows


def save_worst(rows, path, n=20, columns=4):
    """Contact sheet of the n largest differences: old crop left, new crop right."""
    worst = sorted(rows, key=lambda row: -row[0])[:n]
    n_rows = -(-len(worst) // columns)  # rounded up
    sheet = Image.new("L", (columns * 2 * TILE, n_rows * (TILE + CAPTION_HEIGHT)), 255)
    draw = ImageDraw.Draw(sheet)

    for index, (_, name, old_size, new_size, old_crop, new_crop) in enumerate(worst):
        x = (index % columns) * 2 * TILE
        y = (index // columns) * (TILE + CAPTION_HEIGHT)
        for offset, crop in ((0, old_crop), (TILE, new_crop)):
            if crop is not None:
                sheet.paste(Image.fromarray(crop).resize((TILE, TILE)), (x + offset, y))
        caption = f"{name[:24]}  {old_size:.0f} -> {new_size:.0f} px"
        draw.text((x + 4, y + TILE + 2), caption, fill=0)

    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--n", type=int, default=5000, help="0 = all images")
    parser.add_argument("--out", default="check_preprocess.png")
    args = parser.parse_args()

    images = index_renders(args.data or renders_dir())
    if args.n:
        images = random.Random(0).sample(images, min(args.n, len(images)))

    rows = compare(images)
    differences = np.array([row[0] for row in rows])
    changed = differences > 0.02

    print(f"\n{len(rows)} renders checked")
    print(f"size differs by > 2 %: {changed.sum()} ({changed.mean():.2%})")
    print(
        f"median difference: {np.median(differences):.3%}, max: {differences.max():.1%}"
    )
    print("OK" if changed.mean() < 0.01 else "CHECK: more than 1 % changed")

    save_worst(rows, args.out)
    print(f"saved {args.out} (left: old, right: new)")


if __name__ == "__main__":
    main()
