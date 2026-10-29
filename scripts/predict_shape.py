"""Run a trained shape model on photos of single bricks.

    python scripts/predict_shape.py --checkpoint models/shape_<run>.pt photo1.jpg photo2.jpg
    python scripts/predict_shape.py --checkpoint models/shape_<run>.pt my_photos/

Photos: ONE brick per photo, plain background, no strong shadow (a shadow is
counted as part of the brick). If a folder contains subfolders named after part
numbers (my_photos/3001/a.jpg ...), those are used as labels and the script
reports top-1/top-3 accuracy.

Writes predictions.png: what the model actually saw (the 96 x 96 crop) next to
its answer. Look at it first - a bad crop explains most bad predictions.

Models trained with size_feature need --real_scale (photo px -> render px).
Rough calibration: photograph a 2x4 brick (3001) lying flat, from above, with
the same camera height; run this script on it with --real_scale 1 and read the
measured size; then real_scale ~ 373 / measured px (373 px = median size of
3001 in the renders). Sizes are measured after scaling the photo to 1600 px.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from legorec.shape.model import ShapeClassifier

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_SIDE = 1600
PART_DIR = re.compile(r"^\d+[a-z]?$")


def collect(paths):
    """-> [(file, label or None)]; label = parent folder if it looks like a part id."""
    out = []
    for p in map(Path, paths):
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        for f in files if p.is_dir() else [p]:
            parent = f.parent.name
            out.append((f, parent if PART_DIR.match(parent) else None))
    return out


def load(f):
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def contact_sheet(rows, out, parts, tile=192):
    font = ImageFont.load_default(size=13)
    cols = min(6, len(rows))
    h = tile + 64
    sheet = Image.new("RGB", (cols * tile, -(-len(rows) // cols) * h), "white")
    d = ImageDraw.Draw(sheet)
    for i, (f, label, top, arr, px) in enumerate(rows):
        x, y = (i % cols) * tile, (i // cols) * h
        sheet.paste(Image.fromarray(arr).resize((tile, tile)).convert("RGB"), (x, y))
        if label not in parts:  # unlabelled or not a training class
            colour = (0, 0, 0)
        else:
            colour = (20, 130, 60) if top[0][0] == label else (200, 30, 30)
        d.text((x + 4, y + tile + 2), f[:28], fill=(90, 90, 90), font=font)
        d.text(
            (x + 4, y + tile + 17),
            f"pred {top[0][0]} ({top[0][1]:.0%})",
            fill=colour,
            font=font,
        )
        d.text(
            (x + 4, y + tile + 32),
            f"true {label or '?'}  size {px:.0f}px",
            fill=(90, 90, 90),
            font=font,
        )
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--real_scale", type=float, default=None)
    ap.add_argument("--out", default="predictions.png")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        clf = ShapeClassifier(a.checkpoint, dev, a.real_scale)
    except ValueError:
        ap.error(
            "this model was trained with the brick size: pass --real_scale "
            "(rough calibration: see the top of this script)"
        )
    items = collect(a.inputs)
    if not items:
        ap.error("no images found")
    tops, arrays, pxs = clf.predict_images([load(f) for f, _ in items], k=3)

    name = lambda p: clf.names.get(p, p)
    rows, hit1, hit3, n, unknown = [], 0, 0, 0, set()
    for (f, label), top, arr, px in zip(items, tops, arrays, pxs):
        rows.append((f.name, label, top, arr, px))
        guesses = ", ".join(f"{name(p)} {c:.0%}" for p, c in top)
        mark = ""
        if label is not None:
            if label not in clf.parts:
                unknown.add(label)
                mark = "-- "
            else:
                n += 1
                hit1 += top[0][0] == label
                hit3 += label in [p for p, _ in top]
                mark = "OK " if top[0][0] == label else "XX "
        print(f"{mark:4s}{f.name:30s} {px:6.0f}px  {guesses}")

    if n:
        print(f"\n{n} labelled images: top-1 {hit1 / n:.1%}, top-3 {hit3 / n:.1%}")
    if unknown:
        print(f"ignored (part not among the 50 training classes): {sorted(unknown)}")
    print(f"median brick size {np.median(pxs):.0f} px (after scaling to {MAX_SIDE} px)")
    contact_sheet(rows, a.out, set(clf.parts))
    print(f"saved {a.out} - check the crops first")


if __name__ == "__main__":
    main()
