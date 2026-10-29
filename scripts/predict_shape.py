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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_SIDE = 1600  # photos are scaled down to at most this many pixels per side
PART_FOLDER = re.compile(r"^\d+[a-z]?$")  # folder names like "3001" or "3794b"

TILE = 192  # size of one crop in predictions.png
CAPTION_HEIGHT = 64
GREY, GREEN, RED, BLACK = (90, 90, 90), (20, 130, 60), (200, 30, 30), (0, 0, 0)


def collect_images(paths):
    """Files and folders -> [(file, label or None)].

    The label is the name of the folder the photo is in, if it looks like a
    part number (e.g. my_photos/3001/a.jpg -> "3001"). Otherwise None.
    """
    found = []
    for path in map(Path, paths):
        if path.is_dir():
            files = sorted(
                f for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            files = [path]
        for file in files:
            folder = file.parent.name
            label = folder if PART_FOLDER.match(folder) else None
            found.append((file, label))
    return found


def load_photo(path):
    """Open a photo, turn it upright (phones store the rotation separately)
    and scale it down to at most MAX_SIDE pixels."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def save_contact_sheet(rows, path, known_parts):
    """predictions.png: every crop the model saw, with its answer below.

    Green = right, red = wrong, black = no label or a part the model does not know.
    """
    font = ImageFont.load_default(size=13)
    n_columns = min(6, len(rows))
    n_rows = -(-len(rows) // n_columns)  # rounded up
    sheet = Image.new(
        "RGB", (n_columns * TILE, n_rows * (TILE + CAPTION_HEIGHT)), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, (file_name, label, top, crop, size_px) in enumerate(rows):
        x = (index % n_columns) * TILE
        y = (index // n_columns) * (TILE + CAPTION_HEIGHT)
        sheet.paste(Image.fromarray(crop).resize((TILE, TILE)).convert("RGB"), (x, y))

        best_part, best_probability = top[0]
        if label not in known_parts:
            colour = BLACK
        elif best_part == label:
            colour = GREEN
        else:
            colour = RED

        draw.text((x + 4, y + TILE + 2), file_name[:28], fill=GREY, font=font)
        draw.text(
            (x + 4, y + TILE + 17),
            f"pred {best_part} ({best_probability:.0%})",
            fill=colour,
            font=font,
        )
        draw.text(
            (x + 4, y + TILE + 32),
            f"true {label or '?'}  size {size_px:.0f}px",
            fill=GREY,
            font=font,
        )
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("inputs", nargs="+", help="image files and/or folders")
    parser.add_argument("--real_scale", type=float, default=None)
    parser.add_argument("--out", default="predictions.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        classifier = ShapeClassifier(args.checkpoint, device, args.real_scale)
    except ValueError:
        parser.error(
            "this model was trained with the brick size: pass --real_scale "
            "(rough calibration: see the top of this script)"
        )

    photos = collect_images(args.inputs)
    if not photos:
        parser.error("no images found")

    images = [load_photo(file) for file, _ in photos]
    top_3_per_image, crops, sizes_px = classifier.predict_images(images, k=3)

    def readable(part):
        return classifier.names.get(part, part)

    rows = []
    n_labelled, n_top1_right, n_top3_right = 0, 0, 0
    unknown_labels = set()

    for (file, label), top_3, crop, size_px in zip(
        photos, top_3_per_image, crops, sizes_px
    ):
        rows.append((file.name, label, top_3, crop, size_px))
        guesses = ", ".join(f"{readable(part)} {prob:.0%}" for part, prob in top_3)
        best_part = top_3[0][0]

        if label is None:
            mark = ""
        elif label not in classifier.parts:
            unknown_labels.add(label)
            mark = "-- "
        else:
            n_labelled += 1
            n_top1_right += best_part == label
            n_top3_right += label in [part for part, _ in top_3]
            mark = "OK " if best_part == label else "XX "

        print(f"{mark:4s}{file.name:30s} {size_px:6.0f}px  {guesses}")

    if n_labelled:
        print(
            f"\n{n_labelled} labelled images: "
            f"top-1 {n_top1_right / n_labelled:.1%}, "
            f"top-3 {n_top3_right / n_labelled:.1%}"
        )
    if unknown_labels:
        print(
            f"ignored (part not among the 50 training classes): "
            f"{sorted(unknown_labels)}"
        )
    print(
        f"median brick size {np.median(sizes_px):.0f} px "
        f"(after scaling to {MAX_SIDE} px)"
    )

    save_contact_sheet(rows, args.out, set(classifier.parts))
    print(f"saved {args.out} - check the crops first")


if __name__ == "__main__":
    main()
