"""Draw the YOLO labels onto their images, to check them by eye.

    python scripts/show_labels.py                      # all photos in review.csv
    python scripts/show_labels.py 3001_IMG_3482 3004_IMG_8877
    python scripts/show_labels.py --all                # every photo

Every polygon gets its number (1 = first line of the label file, the numbers
used in review.csv). Writes one PNG per photo to <data>/preview/ and a
contact sheet preview/_overview.jpg; open them with any image viewer.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from legorec.paths import processed_dir
from legorec.segment.yolo import read_label_file

COLOUR = (0, 255, 255)  # BGR yellow
TILE = (540, 620)


def draw(image_path: Path, label_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    polygons = read_label_file(label_path) if label_path.exists() else []
    for number, polygon in enumerate(polygons, start=1):
        points = (polygon * (w, h)).astype(np.int32)
        cv2.polylines(image, [points], True, COLOUR, 3)
        x, y = points.min(axis=0)
        cv2.putText(image, str(number), (int(x), max(int(y) - 8, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, COLOUR, 4)  # fmt: skip
    cv2.putText(image, f"{image_path.stem} ({len(polygons)})", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)  # fmt: skip
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="*", help="e.g. 3001_IMG_3482")
    parser.add_argument("--data", default=str(processed_dir() / "yolo_bricks"))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    data = Path(args.data)
    images = {p.stem: p for p in sorted(data.glob("images/*/*.jpg"))}
    if args.photos:
        names = args.photos
    elif args.all:
        names = list(images)
    else:
        with open(data / "review.csv", newline="", encoding="utf-8") as file:
            names = [row["photo"] for row in csv.DictReader(file)]
    unknown = [n for n in names if n not in images]
    if unknown:
        raise SystemExit(f"not in {data}/images: {unknown}")

    out = data / "preview"
    out.mkdir(exist_ok=True)
    tiles = []
    for name in names:
        image_path = images[name]
        label_path = data / "labels" / image_path.parent.name / f"{name}.txt"
        image = draw(image_path, label_path)
        cv2.imwrite(str(out / f"{name}.png"), image)
        tiles.append(cv2.resize(image, TILE))
    columns = 4
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(
        str(out / "_overview.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    print(f"[done] {len(names)} photos -> {out} (overview: _overview.jpg)")


if __name__ == "__main__":
    main()
