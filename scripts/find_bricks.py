"""Find the bricks on sheet photos and say which part each one is.

    python scripts/find_bricks.py data/raw/photos/MIXED*/IMG_8844.jpg
    python scripts/find_bricks.py photo1.jpg photo2.jpg --method threshold
    python scripts/find_bricks.py photo.jpg --shape-model models/shape_rydhagpu.pt
    python scripts/find_bricks.py photo.jpg --no-shape          # only find bricks

For every photo writes to --out (default $LEGOREC_PROCESSED/find_bricks/):

    <photo>.jpg          the brick area with numbered outlines (+ part, %)
    <photo>_b01.png ...  one crop per brick (as the shape CNN gets it)

and prints one line per brick. Finding bricks: legorec.segment.bricks_on_sheet
("yolo" needs models/bricks_yolo.pt, see there). Parts: the shape model
(default models/shape.pt); its confidences are not calibrated yet.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from legorec.paths import processed_dir
from legorec.segment import brick_area, bricks_on_sheet
from legorec.segment.sheet import PX_PER_MM

COLOUR = (255, 200, 0)  # BGR


def draw(area: np.ndarray, crops, labels: list[str]) -> np.ndarray:
    image = area.copy()
    for number, (crop, label) in enumerate(zip(crops, labels), start=1):
        x, y, _, _ = crop.bbox
        mask = np.zeros(area.shape[:2], np.uint8)
        h, w = crop.mask.shape
        mask[y : y + h, x : x + w] = crop.mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, COLOUR, 4, cv2.LINE_AA)
        cx, cy = contours[0][:, 0].min(axis=0)
        cv2.putText(image, f"{number} {label}".strip(), (int(cx), max(int(cy) - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOUR, 3, cv2.LINE_AA)  # fmt: skip
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--method", default="yolo", choices=["yolo", "threshold"])
    parser.add_argument("--brick-model", default=None, help="YOLO model (.pt)")
    parser.add_argument("--shape-model", default=None, help="shape CNN (.pt)")
    parser.add_argument("--no-shape", action="store_true")
    parser.add_argument("--out", default=str(processed_dir() / "find_bricks"))
    args = parser.parse_args()

    classifier = None
    if not args.no_shape:
        from legorec.shape.predict import load_model  # here: needs torch

        classifier = load_model(args.shape_model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for path in map(Path, args.photos):
        photo = cv2.imread(str(path))
        if photo is None:
            print(f"[skip] {path}: not an image")
            continue
        try:
            crops = bricks_on_sheet(photo, args.method, args.brick_model)
        except ValueError as error:
            print(f"[skip] {path}: {error}")
            continue
        parts = classifier.predict(crops) if classifier and crops else []
        labels = [f"{part} {conf:.0%}" for part, conf in parts] or [""] * len(crops)

        print(f"{path.name}: {len(crops)} bricks ({args.method})")
        for number, (crop, label) in enumerate(zip(crops, labels), start=1):
            name = f"{path.stem}_b{number:02d}.png"
            cv2.imwrite(str(out / name), cv2.cvtColor(crop.image, cv2.COLOR_RGB2BGR))
            print(f"  {number:2d}  {label or '-':<12}  {name}")
        area = brick_area(photo, PX_PER_MM)
        cv2.imwrite(str(out / f"{path.stem}.jpg"), draw(area, crops, labels))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
