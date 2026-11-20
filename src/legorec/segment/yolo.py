"""Labels for YOLO segmentation (Ultralytics format), one class: "brick".

YOLO only finds the bricks. Which part a brick is stays the job of our shape
CNN, so the project rule (no ready-made LEGO recognition) holds.

Label file: one line per brick, the outline as a polygon, all coordinates
divided by the image width / height:

    0 x1 y1 x2 y2 x3 y3 ...

The labels are written for the warped brick area (segment.brick_area), not for
the raw photo: there every photo has the same scale (PX_PER_MM), and the
pipeline warps anyway before it looks for bricks.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

BRICK = 0  # the only class
NAMES = {BRICK: "brick"}


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1.0) -> np.ndarray | None:
    """Outline of the largest blob of a boolean mask, as (N, 2) pixel x/y.

    `epsilon`: allowed deviation in pixels when the outline is simplified
    (fewer points, same shape). None if there is no outline with 3 points.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
    return polygon if len(polygon) >= 3 else None


def label_line(polygon: np.ndarray, width: int, height: int, cls: int = BRICK) -> str:
    """One line of a YOLO segmentation label file."""
    xy = polygon.astype(np.float64) / (width, height)
    xy = np.clip(xy, 0.0, 1.0)
    return f"{cls} " + " ".join(f"{v:.5f}" for v in xy.reshape(-1))


def read_label_file(path) -> list[np.ndarray]:
    """Polygons (normalised x/y, shape (N, 2)) of one label file."""
    polygons = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            values = line.split()
            if values:
                polygons.append(np.array(values[1:], float).reshape(-1, 2))
    return polygons


def select_photos(data: Path, names: list[str] | None = None) -> dict[str, Path]:
    """Photos of an exported data set (export_yolo.py): name -> image path.

    `names` None: the photos of review.csv; empty list: all photos.
    Raises ValueError for a name that is not in the data set.
    """
    images = {p.stem: p for p in sorted(Path(data).glob("images/*/*.jpg"))}
    if names is None:
        with open(Path(data) / "review.csv", newline="", encoding="utf-8") as file:
            names = [row["photo"] for row in csv.DictReader(file)]
    elif not names:
        names = list(images)
    unknown = [n for n in names if n not in images]
    if unknown:
        raise ValueError(f"not in {data}/images: {unknown}")
    return {n: images[n] for n in names}
