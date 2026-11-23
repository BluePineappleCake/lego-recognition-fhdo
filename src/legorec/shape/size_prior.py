"""Size prior: how well do the measured dimensions of a brick fit each part?

On a sheet photo the brick area is warped to PX_PER_MM pixels per millimetre,
so the size of a crop is known in mm. Measured: the long and the short side of
the smallest rotated rectangle around its mask. A 1x1 brick (~14 mm) is then a
bad fit for 3004 (1x2), and a 1x2 (short side ~10 mm) a bad fit for 3003
(2x2, ~17 mm), however much they look alike from above. The CNN cannot know
that: it sees every brick scaled to 96 x 96.

The table data/reference/part_sizes.csv holds per part the median and the
spread of log(long side) and log(short side) in mm, measured on the train
sheet photos with the brick finder (scripts/size_check.py). It is the
same measurement as at prediction time, so the mask and perspective (the top
of a brick is closer to the camera than the sheet) are already in it.

Used as log P(dimensions | part), added to the scores of the CNN (= log
posterior up to a constant). A part without a row gets the size distribution
of all parts (see log_likelihood). Only for crops from a sheet: other photos
have no known scale. A brick and a plate with the same footprint stay
unsolved: same dimensions from above.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from legorec.paths import REPO_ROOT

TABLE = REPO_ROOT / "data" / "reference" / "part_sizes.csv"
PX_PER_MM = 8  # same as legorec.segment.sheet.PX_PER_MM (that module needs cv2)
# lower bound of the spread: ~5 % (1 mm at 20 mm); with few crops the
# measured spread can be smaller by chance, and the filter would be too strict
MIN_SD = 0.05

# per part: median of log(long), of log(short), robust sd of each, n crops
Row = tuple[float, float, float, float, int]


def dimensions_px(crop) -> tuple[float, float]:
    """(long side, short side) in pixels of the smallest rotated rectangle
    around the mask of a BrickCrop. Unlike an upright box, it does not depend
    on how the brick is turned."""
    import cv2

    points = cv2.findNonZero(crop.mask.astype(np.uint8))
    if points is None:
        return 1.0, 1.0
    _, (width, height), _ = cv2.minAreaRect(points)
    return max(width, height, 1.0), max(min(width, height), 1.0)


def _median_and_sd(values: np.ndarray) -> tuple[float, float]:
    """Median and robust spread (1.4826 x median absolute deviation).

    Robust, because a few bad crops (two touching bricks found as one) would
    blow up a plain standard deviation.
    """
    median = float(np.median(values))
    sd = 1.4826 * float(np.median(np.abs(values - median)))
    return median, max(sd, MIN_SD)


@dataclass
class SizePrior:
    sizes: dict[str, Row]
    px_per_mm: float = PX_PER_MM
    path: Path | None = field(default=None, compare=False)

    @classmethod
    def fit(cls, dimensions_mm_per_part: dict[str, list]) -> "SizePrior":
        """{part: [(long, short) in mm, ...]} -> SizePrior.

        A part without any measured brick gets no row (it then uses the size
        of all parts, see log_likelihood). An empty row would be NaN, and one
        NaN column makes every answer NaN.
        """
        sizes = {}
        for part, dims in sorted(dimensions_mm_per_part.items()):
            if len(dims) == 0:
                continue
            logs = np.log(np.asarray(dims, dtype=np.float64).reshape(-1, 2))
            long_mm, long_sd = _median_and_sd(logs[:, 0])
            short_mm, short_sd = _median_and_sd(logs[:, 1])
            sizes[part] = (long_mm, short_mm, long_sd, short_sd, len(logs))
        return cls(sizes)

    @classmethod
    def load(cls, path=TABLE) -> "SizePrior":
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        columns = ["log_long", "log_short", "sd_long", "sd_short"]
        sizes = {
            row["part"]: (*(float(row[c]) for c in columns), int(row["n"]))
            for row in rows
        }
        broken = [part for part, row in sizes.items() if not np.all(np.isfinite(row))]
        if broken:
            raise ValueError(
                f"{path}: no valid size for {broken}. Rebuild the table with "
                "scripts/size_check.py"
            )
        return cls(sizes, path=Path(path))

    def save(self, path=TABLE) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(
                ["part", "n", "long_mm", "short_mm"]
                + ["log_long", "log_short", "sd_long", "sd_short"]
            )
            for part, (log_long, log_short, sd_long, sd_short, n) in sorted(
                self.sizes.items()
            ):
                writer.writerow(
                    [part, n, f"{np.exp(log_long):.1f}", f"{np.exp(log_short):.1f}"]
                    + [f"{v:.4f}" for v in (log_long, log_short, sd_long, sd_short)]
                )

    def log_likelihood(self, crops, parts: list[str]) -> np.ndarray:
        """N crops of a sheet photo -> (N, len(parts)), to add to the scores.

        Sum of two normal log densities, one per side. A part without a row
        gets the density of all parts in the table (a wide normal per side):
        worse than a part whose size fits, better than one whose size does
        not. With 0 instead, a model with more parts than the table would move
        its answers to the parts without a size.
        """
        dims = np.array([dimensions_px(crop) for crop in crops], dtype=np.float64)
        logs = np.log(dims.reshape(-1, 2) / self.px_per_mm)
        rows = np.array([row[:4] for row in self.sizes.values()])
        everything = (
            *rows[:, :2].mean(axis=0),
            *np.maximum(rows[:, :2].std(axis=0), MIN_SD),
        )
        out = np.empty((len(logs), len(parts)), dtype=np.float32)
        for j, part in enumerate(parts):
            log_long, log_short, sd_long, sd_short = self.sizes.get(part, everything)[
                :4
            ]
            out[:, j] = (
                -np.log(sd_long)
                - 0.5 * ((logs[:, 0] - log_long) / sd_long) ** 2
                - np.log(sd_short)
                - 0.5 * ((logs[:, 1] - log_short) / sd_short) ** 2
            )
        return out
