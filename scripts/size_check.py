"""Do the renders keep the true brick size?

Prints the median brick size (px) for parts that differ only in length. If the
ratios match the real geometry, the render camera distance is constant and the
size is usable information (it was: see docs/experiments.md).

    python scripts/size_check.py [--data <folder with the render PNGs>]
"""

import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

from legorec.data.renders import index_renders
from legorec.paths import renders_dir
from legorec.shape.preprocess import preprocess_image

# pairs of parts that differ only in length
PARTS = {"3001", "3002", "3010", "3622", "3069", "3070"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    sizes_per_part = defaultdict(list)
    for image in index_renders(args.data or renders_dir()):
        if image.part in PARTS:
            _, brick_size = preprocess_image(Image.open(image.path))
            sizes_per_part[image.name].append(brick_size)

    for name, sizes in sorted(sizes_per_part.items()):
        low, high = np.percentile(sizes, [10, 90])
        print(
            f"{name:20s} median {np.median(sizes):6.1f} px   "
            f"10-90 %: {low:.0f} to {high:.0f} px"
        )


if __name__ == "__main__":
    main()
