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

PARTS = {"3001", "3002", "3010", "3622", "3069", "3070"}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None)
a = ap.parse_args()

sizes = defaultdict(list)
for it in index_renders(a.data or renders_dir()):
    if it.part in PARTS:
        sizes[it.name].append(preprocess_image(Image.open(it.path))[1])
for name, v in sorted(sizes.items()):
    print(
        f"{name:20s} median {np.median(v):6.1f} px   10-90 %: {np.percentile(v, [10, 90])}"
    )
