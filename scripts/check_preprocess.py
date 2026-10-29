"""Compare the old (grey threshold) and new (brick_mask) brick detection on renders.

On renders (one brick, plain background) both must give almost the same size.
If more than ~1 % of the images differ by more than 2 %, something is wrong.

    python scripts/check_preprocess.py --data C:\\lego\\data [--n 5000]

Writes check_preprocess.png: the 20 largest differences, old crop | new crop.
"""

import argparse
import random

import numpy as np
from PIL import Image, ImageDraw

from legorec.data.renders import index_renders
from legorec.paths import renders_dir
from legorec.shape.preprocess import _square_crop, _to_grey, preprocess_image


def old_preprocess(img, size=96, margin=0.08, thr=20):
    """The detection used for the v0 results: every pixel that differs by more
    than `thr` grey values from the border counts as brick."""
    g = _to_grey(img)
    bg = int(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])))
    ys, xs = np.nonzero(np.abs(g - bg) > thr)
    if not len(xs):
        return None, float(max(g.shape))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    px = float(max(y1 - y0, x1 - x0) + 1)
    return _square_crop(g, y0, y1, x0, x1, bg, size, margin), px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=5000, help="0 = all images")
    ap.add_argument("--out", default="check_preprocess.png")
    a = ap.parse_args()

    items = index_renders(a.data or renders_dir())
    if a.n:
        items = random.Random(0).sample(items, min(a.n, len(items)))

    rows = []
    for i, it in enumerate(items):
        img = Image.open(it.path)
        x_old, px_old = old_preprocess(img)
        x_new, px_new = preprocess_image(img)
        rows.append(
            (abs(px_new - px_old) / px_old, it.name, px_old, px_new, x_old, x_new)
        )
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(items)}")

    diff = np.array([r[0] for r in rows])
    changed = diff > 0.02
    print(f"\n{len(rows)} renders checked")
    print(f"size differs by > 2 %: {changed.sum()} ({changed.mean():.2%})")
    print(f"median difference: {np.median(diff):.3%}, max: {diff.max():.1%}")
    print("OK" if changed.mean() < 0.01 else "CHECK: more than 1 % changed")

    worst = sorted(rows, key=lambda r: -r[0])[:20]
    tile, h = 96 * 2, 96 * 2 + 30
    sheet = Image.new("L", (4 * 2 * tile, 5 * h), 255)
    d = ImageDraw.Draw(sheet)
    for k, (dv, name, po, pn, xo, xn) in enumerate(worst):
        x, y = (k % 4) * 2 * tile, (k // 4) * h
        for j, arr in enumerate((xo, xn)):
            if arr is not None:
                sheet.paste(
                    Image.fromarray(arr).resize((tile, tile)), (x + j * tile, y)
                )
        d.text((x + 4, y + tile + 2), f"{name[:24]}  {po:.0f} -> {pn:.0f} px", fill=0)
    sheet.save(a.out)
    print(f"saved {a.out} (left: old, right: new)")


if __name__ == "__main__":
    main()
