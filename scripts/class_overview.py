"""
Overview of all brick classes in the Kaggle render dataset ("Images of LEGO Bricks").

One cell per class: part number + name, image/pose counts and N renders of
different poses side by side. Classes are grouped by part type (brick, plate,
tile, slope, ...) so look-alikes and length variants end up next to each other.

    python scripts/class_overview.py --data C:\\lego\\data
    python scripts/class_overview.py --data C:\\lego\\data --views 3 --cols 4 --out docs/figures/classes

Outputs  <out>.png  (slides / README),  <out>.pdf  (report, vector text)
         <out>.csv  (part, name, type, images, poses)

--data must point to the folder that directly contains the 40,000 PNGs
(the parent folder also holds an older version of the dataset).
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# same pattern as index_renders: "<part> <name> <idx><L|R>.png"
NAME_RE = re.compile(r"^(?P<cls>.+?)\s+(?P<idx>\d+)(?P<view>[LR]?)$")

# order of the type groups in the figure
TECHNIC = (
    "technic",
    "peg",
    "beam",
    "lever",
    "connector",
    "bush",
    "cross block",
    "axle",
)
TYPE_ORDER = ["Brick", "Plate", "Tile", "Slope", "Technic", "Other"]


def part_type(name):
    low = name.lower()
    if any(k in low for k in TECHNIC):
        return "Technic"
    if "roof" in low:  # roof tiles / roof corners are slopes, not tiles
        return "Slope"
    for t in ("Brick", "Plate", "Tile"):
        if t.lower() in low:
            return t
    return "Other"


def part_key(part):
    m = re.match(r"(\d+)", part)  # numeric sort: 3001 < 3022 < 27925
    return (int(m[1]) if m else 10**9, part)


def index(root):
    classes = defaultdict(lambda: {"name": None, "poses": defaultdict(dict)})
    for p in Path(root).rglob("*.png"):
        m = NAME_RE.match(p.stem)
        if not m:
            continue
        cls = m["cls"].strip()
        part = cls.split()[0]
        c = classes[part]
        c["name"] = cls[len(part) :].strip()
        c["poses"][m["idx"]][m["view"] or "-"] = p
    if not classes:
        raise SystemExit(f"no matching .png under {root}")
    return classes


def pick_views(poses, n, rng):
    """n different poses, one camera view each (L preferred) -> shows pose variety."""
    ids = sorted(poses)
    ids = rng.sample(ids, min(n, len(ids)))
    return [poses[i].get("L") or next(iter(poses[i].values())) for i in sorted(ids)]


def crop_to_brick(img, margin=0.12, thr=20):
    """Square crop around the brick (same idea as preprocess, but in colour).
    Makes small parts visible; absolute size is lost -> use --no_crop to compare sizes."""
    rgb = np.asarray(img.convert("RGB"))
    g = rgb.astype(np.int16).mean(2)
    bg = np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]))
    ys, xs = np.nonzero(np.abs(g - bg) > thr)
    if not len(xs):
        return rgb
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * margin)) + 1
    r0, c0 = cy - side // 2, cx - side // 2
    out = np.full((side, side, 3), np.median(rgb[0], 0), np.uint8)  # pad, do not clamp
    a0, b0 = max(r0, 0), max(c0, 0)
    a1, b1 = min(r0 + side, rgb.shape[0]), min(c0 + side, rgb.shape[1])
    out[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0] = rgb[a0:a1, b0:b1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="class_overview")
    ap.add_argument("--views", type=int, default=4, help="renders per class")
    ap.add_argument("--cols", type=int, default=5, help="classes per row")
    ap.add_argument("--seed", type=int, default=0, help="which poses are shown")
    ap.add_argument(
        "--no_crop", action="store_true", help="full render, keeps relative size"
    )
    ap.add_argument("--dpi", type=int, default=200)
    cfg = ap.parse_args()
    rng = random.Random(cfg.seed)

    classes = index(cfg.data)
    parts = sorted(
        classes,
        key=lambda p: (TYPE_ORDER.index(part_type(classes[p]["name"])), part_key(p)),
    )
    n_img = {p: sum(len(v) for v in classes[p]["poses"].values()) for p in parts}

    # ---- table ----
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part", "name", "type", "images", "poses"])
        for p in parts:
            c = classes[p]
            w.writerow([p, c["name"], part_type(c["name"]), n_img[p], len(c["poses"])])

    # ---- figure: one block per type, each block starts on a new row ----
    layout = []  # list of rows; a row is a list of parts or a type header string
    for t in TYPE_ORDER:
        group = [p for p in parts if part_type(classes[p]["name"]) == t]
        if not group:
            continue
        layout.append(f"{t}  ({len(group)})")
        layout += [group[i : i + cfg.cols] for i in range(0, len(group), cfg.cols)]

    cell_w, img_h, title_h, header_h = 0.95, 0.95, 0.42, 0.38  # inches
    fig_w = cfg.cols * cfg.views * cell_w + (cfg.cols - 1) * 0.25
    heights = [header_h if isinstance(r, str) else img_h + title_h for r in layout]
    fig = plt.figure(figsize=(fig_w, sum(heights) + 0.9))
    fig.suptitle(
        f"Kaggle 'Images of LEGO Bricks': {len(parts)} classes, "
        f"{sum(n_img.values()):,} renders"
        + ("" if cfg.no_crop else "  (cropped, not to scale)"),
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    rows = fig.subfigures(len(layout), 1, height_ratios=heights, hspace=0)
    rows = [rows] if len(layout) == 1 else rows

    for sub, row in zip(rows, layout):
        if isinstance(row, str):
            sub.text(0.0, 0.25, row, fontsize=13, fontweight="bold", color="#4C72B0")
            continue
        cells = sub.subfigures(1, cfg.cols, wspace=0.03)
        cells = [cells] if cfg.cols == 1 else cells
        for k, cell in enumerate(cells):
            if k >= len(row):
                continue
            p = row[k]
            c = classes[p]
            cell.suptitle(
                f"{p}  {c['name']}\n{n_img[p]} images, {len(c['poses'])} poses",
                fontsize=8.5,
                y=0.99,
                va="top",
            )
            axes = cell.subplots(1, cfg.views)
            axes = [axes] if cfg.views == 1 else axes
            cell.subplots_adjust(
                left=0.01,
                right=0.99,
                bottom=0.01,
                top=img_h / (img_h + title_h),
                wspace=0.04,
            )
            for ax, path in zip(
                axes, pick_views(c["poses"], cfg.views, rng) + [None] * cfg.views
            ):
                ax.set_axis_off()
                if path is not None:
                    img = Image.open(path)
                    ax.imshow(
                        np.asarray(img.convert("RGB"))
                        if cfg.no_crop
                        else crop_to_brick(img)
                    )

    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), dpi=cfg.dpi, facecolor="white")
    print(f"{len(parts)} classes -> {out.with_suffix('.png')}, .pdf, .csv")
    for t in TYPE_ORDER:
        g = [
            f"{p} {classes[p]['name']}"
            for p in parts
            if part_type(classes[p]["name"]) == t
        ]
        if g:
            print(f"  {t}: " + "; ".join(g))


if __name__ == "__main__":
    main()
