"""Evaluate shape models on our own photos.

    python scripts/eval_own.py                     # data/raw/own_crops, all models
    python scripts/eval_own.py --models models/shape_rydhagpu.pt
    python scripts/eval_own.py --source yolo       # whole pipeline, test photos
    python scripts/eval_own.py --source threshold  # same with find_bricks

Two kinds of input (--source):

    crops       data/raw/own_crops: cut with the first find_bricks, bad crops
                removed by hand. The shape model alone, on clean crops; fixed,
                so numbers stay comparable over time.
    yolo /      the sheet photos (data/raw/own_photos) cut by bricks_on_sheet,
    threshold   as in the real pipeline. Every brick on a photo of a part
                folder is that part, so no labelling is needed (MIXED photos
                are left out). Whatever the segmentation finds is counted,
                also a half brick or a shadow: photo to part number, end to
                end. Missed bricks do not show up here (see export_yolo.py).
                Default --split test: YOLO was trained on train / val photos.

To compare the two on the same photos: --source crops --split test.

--size-prior (yolo / threshold): also weigh the measured size of every brick
in mm (legorec.shape.size_prior). Run once with and once without it.

Only crops of parts that every evaluated model knows are used (13 for the
Kaggle models; pass --models to evaluate one model on all of its parts). Crops of one photo are not independent (same part, same moment, same
light), so the honest numbers are per photo:

    top1 / top3 / mF1 / conf   per crop, 50-way, as for the renders
    top1_k        best guess among the k parts that occur in these crops only
    photo_top1    one answer per photo: mean probability over its crops

Models are grouped into variants by their config (tags + contrast); the table
shows mean and range over the seeds of a variant. Then per part and per
session, and an exact McNemar test per photo between seed-matched models of
two variants.

Predictions: $LEGOREC_PROCESSED/eval_own_<model>.csv (eval_own_<source>_<model>.csv
for yolo / threshold).
Size-feature models need the brick size in render pixels. The sheet photos
are warped to 8 px/mm, so for yolo / threshold one factor converts them:
SHEET_REAL_SCALE (scripts/size_check.py --sheet). The crops in own_crops have
no common scale (the .jpg are cut by hand): size models are skipped there,
unless --real_scale is given.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from legorec.data.own import OwnCrop, index_own_crops, photo_sessions
from legorec.data.sheets import index_sheet_photos, load_or_create_sheet_split
from legorec.eval.metrics import (
    classification_metrics,
    group_top1,
    mcnemar_p,
    softmax,
    top1_among,
)
from legorec.paths import MODELS_DIR, own_crops_dir, own_photos_dir, processed_dir
from legorec.shape.model import ShapeClassifier, parts_known_by_all
from legorec.shape.size_prior import SizePrior

# render px / sheet px for the same part, median over the 13 parts on the
# train sheet photos (1.16 to 1.43 per part): scripts/size_check.py --sheet
SHEET_REAL_SCALE = 1.28


def variant_and_seed(checkpoint: Path) -> tuple[str, int]:
    cfg = torch.load(checkpoint, map_location="cpu", weights_only=False)["cfg"]
    tags = [t for t in cfg.get("tags") or [] if t != "contrast"]
    tags = "+".join(tags + ["size"] * bool(cfg.get("size_feature")))
    return f"{tags or 'untagged'}/{cfg.get('contrast') or 'none'}", cfg.get("seed")


def sheet_photos_per_split():
    """{"train": [...], "val": [...], "test": [...]} of our sheet photos."""
    photos = index_sheet_photos(own_photos_dir())
    return load_or_create_sheet_split(photos)[0]


def photos_of(photos_per_split, split):
    """The photos of one split ("all": every photo)."""
    return [
        photo
        for name, in_split in photos_per_split.items()
        if split in ("all", name)
        for photo in in_split
    ]


def own_crops(folder, split, known):
    """The hand-checked crops -> (crops, their images, skipped folder names)."""
    all_crops = index_own_crops(folder)
    if split != "all":  # sheet photo groups look like "3001_IMG_3415"
        groups = {photo.group for photo in photos_of(sheet_photos_per_split(), split)}
        all_crops = [c for c in all_crops if f"{c.part}_{c.group}" in groups]
    crops = [c for c in all_crops if c.part in known]
    skipped = {c.name for c in all_crops} - {c.name for c in crops}
    images = [Image.open(c.path).convert("RGB") for c in crops]
    return crops, images, skipped


def sheet_crops(method, split, known):
    """Cut the sheet photos of `split` with bricks_on_sheet.

    Returns (crops, their BrickCrops, skipped folder names).
    """
    import cv2

    from legorec.segment import bricks_on_sheet

    photos_per_split = sheet_photos_per_split()
    every_photo = photos_of(photos_per_split, "all")
    chosen = photos_of(photos_per_split, split)
    skipped = {p.folder for p in chosen if p.part not in known}  # MIXED too
    chosen = sorted((p for p in chosen if p.part in known), key=lambda p: p.group)

    # sessions from all photos, so they are the same as for --source crops
    number = {p.group: int(re.search(r"IMG_(\d+)", p.group)[1]) for p in every_photo}
    session_of = photo_sessions(sorted(set(number.values())))
    crops, inputs, no_markers, empty = [], [], [], []
    for photo in chosen:
        try:
            found = bricks_on_sheet(cv2.imread(photo.path), method)
        except ValueError:
            no_markers.append(photo.group)
            continue
        if not found:
            empty.append(photo.group)
        for k, brick in enumerate(found, start=1):
            session = session_of[number[photo.group]]
            crops.append(
                OwnCrop(
                    f"{photo.path}#{k}", photo.part, photo.folder, photo.group, session
                )
            )
            inputs.append(brick)
    print(f"[data] {method} on {len(chosen)} {split} photos")
    if no_markers:
        print(f"[warn] markers not found, photo left out: {', '.join(no_markers)}")
    if empty:
        print(f"[warn] no brick found (counts as nothing): {', '.join(empty)}")
    if split != "test" and method == "yolo":
        print(
            "[warn] YOLO was trained on train / val photos: finding is too easy there"
        )
    return crops, inputs, skipped


def evaluate(checkpoint, crops, inputs, device, real_scale=None, size_prior=None):
    # ValueError for a size-feature model without real_scale
    classifier = ShapeClassifier(
        checkpoint, device=device, real_scale=real_scale, size_prior=size_prior
    )
    parts = classifier.parts
    y = np.array([parts.index(crop.part) for crop in crops])
    if isinstance(inputs[0], Image.Image):
        scores = classifier.scores(inputs)
    else:
        scores = classifier.crop_scores(inputs)
    m = classification_metrics(scores, y)
    m["top1_k"] = top1_among(scores, y, sorted(set(y)))
    photo_right = group_top1(scores, y, [crop.group for crop in crops])
    m["photo_top1"] = float(np.mean(list(photo_right.values())))
    pred = scores.argmax(axis=1)
    probabilities = softmax(scores)
    return classifier, m, pred, probabilities, y, photo_right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["crops", "yolo", "threshold"],
        default="crops",
        help="hand-checked crops, or the sheet photos cut by bricks_on_sheet",
    )
    parser.add_argument(
        "--data",
        default=str(own_crops_dir()),
        help="--source crops: one subfolder per part (default: data/raw/own_crops)",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train", "all"],
        default=None,
        help="photos of this split of sheets.csv (default: all for crops, "
        "test for yolo / threshold; same split = same photos, comparable)",
    )
    parser.add_argument(
        "--real_scale",
        type=float,
        default=None,
        help=f"for size-feature models (default: {SHEET_REAL_SCALE} for yolo / "
        "threshold, none for crops)",
    )
    parser.add_argument(
        "--size-prior",
        action="store_true",
        help="yolo / threshold: also weigh the measured size in mm "
        "(data/reference/part_sizes.csv, from the train photos)",
    )
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    if args.size_prior and args.source == "crops":
        parser.error("--size-prior needs a sheet source (yolo / threshold)")
    size_prior = SizePrior.load() if args.size_prior else None
    real_scale = args.real_scale
    if real_scale is None and args.source != "crops":
        real_scale = SHEET_REAL_SCALE

    checkpoints = (
        [Path(p) for p in args.models]
        if args.models
        else sorted(
            MODELS_DIR.glob("shape_*.pt")  # shape.pt is a copy of one of them
        )
    )
    known = parts_known_by_all(checkpoints)
    if args.source == "crops":
        crops, inputs, skipped = own_crops(args.data, args.split or "all", known)
    else:
        crops, inputs, skipped = sheet_crops(args.source, args.split or "test", known)
    if not crops:
        raise SystemExit("no crops of a part that every model knows")
    print(
        f"[data] {len(crops)} crops from {len({c.group for c in crops})} photos, "
        f"{len({c.part for c in crops})} parts, sessions: "
        f"{', '.join(sorted({c.session for c in crops}))}"
    )
    print(
        f"[data] not known by every model, skipped: {', '.join(sorted(skipped)) or '-'}\n"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}  # run -> dict
    for checkpoint in checkpoints:
        run = checkpoint.stem.removeprefix("shape_")
        try:
            classifier, m, pred, prob, y, photo_right = evaluate(
                checkpoint, crops, inputs, device, real_scale, size_prior
            )
        except ValueError:
            print(f"[skip] {run}: size-feature model")
            continue
        variant, seed = variant_and_seed(checkpoint)
        results[run] = {
            "variant": variant,
            "seed": seed,
            "m": m,
            "pred": pred,
            "y": y,
            "parts": classifier.parts,
            "names": classifier.names,
            "photo_right": photo_right,
        }
        prefix = "eval_own" if args.source == "crops" else f"eval_own_{args.source}"
        prefix += "_size" if size_prior else ""
        out = processed_dir() / f"{prefix}_{run}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["path", "photo", "session", "part", "pred", "prob"])
            for c, p, pr in zip(crops, pred, prob.max(axis=1)):
                writer.writerow(
                    [
                        c.path,
                        c.group,
                        c.session,
                        c.part,
                        classifier.parts[p],
                        f"{pr:.3f}",
                    ]
                )

    keys = ["top1", "top3", "top1_k", "photo_top1", "macro_f1", "conf"]
    print(
        f"{'model':<10}{'seed':>5}  {'variant':<28}" + "".join(f"{k:>11}" for k in keys)
    )
    for run, r in sorted(
        results.items(), key=lambda kv: (kv[1]["variant"], str(kv[1]["seed"]))
    ):
        print(
            f"{run:<10}{r['seed']!s:>5}  {r['variant']:<28}"
            + "".join(f"{r['m'][k]:>11.3f}" for k in keys)
        )

    by_variant = defaultdict(list)
    for run, r in results.items():
        by_variant[r["variant"]].append(run)
    print("\nPer variant (mean [min, max] over seeds):")
    for variant, runs in sorted(by_variant.items()):
        cells = []
        for k in ("top1", "photo_top1", "macro_f1"):
            v = np.array([results[x]["m"][k] for x in runs])
            cells.append(f"{k} {v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]")
        print(f"  {variant:<28} n={len(runs)}  " + "   ".join(cells))

    print("\nTop-1 per part (crops), mean over seeds of each variant:")
    parts_present = sorted({c.part for c in crops})
    names = next(iter(results.values()))["names"]
    variants = sorted(by_variant)
    print(f"  {'part':<26}{'n':>5}" + "".join(f"{v[:24]:>26}" for v in variants))
    for part in parts_present:
        idx = np.array([c.part == part for c in crops])
        row = []
        for v in variants:
            row.append(
                np.mean(
                    [
                        np.mean(results[x]["pred"][idx] == results[x]["y"][idx])
                        for x in by_variant[v]
                    ]
                )
            )
        label = names.get(part, part)[:25]
        print(f"  {label:<26}{idx.sum():>5}" + "".join(f"{a:>26.2f}" for a in row))

    print("\nTop-1 per session (crops), mean over seeds:")
    for session in sorted({c.session for c in crops}):
        idx = np.array([c.session == session for c in crops])
        row = [
            np.mean(
                [
                    np.mean(results[x]["pred"][idx] == results[x]["y"][idx])
                    for x in by_variant[v]
                ]
            )
            for v in variants
        ]
        print(f"  {session:<26}{idx.sum():>5}" + "".join(f"{a:>26.2f}" for a in row))

    if len(variants) > 1:
        print("\nMcNemar per photo, seed-matched (only A right / only B right, p):")
        for i, a in enumerate(variants):
            for b in variants[i + 1 :]:
                for ra in by_variant[a]:
                    for rb in by_variant[b]:
                        if results[ra]["seed"] != results[rb]["seed"]:
                            continue
                        groups = sorted(results[ra]["photo_right"])
                        pa = np.array([results[ra]["photo_right"][g] for g in groups])
                        pb = np.array([results[rb]["photo_right"][g] for g in groups])
                        oa, ob, p = mcnemar_p(pa, pb)
                        print(
                            f"  seed {results[ra]['seed']}: {a} ({ra}) vs {b} ({rb}): "
                            f"{oa}/{ob} p={p:.3f}"
                        )


if __name__ == "__main__":
    main()
