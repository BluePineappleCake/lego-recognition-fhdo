"""Evaluate shape models on our own sheet photos: the whole pipeline.

    python scripts/eval_own.py --size-prior            # all models, test photos
    python scripts/eval_own.py --size-prior --models models/shape_ok1tq8ks.pt
    python scripts/eval_own.py --size-prior --split val

The sheet photos of one split (data/raw/own_photos, data/splits/sheets.csv)
are cut by bricks_on_sheet (YOLO), as in the real pipeline. Every brick on a
photo of a part folder is that part, so no labelling is needed (MIXED photos
are left out). Whatever YOLO finds is counted, also a half brick or a
shadow: photo to part number, end to end. Missed bricks do not show up here
(see export_yolo.py). Default --split test: YOLO was trained on train / val
photos.

--size-prior: also weigh the measured size of every brick in mm
(legorec.shape.size_prior). Run once with and once without it.

Only crops of parts that every evaluated model knows are used (pass --models
to evaluate one model on all of its parts). Crops of one photo are not
independent (same part, same moment, same light), so the honest numbers are
per photo:

    top1 / top3 / mF1 / conf   per crop, all parts of the model, as for the renders
    top1_k        best guess among the k parts that occur in these crops only
    photo_top1    one answer per photo: mean probability over its crops

Models are grouped into variants by the tags of their config; the table
shows mean and range over the seeds of a variant. Then per part and per
session, and an exact McNemar test per photo between seed-matched models of
two variants.

Predictions: $LEGOREC_PROCESSED/eval_own[_size]_<split>_<model>.csv.
Older models that need the Git tag v1-legacy are skipped.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from legorec.data.sheets import (
    index_sheet_photos,
    load_or_create_sheet_split,
    photo_sessions,
)
from legorec.eval.metrics import (
    classification_metrics,
    group_top1,
    mcnemar_p,
    softmax,
    top1_among,
)
from legorec.paths import MODELS_DIR, own_photos_dir, processed_dir
from legorec.shape.classifier import ShapeClassifier
from legorec.shape.model import load_checkpoint, parts_known_by_all, usable_checkpoints
from legorec.shape.size_prior import SizePrior


@dataclass(frozen=True)
class Brick:
    """One brick found on a sheet photo, with the part of its photo."""

    path: str  # photo path + "#" + number of the brick on it
    part: str
    folder: str
    group: str  # the photo
    session: str


def variant_and_seed(checkpoint: Path) -> tuple[str, int]:
    cfg = load_checkpoint(checkpoint)["cfg"]
    tags = [t for t in cfg.get("tags") or [] if t not in ("contrast", "yolo-mask")]
    return "+".join(tags) or "untagged", cfg.get("seed")


def sheet_bricks(split, known):
    """Cut the sheet photos of `split` with bricks_on_sheet.

    Returns (Bricks, their BrickCrops, skipped folder names).
    """
    import cv2

    from legorec.segment import bricks_on_sheet

    photos = index_sheet_photos(own_photos_dir())
    photos_per_split = load_or_create_sheet_split(photos)[0]
    chosen = [
        photo
        for name, in_split in photos_per_split.items()
        if split in ("all", name)
        for photo in in_split
    ]
    skipped = {p.folder for p in chosen if p.part not in known}  # MIXED too
    chosen = sorted((p for p in chosen if p.part in known), key=lambda p: p.group)

    session_of = photo_sessions(photos)  # from all photos: the same for every split
    bricks, crops, no_markers, empty = [], [], [], []
    for photo in chosen:
        try:
            found = bricks_on_sheet(cv2.imread(photo.path))
        except ValueError:
            no_markers.append(photo.group)
            continue
        if not found:
            empty.append(photo.group)
        for k, crop in enumerate(found, start=1):
            bricks.append(
                Brick(
                    f"{photo.path}#{k}",
                    photo.part,
                    photo.folder,
                    photo.group,
                    session_of[photo.group],
                )
            )
            crops.append(crop)
    print(f"[data] YOLO on {len(chosen)} {split} photos")
    if no_markers:
        print(f"[warn] markers not found, photo left out: {', '.join(no_markers)}")
    if empty:
        print(f"[warn] no brick found (counts as nothing): {', '.join(empty)}")
    if split != "test":
        print(
            "[warn] YOLO was trained on train / val photos: finding is too easy there"
        )
    return bricks, crops, skipped


def evaluate(checkpoint, bricks, crops, device, size_prior=None):
    classifier = ShapeClassifier(checkpoint, device=device, size_prior=size_prior)
    parts = classifier.parts
    y = np.array([parts.index(brick.part) for brick in bricks])
    scores = classifier.scores(crops)
    m = classification_metrics(scores, y)
    m["top1_k"] = top1_among(scores, y, sorted(set(y)))
    photo_right = group_top1(scores, y, [brick.group for brick in bricks])
    m["photo_top1"] = float(np.mean(list(photo_right.values())))
    pred = scores.argmax(axis=1)
    probabilities = softmax(scores)
    return classifier, m, pred, probabilities, y, photo_right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["test", "val", "train", "all"],
        default="test",
        help="photos of this split of sheets.csv (same split = same photos, comparable)",
    )
    parser.add_argument(
        "--size-prior",
        action="store_true",
        help="also weigh the measured size in mm "
        "(data/reference/part_sizes.csv, from the train photos)",
    )
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    size_prior = SizePrior.load() if args.size_prior else None

    checkpoints = usable_checkpoints(
        args.models or sorted(MODELS_DIR.glob("shape_*.pt"))  # shape.pt: a copy
    )
    known = parts_known_by_all(checkpoints)
    bricks, crops, skipped = sheet_bricks(args.split, known)
    if not bricks:
        raise SystemExit("no bricks of a part that every model knows")
    print(
        f"[data] {len(bricks)} crops from {len({b.group for b in bricks})} photos, "
        f"{len({b.part for b in bricks})} parts, sessions: "
        f"{', '.join(sorted({b.session for b in bricks}))}"
    )
    print(
        f"[data] not known by every model, skipped: {', '.join(sorted(skipped)) or '-'}\n"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}  # run -> dict
    for checkpoint in checkpoints:
        run = checkpoint.stem.removeprefix("shape_")
        classifier, m, pred, prob, y, photo_right = evaluate(
            checkpoint, bricks, crops, device, size_prior
        )
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
        prefix = "eval_own_size" if size_prior else "eval_own"
        out = processed_dir() / f"{prefix}_{args.split}_{run}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["path", "photo", "session", "part", "pred", "prob"])
            for c, p, pr in zip(bricks, pred, prob.max(axis=1)):
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
    parts_present = sorted({b.part for b in bricks})
    names = next(iter(results.values()))["names"]
    variants = sorted(by_variant)
    print(f"  {'part':<26}{'n':>5}" + "".join(f"{v[:24]:>26}" for v in variants))
    for part in parts_present:
        idx = np.array([b.part == part for b in bricks])
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
    for session in sorted({b.session for b in bricks}):
        idx = np.array([b.session == session for b in bricks])
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
