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
    photo_top1    one answer per photo: mean probability over its crops; a
                  photo where YOLO finds no brick counts as wrong

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

    Returns (Bricks, their BrickCrops, skipped folder names, photos where YOLO
    found no brick).
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
        print(f"[warn] no brick found (wrong in photo_top1): {', '.join(empty)}")
    if split != "test":
        print(
            "[warn] YOLO was trained on train / val photos: finding is too easy there"
        )
    return bricks, crops, skipped, empty


def evaluate(checkpoint, bricks, crops, device, size_prior=None, empty=()) -> dict:
    """One model on the bricks: metrics and what it answered for every brick.

    empty: photos where YOLO found no brick; wrong in photo_top1.
    """
    classifier = ShapeClassifier(checkpoint, device=device, size_prior=size_prior)
    parts = classifier.parts
    y = np.array([parts.index(brick.part) for brick in bricks])
    scores = classifier.scores(crops)
    m = classification_metrics(scores, y)
    m["top1_k"] = top1_among(scores, y, sorted(set(y)))
    photo_right = group_top1(scores, y, [brick.group for brick in bricks])
    photo_right |= dict.fromkeys(empty, False)  # the pipeline named nothing there
    m["photo_top1"] = float(np.mean(list(photo_right.values())))
    variant, seed = variant_and_seed(checkpoint)
    return {
        "variant": variant,
        "seed": seed,
        "m": m,
        "pred": scores.argmax(axis=1),
        "prob": softmax(scores).max(axis=1),
        "y": y,
        "parts": parts,
        "names": classifier.names,
        "photo_right": photo_right,
    }


def write_predictions(path: Path, bricks, result) -> None:
    """One line per brick: which part it is and what the model said."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["path", "photo", "session", "part", "pred", "prob"])
        for brick, pred, prob in zip(bricks, result["pred"], result["prob"]):
            part = result["parts"][pred]
            writer.writerow(
                [
                    brick.path,
                    brick.group,
                    brick.session,
                    brick.part,
                    part,
                    f"{prob:.3f}",
                ]
            )


def runs_per_variant(results) -> dict[str, list[str]]:
    by_variant = defaultdict(list)
    for run, r in results.items():
        by_variant[r["variant"]].append(run)
    return dict(sorted(by_variant.items()))


def print_models(results) -> None:
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


def print_per_variant(results) -> None:
    print("\nPer variant (mean [min, max] over seeds):")
    for variant, runs in runs_per_variant(results).items():
        cells = []
        for k in ("top1", "photo_top1", "macro_f1"):
            v = np.array([results[x]["m"][k] for x in runs])
            cells.append(f"{k} {v.mean():.3f} [{v.min():.3f}, {v.max():.3f}]")
        print(f"  {variant:<28} n={len(runs)}  " + "   ".join(cells))


def top1_of_variants(results, selected) -> list[float]:
    """Top-1 on the selected bricks (bool array), mean over the seeds of each
    variant."""
    return [
        np.mean(
            [
                np.mean(results[x]["pred"][selected] == results[x]["y"][selected])
                for x in runs
            ]
        )
        for runs in runs_per_variant(results).values()
    ]


def print_per_part(results, bricks) -> None:
    print("\nTop-1 per part (crops), mean over seeds of each variant:")
    names = next(iter(results.values()))["names"]
    variants = runs_per_variant(results)
    print(f"  {'part':<26}{'n':>5}" + "".join(f"{v[:24]:>26}" for v in variants))
    for part in sorted({b.part for b in bricks}):
        selected = np.array([b.part == part for b in bricks])
        row = top1_of_variants(results, selected)
        label = names.get(part, part)[:25]
        print(f"  {label:<26}{selected.sum():>5}" + "".join(f"{a:>26.2f}" for a in row))


def print_per_session(results, bricks) -> None:
    print("\nTop-1 per session (crops), mean over seeds:")
    for session in sorted({b.session for b in bricks}):
        selected = np.array([b.session == session for b in bricks])
        row = top1_of_variants(results, selected)
        print(
            f"  {session:<26}{selected.sum():>5}" + "".join(f"{a:>26.2f}" for a in row)
        )


def print_mcnemar(results) -> None:
    """Paired test per photo between two variants, for models of the same seed."""
    variants = runs_per_variant(results)
    if len(variants) < 2:
        return
    print("\nMcNemar per photo, seed-matched (only A right / only B right, p):")
    names = list(variants)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for ra in variants[a]:
                for rb in variants[b]:
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


def parse_args():
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
    return parser.parse_args()


def main():
    args = parse_args()
    size_prior = SizePrior.load() if args.size_prior else None
    checkpoints = usable_checkpoints(
        args.models or sorted(MODELS_DIR.glob("shape_*.pt"))  # shape.pt: a copy
    )

    bricks, crops, skipped, empty = sheet_bricks(
        args.split, parts_known_by_all(checkpoints)
    )
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
    prefix = "eval_own_size" if size_prior else "eval_own"
    results = {}  # run id -> result of evaluate()
    for checkpoint in checkpoints:
        run = checkpoint.stem.removeprefix("shape_")
        results[run] = evaluate(checkpoint, bricks, crops, device, size_prior, empty)
        out = processed_dir() / f"{prefix}_{args.split}_{run}.csv"
        write_predictions(out, bricks, results[run])

    print_models(results)
    print_per_variant(results)
    print_per_part(results, bricks)
    print_per_session(results, bricks)
    print_mcnemar(results)


if __name__ == "__main__":
    main()
