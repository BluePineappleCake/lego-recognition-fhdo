"""Evaluate shape models on the internet photos (real photos, not renders).

    python scripts/eval_internet.py                       # all models/shape_*.pt on val
    python scripts/eval_internet.py --models models/shape_56b289he.pt models/shape_hki5v89j.pt
    python scripts/eval_internet.py --split test          # only for final numbers!

Only the 13 parts that are also render classes are used. Reported per model:

    top1 / top3 / macro_f1 / conf   as for the renders (50-way: the model may
                                    answer any of its 50 parts)
    top1_13      best guess among the 13 parts that occur in these photos only
                 (how much of the error is "answered a part that is not here")
    capture_top1 one answer per capture: mean probability over its cameras
    length / other errors           see legorec.eval.metrics

Then a paired McNemar test for every pair of models. Careful: the cameras of
one capture show the same brick, so images are not independent. The test on
captures is the honest one; the one on images is too optimistic.

Predictions are written to $LEGOREC_PROCESSED/eval_internet_<split>_<model>.csv.
Size-feature models are skipped: the photos are tightly cropped, so the brick
size in pixels means nothing here.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from legorec.data.internet import (
    index_photos,
    load_or_create_photo_split,
    only_parts,
    render_parts,
)
from legorec.eval.metrics import (
    classification_metrics,
    confused_pairs,
    error_breakdown,
    group_top1,
    mcnemar_p,
    softmax,
    top1_among,
)
from legorec.paths import MODELS_DIR, internet_dir, processed_dir
from legorec.shape.model import ShapeClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    if args.split == "test":
        print("[warn] test split: report these numbers, never tune on them\n")

    photos_per_split, split_hash = load_or_create_photo_split(
        index_photos(args.data or internet_dir())
    )
    photos = only_parts(photos_per_split[args.split], render_parts())
    images = [Image.open(photo.path).convert("RGB") for photo in photos]
    groups = sorted({photo.group for photo in photos})
    print(
        f"[data] internet/{args.split} (split {split_hash}): {len(photos)} photos, "
        f"{len(groups)} captures, {len({p.part for p in photos})} parts\n"
    )

    checkpoints = args.models or sorted(
        p
        for p in MODELS_DIR.glob("shape_*.pt")  # shape.pt is a copy of one of them
    )

    results = {}  # model -> per image right?, per capture right?
    rows = []
    for checkpoint in map(Path, checkpoints):
        run = checkpoint.stem.removeprefix("shape_")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            classifier = ShapeClassifier(checkpoint, device=device)
        except ValueError:
            print(f"[skip] {run}: size-feature model, brick size meaningless here")
            continue

        parts = classifier.parts
        y = np.array([parts.index(photo.part) for photo in photos])
        scores = classifier.scores(images)
        metrics = classification_metrics(scores, y)

        # best guess among the parts that occur in these photos
        present = {parts.index(photo.part) for photo in photos}
        metrics["top1_13"] = top1_among(scores, y, sorted(present))

        # one answer per capture: mean probability over its cameras
        capture_right = np.array(
            list(group_top1(scores, y, [photo.group for photo in photos]).values())
        )
        metrics["capture_top1"] = float(capture_right.mean())

        pred = scores.argmax(axis=1)
        metrics |= error_breakdown(y, pred, parts)
        results[run] = (pred == y, capture_right)
        rows.append((run, classifier.contrast, metrics))

        out = processed_dir() / f"eval_internet_{args.split}_{run}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["path", "group", "part", "pred", "prob"])
            for photo, p, prob in zip(photos, pred, softmax(scores).max(axis=1)):
                writer.writerow(
                    [photo.path, photo.group, photo.part, parts[p], f"{prob:.3f}"]
                )

        print(f"[{run}] most frequent confusions:")
        for pair, n in confused_pairs(y, pred, parts).most_common(5):
            a, b = sorted(pair)
            print(
                f"    {classifier.names.get(a, a)} <-> {classifier.names.get(b, b)}: {n}"
            )
        print()

    header = (
        f"{'model':<10}{'contrast':<10}{'top1':>7}{'top3':>7}{'top1_13':>9}"
        f"{'capture':>9}{'mF1':>7}{'conf':>7}{'len err':>9}{'other':>7}"
    )
    print(header)
    print("-" * len(header))
    for run, contrast, m in rows:
        print(
            f"{run:<10}{contrast:<10}{m['top1']:>7.1%}{m['top3']:>7.1%}"
            f"{m['top1_13']:>9.1%}{m['capture_top1']:>9.1%}{m['macro_f1']:>7.3f}"
            f"{m['conf']:>7.2f}{m['errors_length']:>9}{m['errors_other']:>7}"
        )

    runs = list(results)
    if len(runs) > 1:
        print("\nMcNemar (only A right / only B right, p):")
        for i, a in enumerate(runs):
            for b in runs[i + 1 :]:
                img = mcnemar_p(results[a][0], results[b][0])
                cap = mcnemar_p(results[a][1], results[b][1])
                print(
                    f"  {a} vs {b}:  images {img[0]}/{img[1]} p={img[2]:.3f}"
                    f"   captures {cap[0]}/{cap[1]} p={cap[2]:.3f}"
                )


if __name__ == "__main__":
    main()
