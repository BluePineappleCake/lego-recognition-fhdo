"""Evaluate shape models on the internet photos (real photos, not renders).

    python scripts/eval_internet.py                       # all models/shape_*.pt on val
    python scripts/eval_internet.py --models models/shape_56b289he.pt models/shape_hki5v89j.pt
    python scripts/eval_internet.py --split test          # only for final numbers!

Only parts that every evaluated model knows are used (pass --models to
evaluate one model on all of its parts). Reported per model:

    top1 / top3 / macro_f1 / conf   as for the renders (the model may answer
                                    any of its parts)
    top1_k       best guess among the k parts that occur in these photos only
                 (how much of the error is "answered a part that is not here")
    capture_top1 one answer per capture: mean probability over its cameras
    length / other errors           see legorec.eval.metrics

Then a paired McNemar test for every pair of models. Careful: the cameras of
one capture show the same brick, so images are not independent. The test on
captures is the honest one; the one on images is too optimistic.

Predictions are written to $LEGOREC_PROCESSED/eval_internet_<split>_<model>.csv.
Older models that need the Git tag v1-legacy are skipped.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from legorec.data.internet import index_photos, load_or_create_photo_split, only_parts
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
from legorec.segment.single import single_brick_crops
from legorec.shape.classifier import ShapeClassifier
from legorec.shape.model import parts_known_by_all, usable_checkpoints


def evaluate(
    checkpoint, photos, crops, device
) -> tuple[dict, ShapeClassifier, np.ndarray, np.ndarray]:
    """One model on the photos -> (metrics, classifier, labels, scores)."""
    classifier = ShapeClassifier(checkpoint, device=device)
    parts = classifier.parts
    y = np.array([parts.index(photo.part) for photo in photos])
    scores = classifier.scores(crops)  # no size prior: scale unknown
    metrics = classification_metrics(scores, y)

    # best guess among the parts that occur in these photos
    present = {parts.index(photo.part) for photo in photos}
    metrics["top1_k"] = top1_among(scores, y, sorted(present))

    # one answer per capture: mean probability over its cameras
    metrics["capture_right"] = np.array(
        list(group_top1(scores, y, [photo.group for photo in photos]).values())
    )
    metrics["capture_top1"] = float(metrics["capture_right"].mean())
    metrics |= error_breakdown(y, scores.argmax(axis=1), parts)
    return metrics, classifier, y, scores


def write_predictions(path: Path, photos, parts, scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["path", "group", "part", "pred", "prob"])
        pred, prob = scores.argmax(axis=1), softmax(scores).max(axis=1)
        for photo, p, pr in zip(photos, pred, prob):
            writer.writerow(
                [photo.path, photo.group, photo.part, parts[p], f"{pr:.3f}"]
            )


def print_confusions(run, classifier, y, scores) -> None:
    print(f"[{run}] most frequent confusions:")
    names = classifier.names
    for pair, n in confused_pairs(
        y, scores.argmax(axis=1), classifier.parts
    ).most_common(5):
        a, b = sorted(pair)
        print(f"    {names.get(a, a)} <-> {names.get(b, b)}: {n}")
    print()


def print_table(rows) -> None:
    header = (
        f"{'model':<10}{'top1':>7}{'top3':>7}{'top1_k':>9}"
        f"{'capture':>9}{'mF1':>7}{'conf':>7}{'len err':>9}{'other':>7}"
    )
    print(header)
    print("-" * len(header))
    for run, m in rows:
        print(
            f"{run:<10}{m['top1']:>7.1%}{m['top3']:>7.1%}"
            f"{m['top1_k']:>9.1%}{m['capture_top1']:>9.1%}{m['macro_f1']:>7.3f}"
            f"{m['conf']:>7.2f}{m['errors_length']:>9}{m['errors_other']:>7}"
        )


def print_mcnemar(results) -> None:
    """Paired tests between every two models, on images and on captures."""
    runs = list(results)
    if len(runs) < 2:
        return
    print("\nMcNemar (only A right / only B right, p):")
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            img = mcnemar_p(results[a][0], results[b][0])
            cap = mcnemar_p(results[a][1], results[b][1])
            print(
                f"  {a} vs {b}:  images {img[0]}/{img[1]} p={img[2]:.3f}"
                f"   captures {cap[0]}/{cap[1]} p={cap[2]:.3f}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split == "test":
        print("[warn] test split: report these numbers, never tune on them\n")

    photos_per_split, split_hash = load_or_create_photo_split(
        index_photos(args.data or internet_dir())
    )
    checkpoints = usable_checkpoints(
        args.models or sorted(MODELS_DIR.glob("shape_*.pt"))  # shape.pt: a copy
    )
    photos = only_parts(photos_per_split[args.split], parts_known_by_all(checkpoints))
    images = [Image.open(photo.path).convert("RGB") for photo in photos]
    crops, fallbacks = single_brick_crops(images)  # as in training: YOLO mask
    print(f"[data] YOLO found no brick in {fallbacks} photos: brick_mask there")
    groups = sorted({photo.group for photo in photos})
    print(
        f"[data] internet/{args.split} (split {split_hash}): {len(photos)} photos, "
        f"{len(groups)} captures, {len({p.part for p in photos})} parts "
        f"(the ones every model knows)\n"
    )

    results, rows = {}, []  # run -> (per image right?, per capture right?)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for checkpoint in checkpoints:
        run = checkpoint.stem.removeprefix("shape_")
        metrics, classifier, y, scores = evaluate(checkpoint, photos, crops, device)
        out = processed_dir() / f"eval_internet_{args.split}_{run}.csv"
        write_predictions(out, photos, classifier.parts, scores)
        print_confusions(run, classifier, y, scores)
        results[run] = (scores.argmax(axis=1) == y, metrics.pop("capture_right"))
        rows.append((run, metrics))

    print_table(rows)
    print_mcnemar(results)


if __name__ == "__main__":
    main()
