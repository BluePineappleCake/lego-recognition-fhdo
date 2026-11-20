"""Evaluate shape models on our own photos (single-brick crops).

    python scripts/eval_own.py --data path/to/photos_own_cutout
    python scripts/eval_own.py --data ... --models models/shape_rydhagpu.pt

Only crops of the render classes are used (the model cannot answer the other
parts). Crops of one photo are not independent (same part, same moment, same
light), so the honest numbers are per photo:

    top1 / top3 / mF1 / conf   per crop, 50-way, as for the renders
    top1_k        best guess among the k parts that occur in these crops only
    photo_top1    one answer per photo: mean probability over its crops

Models are grouped into variants by their config (tags + contrast); the table
shows mean and range over the seeds of a variant. Then per part and per
session, and an exact McNemar test per photo between seed-matched models of
two variants.

Predictions: $LEGOREC_PROCESSED/eval_own_<model>.csv.
Size-feature models are skipped: the crops have no common scale yet.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_internet import all_scores, mcnemar_p, softmax

from legorec.data.internet import render_parts
from legorec.data.own import index_own_crops
from legorec.eval.metrics import classification_metrics
from legorec.paths import MODELS_DIR, processed_dir
from legorec.shape.model import ShapeClassifier


def variant_and_seed(checkpoint: Path) -> tuple[str, int]:
    cfg = torch.load(checkpoint, map_location="cpu", weights_only=False)["cfg"]
    tags = "+".join(t for t in cfg.get("tags") or [] if t != "contrast")
    return f"{tags or 'untagged'}/{cfg.get('contrast') or 'none'}", cfg.get("seed")


def evaluate(checkpoint, crops, images, device):
    classifier = ShapeClassifier(checkpoint, device=device)  # ValueError: size feature
    parts = classifier.parts
    y = np.array([parts.index(crop.part) for crop in crops])
    scores = all_scores(classifier, images)
    m = classification_metrics(scores, y)

    present = sorted(set(y))
    masked = np.full_like(scores, -np.inf)
    masked[:, present] = scores[:, present]
    m["top1_k"] = float(np.mean(masked.argmax(axis=1) == y))

    probabilities = softmax(scores)
    rows_of = defaultdict(list)
    for i, crop in enumerate(crops):
        rows_of[crop.group].append(i)
    groups = sorted(rows_of)
    photo_right = np.array(
        [
            probabilities[rows_of[g]].mean(axis=0).argmax() == y[rows_of[g][0]]
            for g in groups
        ]
    )
    m["photo_top1"] = float(photo_right.mean())
    pred = scores.argmax(axis=1)
    return classifier, m, pred, probabilities, y, dict(zip(groups, photo_right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", required=True, help="folder with one subfolder per part"
    )
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    all_crops = index_own_crops(args.data)
    known = render_parts()
    crops = [c for c in all_crops if c.part in known]
    skipped = sorted({c.name for c in all_crops} - {c.name for c in crops})
    images = [Image.open(c.path).convert("RGB") for c in crops]
    print(
        f"[data] {len(crops)} crops from {len({c.group for c in crops})} photos, "
        f"{len({c.part for c in crops})} parts, sessions: "
        f"{', '.join(sorted({c.session for c in crops}))}"
    )
    print(f"[data] not a render class, skipped: {', '.join(skipped) or '-'}\n")

    checkpoints = (
        [Path(p) for p in args.models]
        if args.models
        else sorted(
            MODELS_DIR.glob("shape_*.pt")  # shape.pt is a copy of one of them
        )
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}  # run -> dict
    for checkpoint in checkpoints:
        run = checkpoint.stem.removeprefix("shape_")
        try:
            classifier, m, pred, prob, y, photo_right = evaluate(
                checkpoint, crops, images, device
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
        out = processed_dir() / f"eval_own_{run}.csv"
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
        label = f"{part} {names.get(part, '')}"[:25]
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
