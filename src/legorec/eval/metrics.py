"""Metrics for the shape classifier. Same functions for renders and real photos."""

import math
from collections import Counter, defaultdict

import numpy as np

# Footprint of the parts that come in several lengths: family, width, length
# (in studs, as in the LEGO name; for slopes "2 x 3" = 2 deep, 3 wide).
# Also lists parts that no model knows yet; they simply never occur.
PART_GEOMETRY = {
    "3005": ("brick", 1, 1),
    "3004": ("brick", 1, 2),
    "3622": ("brick", 1, 3),
    "3010": ("brick", 1, 4),
    "3009": ("brick", 1, 6),
    "3003": ("brick", 2, 2),
    "3002": ("brick", 2, 3),
    "3001": ("brick", 2, 4),
    "4490": ("arch", 1, 3),
    "3659": ("arch", 1, 4),
    "3024": ("plate", 1, 1),
    "3023": ("plate", 1, 2),
    "3623": ("plate", 1, 3),
    "3710": ("plate", 1, 4),
    "3666": ("plate", 1, 6),
    "3022": ("plate", 2, 2),
    "3021": ("plate", 2, 3),
    "3020": ("plate", 2, 4),
    "3070": ("tile", 1, 1),
    "3069": ("tile", 1, 2),
    "3040": ("slope 45", 2, 1),
    "3039": ("slope 45", 2, 2),
    "3038": ("slope 45", 2, 3),
    "3037": ("slope 45", 2, 4),
    "4286": ("slope 33", 3, 1),
    "3298": ("slope 33", 3, 2),
    "54200": ("slope 30", 1, 1),
    "85984": ("slope 30", 1, 2),
    "41677": ("liftarm thin", 1, 2),
    "6632": ("liftarm thin", 1, 3),
    "18654": ("liftarm thick", 1, 1),
    "43857": ("liftarm thick", 1, 2),
}

# Pairs of parts that differ only in length: same family and width, different
# length (3004 brick 1x2 / 3010 brick 1x4). Defined from the geometry, not from
# the errors a model makes, so every model is judged on a fixed set.
# Look-alikes of other kinds (3005 brick 1x1 / 3040 slope 2x1) are not length
# errors. Pairs are unordered.
LENGTH_PAIRS = {
    frozenset((a, b))
    for a, (family_a, width_a, length_a) in PART_GEOMETRY.items()
    for b, (family_b, width_b, length_b) in PART_GEOMETRY.items()
    if family_a == family_b and width_a == width_b and length_a != length_b
}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score for every class separately, then the plain average.

    Unlike accuracy, every class counts the same, also small or rare ones.
    F1 of a class = 2 * TP / (2 * TP + FP + FN), with
        TP: images of this class predicted as this class (correct)
        FP: images of other classes predicted as this class
        FN: images of this class predicted as something else

    Only classes that occur in y_true or y_pred are averaged (same as
    sklearn's f1_score with average="macro"). Our photos will not show every
    part of the model: a part that is neither in the photos nor predicted
    must not count as F1 = 0.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1_per_class = []
    for c in np.union1d(y_true, y_pred):
        true_pos = np.sum((y_pred == c) & (y_true == c))
        false_pos = np.sum((y_pred == c) & (y_true != c))
        false_neg = np.sum((y_pred != c) & (y_true == c))
        # never 0 / 0: every class here occurs in y_true or y_pred
        f1_per_class.append(2 * true_pos / (2 * true_pos + false_pos + false_neg))
    return float(np.mean(f1_per_class))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Scores -> probabilities per row (subtracting the max avoids overflow)."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Model scores (N images x C classes) and true labels (N,) ->

    top1:     share of images where the best guess is right
    top3:     share of images where the right class is among the 3 best guesses
    macro_f1: see macro_f1
    conf:     mean probability of the best guess ("how sure the model is")
    """
    n_classes = logits.shape[1]
    ranking = np.argsort(-logits, axis=1)  # classes sorted from best to worst
    best_guess = ranking[:, 0]
    top_3 = ranking[:, : min(3, n_classes)]
    probabilities = softmax(logits)

    return {
        "top1": float(np.mean(best_guess == y)),
        "top3": float(np.mean((top_3 == y[:, None]).any(axis=1))),
        "macro_f1": macro_f1(y, best_guess),
        "conf": float(probabilities.max(axis=1).mean()),
    }


def confused_pairs(y, pred, parts: list[str]) -> Counter:
    """Which parts get mixed up how often?

    Returns {frozenset({part_a, part_b}): number of errors}, counting both
    directions together (3001 as 3002 and 3002 as 3001).
    """
    counts = Counter()
    for true_index, predicted_index in zip(y, pred):
        if true_index != predicted_index:
            pair = frozenset((parts[true_index], parts[predicted_index]))
            counts[pair] += 1
    return counts


def error_breakdown(y, pred, parts: list[str]) -> dict[str, int]:
    """Split the errors into length/size confusions (LENGTH_PAIRS) and the rest."""
    pairs = confused_pairs(y, pred, parts)
    length_errors = sum(n for pair, n in pairs.items() if pair in LENGTH_PAIRS)
    all_errors = sum(pairs.values())
    return {"errors_length": length_errors, "errors_other": all_errors - length_errors}


def top1_among(logits: np.ndarray, y: np.ndarray, classes) -> float:
    """Top-1 if the model may only answer one of `classes` (e.g. the parts
    that occur in a test set): how much of the error is "answered a part that
    is not there at all"?"""
    masked = np.full_like(logits, -np.inf)
    masked[:, list(classes)] = logits[:, list(classes)]
    return float(np.mean(masked.argmax(axis=1) == y))


def group_top1(logits: np.ndarray, y: np.ndarray, groups: list[str]) -> dict[str, bool]:
    """One answer per group (all cameras of a capture, all crops of a photo):
    the mean probability over the group's images. Group -> right?

    Images of one group are not independent; counting per group is the honest
    unit for a significance test (mcnemar_p)."""
    probabilities = softmax(logits)
    rows = defaultdict(list)
    for i, group in enumerate(groups):
        rows[group].append(i)
    return {
        group: bool(probabilities[index].mean(axis=0).argmax() == y[index[0]])
        for group, index in sorted(rows.items())
    }


def mcnemar_p(a_right: np.ndarray, b_right: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired right/wrong answers of models A, B.

    Returns (only A right, only B right, p-value).
    """
    only_a = int(np.sum(a_right & ~b_right))
    only_b = int(np.sum(~a_right & b_right))
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return only_a, only_b, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return only_a, only_b, min(1.0, p)
