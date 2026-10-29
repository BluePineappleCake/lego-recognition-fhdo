"""Metrics for the shape classifier. Same functions for renders and real photos."""

from collections import Counter

import numpy as np

# Same part type in a different length / size. Defined after the first error
# analysis (docs/experiments.md); used to check whether the size feature helps
# where it should. Pairs are unordered.
LENGTH_PAIRS = {
    frozenset(p)
    for p in [
        ("3010", "3622"),
        ("3001", "3002"),
        ("3069", "3070"),
        ("3020", "3021"),
        ("3659", "4490"),
        ("3023", "3024"),
        ("3004", "3622"),
        ("3002", "3003"),
        ("3021", "3022"),
        ("3037", "3038"),
        ("3039", "3298"),
        ("3004", "3005"),
        ("3068", "3069"),
        ("3024", "3794"),
        ("3005", "3040"),
        ("3070", "54200"),
        ("3040", "3045"),
    ]
}


def macro_f1(y_true, y_pred, n_classes) -> float:
    """F1 score for every class separately, then the plain average.

    Unlike accuracy, every class counts the same, also small or rare ones.
    F1 of a class = 2 * TP / (2 * TP + FP + FN), with
        TP: images of this class predicted as this class (correct)
        FP: images of other classes predicted as this class
        FN: images of this class predicted as something else
    """
    f1_per_class = []
    for c in range(n_classes):
        true_pos = np.sum((y_pred == c) & (y_true == c))
        false_pos = np.sum((y_pred == c) & (y_true != c))
        false_neg = np.sum((y_pred != c) & (y_true == c))
        if true_pos == 0:
            f1_per_class.append(0.0)
        else:
            f1_per_class.append(2 * true_pos / (2 * true_pos + false_pos + false_neg))
    return float(np.mean(f1_per_class))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Scores -> probabilities per row (subtracting the max avoids overflow)."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
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
    probabilities = _softmax(logits)

    return {
        "top1": float(np.mean(best_guess == y)),
        "top3": float(np.mean((top_3 == y[:, None]).any(axis=1))),
        "macro_f1": macro_f1(y, best_guess, n_classes),
        "conf": float(probabilities.max(axis=1).mean()),
    }


def confused_pairs(y, pred, parts) -> Counter:
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


def error_breakdown(y, pred, parts) -> dict:
    """Split the errors into length/size confusions (LENGTH_PAIRS) and the rest."""
    pairs = confused_pairs(y, pred, parts)
    length_errors = sum(n for pair, n in pairs.items() if pair in LENGTH_PAIRS)
    all_errors = sum(pairs.values())
    return {"errors_length": length_errors, "errors_other": all_errors - length_errors}
