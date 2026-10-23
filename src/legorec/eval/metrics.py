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
    f1 = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    """logits (N, C), y (N,) -> top1, top3, macro_f1, mean confidence."""
    n_classes = logits.shape[1]
    order = np.argsort(-logits, 1)
    pred = order[:, 0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    prob = e / e.sum(1, keepdims=True)
    return {
        "top1": float(np.mean(pred == y)),
        "top3": float(np.mean((order[:, : min(3, n_classes)] == y[:, None]).any(1))),
        "macro_f1": macro_f1(y, pred, n_classes),
        "conf": float(prob.max(1).mean()),
    }


def confused_pairs(y, pred, parts) -> Counter:
    """Unordered pair of part ids -> number of errors in both directions."""
    c = Counter()
    for t, p in zip(y, pred):
        if t != p:
            c[frozenset((parts[t], parts[p]))] += 1
    return c


def error_breakdown(y, pred, parts) -> dict:
    """Split the errors into length/size confusions and everything else."""
    pairs = confused_pairs(y, pred, parts)
    length = sum(n for k, n in pairs.items() if k in LENGTH_PAIRS)
    return {"errors_length": length, "errors_other": sum(pairs.values()) - length}
