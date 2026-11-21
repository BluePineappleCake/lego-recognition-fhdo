import numpy as np
import pytest

from legorec.eval.metrics import (
    LENGTH_PAIRS,
    PART_GEOMETRY,
    classification_metrics,
    error_breakdown,
    group_top1,
    macro_f1,
    mcnemar_p,
    top1_among,
)


def test_metrics_and_error_breakdown():
    parts = ["3001", "3002", "3022", "33909"]
    y = np.array([0, 1, 2, 3])
    pred = np.array([1, 1, 3, 3])  # one length error (3001->3002), one other error
    logits = np.eye(4)[pred] * 5
    m = classification_metrics(logits, y)
    assert m["top1"] == 0.5
    assert error_breakdown(y, pred, parts) == {"errors_length": 1, "errors_other": 1}


def test_macro_f1_by_hand():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 2])
    # class 0: TP 1, FP 0, FN 1 -> 2/3
    # class 1: TP 1, FP 1, FN 1 -> 1/2
    # class 2: TP 0, FP 1, FN 0 -> 0 (predicted but never true: still counts)
    assert macro_f1(y, pred) == pytest.approx((2 / 3 + 1 / 2 + 0) / 3)


def test_macro_f1_ignores_parts_that_are_not_in_the_photos():
    # photos of only 2 of the 50 parts, all recognised correctly
    y = np.array([3, 3, 7, 7])
    assert macro_f1(y, y) == 1.0
    assert classification_metrics(np.eye(50)[y] * 5, y)["macro_f1"] == 1.0


def test_macro_f1_same_as_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 20, 500)  # 20 of the 50 classes in the data
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, 50, 500))
    expected = sklearn_metrics.f1_score(y, pred, average="macro")
    assert macro_f1(y, pred) == pytest.approx(expected)


def test_one_answer_per_group_from_the_mean_probability():
    # group "a": one image clearly class 0, one weakly class 1 -> mean says 0
    logits = np.array([[5.0, 0.0], [0.0, 0.5], [0.0, 3.0]])
    y = np.array([0, 0, 1])
    assert group_top1(logits, y, ["a", "a", "b"]) == {"a": True, "b": True}


def test_top1_among_ignores_classes_that_are_not_there():
    logits = np.array([[0.0, 1.0, 9.0]])  # best guess: class 2, not in the set
    assert top1_among(logits, np.array([1]), [0, 1]) == 1.0


def test_mcnemar_counts_only_the_disagreements():
    a = np.array([True] * 10 + [False] * 5)
    b = np.array([False] * 10 + [False] * 5)
    only_a, only_b, p = mcnemar_p(a, b)
    assert (only_a, only_b) == (10, 0)
    assert p == pytest.approx(2 / 2**10)
    assert mcnemar_p(a, a)[2] == 1.0


def test_length_pairs_come_from_the_geometry():
    assert frozenset(("3004", "3010")) in LENGTH_PAIRS  # brick 1x2 / 1x4
    assert frozenset(("3023", "3623")) in LENGTH_PAIRS  # plate 1x2 / 1x3
    assert frozenset(("3004", "3003")) not in LENGTH_PAIRS  # 1x2 / 2x2: width
    assert frozenset(("3005", "3040")) not in LENGTH_PAIRS  # brick / slope
    for pair in LENGTH_PAIRS:
        (family_a, width_a, _), (family_b, width_b, _) = (
            PART_GEOMETRY[part] for part in pair
        )
        assert family_a == family_b and width_a == width_b
