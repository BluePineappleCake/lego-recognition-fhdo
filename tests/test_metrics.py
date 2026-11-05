import numpy as np
import pytest

from legorec.eval.metrics import classification_metrics, error_breakdown, macro_f1


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
