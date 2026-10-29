import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legorec.eval.metrics import classification_metrics, error_breakdown
from legorec.shape.augment import gpu_augment, normalize
from legorec.shape.model import SizeNorm, SmallCNN


@pytest.mark.parametrize("size_feature", [False, True])
def test_forward_shape(size_feature):
    m = SmallCNN(50, size_feature=size_feature).eval()
    out = m(torch.zeros(4, 1, 96, 96), torch.zeros(4))
    assert out.shape == (4, 50)


def test_augment_keeps_shape_and_has_no_black_corners():
    x = normalize(torch.full((8, 1, 96, 96), 200, dtype=torch.uint8))
    y = gpu_augment(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    # a plain light image stays plain after rotation (no black fill)
    assert (y > normalize(torch.tensor(50, dtype=torch.uint8))).all()


def test_size_norm():
    sn = SizeNorm.fit([100, 200, 400])
    z = sn([100, 200, 400])
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-5


def test_metrics_and_error_breakdown():
    parts = ["3001", "3002", "3022", "33909"]
    y = np.array([0, 1, 2, 3])
    pred = np.array([1, 1, 3, 3])  # one length error (3001->3002), one other error
    logits = np.eye(4)[pred] * 5
    m = classification_metrics(logits, y)
    assert m["top1"] == 0.5
    assert error_breakdown(y, pred, parts) == {"errors_length": 1, "errors_other": 1}


def test_shape_classifier_roundtrip(tmp_path):
    from PIL import Image

    from legorec.shape.model import ShapeClassifier

    parts = ["3001", "3002", "3003"]
    model = SmallCNN(len(parts), width=8, size_feature=True)
    ck = tmp_path / "m.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "parts": parts,
            "names": {p: f"brick {p}" for p in parts},
            "cfg": {"size": 96, "width": 8, "dropout": 0.3, "size_feature": True},
            "size_norm": {"mu": 5.0, "sd": 0.3},
        },
        ck,
    )
    with pytest.raises(ValueError):
        ShapeClassifier(ck)  # size model without real_scale
    clf = ShapeClassifier(ck, real_scale=0.5)
    img = np.full((300, 300), 200, np.uint8)
    img[100:200, 120:180] = 60
    top, arrays, pxs = clf.predict_images([Image.fromarray(img)], k=3)
    assert [p for p, _ in top[0]] and len(top[0]) == 3
    assert abs(sum(c for _, c in top[0]) - 1) < 1e-5
    assert arrays[0].shape == (96, 96) and pxs[0] == 100
