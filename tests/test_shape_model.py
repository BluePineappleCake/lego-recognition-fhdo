import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legorec.shape.augment import augment_batch, normalize
from legorec.shape.model import SizeNorm, SmallCNN


@pytest.mark.parametrize("size_feature", [False, True])
def test_forward_shape(size_feature):
    m = SmallCNN(50, size_feature=size_feature).eval()
    out = m(torch.zeros(4, 1, 96, 96), torch.zeros(4))
    assert out.shape == (4, 50)


def test_augment_keeps_shape_and_fills_corners_with_the_edge_colour():
    torch.manual_seed(0)
    x = normalize(torch.full((8, 1, 96, 96), 200, dtype=torch.uint8))
    y = augment_batch(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    # without the brightness/contrast change, a plain image must stay exactly
    # plain after rotation: the corners get the edge colour, not a fill value
    y = augment_batch(x, light=(1, 1))
    assert torch.allclose(y, x, atol=1e-4)


def test_size_norm():
    sn = SizeNorm.fit([100, 200, 400])
    z = sn([100, 200, 400])
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-5


def test_shape_classifier_roundtrip(tmp_path):
    from PIL import Image

    from legorec.shape.model import ShapeClassifier, save_checkpoint

    parts = ["3001", "3002", "3003"]
    model = SmallCNN(len(parts), width=8, size_feature=True)
    ck = tmp_path / "m.pt"
    config = {"size": 96, "width": 8, "dropout": 0.3, "size_feature": True}
    names = {p: f"brick {p}" for p in parts}
    save_checkpoint(ck, model, parts, names, config, "test", SizeNorm(5.0, 0.3))
    with pytest.raises(ValueError):
        ShapeClassifier(ck)  # size model without real_scale
    clf = ShapeClassifier(ck, real_scale=0.5)
    img = np.full((300, 300), 200, np.uint8)
    img[100:200, 120:180] = 60
    top, arrays, pxs = clf.predict_images([Image.fromarray(img)], k=3)
    assert [p for p, _ in top[0]] and len(top[0]) == 3
    assert abs(sum(c for _, c in top[0]) - 1) < 1e-5
    assert arrays[0].shape == (96, 96) and pxs[0] == 100


def test_batches_cover_every_sample_once():
    from legorec.shape.loaders import Batches

    x = np.zeros((10, 8, 8), np.uint8)
    z = np.zeros(10, np.float32)
    seen = torch.cat([y for _, _, y in Batches(x, z, np.arange(10), "cpu", 4, True)])
    assert sorted(seen.tolist()) == list(range(10))
    assert len(Batches(x, z, np.arange(10), "cpu", 4)) == 3
