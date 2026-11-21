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
    scores = clf.scores([Image.fromarray(img)])
    assert scores.shape == (1, len(parts))
    assert parts[int(scores.argmax())] == top[0][0][0]  # same model, same answer


def test_crop_scores_match_predict():
    from legorec.shape.model import ShapeClassifier
    from legorec.types import BrickCrop

    parts = ["3001", "3002", "3003"]
    clf = ShapeClassifier.__new__(ShapeClassifier)  # no checkpoint needed
    clf.parts, clf.size, clf.contrast, clf.device = parts, 96, "none", "cpu"
    clf.real_scale, clf.size_norm, clf.size_prior = 1.0, SizeNorm(0.0, 1.0), None
    clf.model = SmallCNN(len(parts), width=8).eval()
    image = np.full((120, 120, 3), 200, np.uint8)
    mask = np.zeros((120, 120), bool)
    image[30:90, 40:80], mask[30:90, 40:80] = 60, True
    crop = BrickCrop(image, mask, (0, 0, 120, 120))
    scores = clf.crop_scores([crop, crop])
    assert scores.shape == (2, len(parts))
    assert parts[int(scores[0].argmax())] == clf.predict([crop])[0][0]


def test_batches_cover_every_sample_once():
    from legorec.shape.loaders import Batches

    x = np.zeros((10, 8, 8), np.uint8)
    z = np.zeros(10, np.float32)
    seen = torch.cat([y for _, _, y in Batches(x, z, np.arange(10), "cpu", 4, True)])
    assert sorted(seen.tolist()) == list(range(10))
    assert len(Batches(x, z, np.arange(10), "cpu", 4)) == 3


def _rectangle_crop(long_mm, short_mm, px_per_mm=8):
    import cv2

    from legorec.types import BrickCrop

    mask = np.zeros((400, 400), np.uint8)
    box = cv2.boxPoints(((200, 200), (long_mm * px_per_mm, short_mm * px_per_mm), 30))
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    return BrickCrop(
        np.zeros((400, 400, 3), np.uint8), mask.astype(bool), (0, 0, 400, 400)
    )


def test_size_prior_prefers_the_part_of_the_right_size(tmp_path):
    from legorec.segment.sheet import PX_PER_MM
    from legorec.shape import size_prior
    from legorec.shape.size_prior import SizePrior, dimensions_px

    assert size_prior.PX_PER_MM == PX_PER_MM  # copied to avoid importing cv2
    # turned by 30 degrees: the rotated rectangle still measures 20 x 10 mm
    long_px, short_px = dimensions_px(_rectangle_crop(20, 10))
    assert abs(long_px / PX_PER_MM - 20) < 0.5 and abs(short_px / PX_PER_MM - 10) < 0.5

    rng = np.random.default_rng(0)
    sizes = {  # 1x2 brick and 2x2 brick: same long side, other short side
        "3004": np.c_[rng.normal(20, 0.8, 30), rng.normal(10, 0.5, 30)],
        "3003": np.c_[rng.normal(20, 0.8, 30), rng.normal(18, 0.8, 30)],
    }
    prior = SizePrior.fit(sizes)
    prior.save(tmp_path / "sizes.csv")
    loaded = SizePrior.load(tmp_path / "sizes.csv")
    assert loaded.sizes["3004"][4] == 30
    crops = [_rectangle_crop(20, 10), _rectangle_crop(20, 18)]
    ll = loaded.log_likelihood(crops, ["3004", "3003", "9999"])
    assert ll[0, 0] > ll[0, 1] and ll[1, 1] > ll[1, 0]
    # unknown part: between a fitting and a not fitting size
    assert ll[0, 0] > ll[0, 2] > ll[0, 1] and ll[1, 1] > ll[1, 2] > ll[1, 0]
