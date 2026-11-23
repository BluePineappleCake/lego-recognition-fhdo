import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legorec.shape.augment import augment_batch, normalize
from legorec.shape.model import SmallCNN


def test_forward_shape():
    m = SmallCNN(50).eval()
    assert m(torch.zeros(4, 1, 96, 96)).shape == (4, 50)


def test_augment_keeps_shape_and_fills_corners_with_the_edge_colour():
    torch.manual_seed(0)
    x = normalize(torch.full((8, 1, 96, 96), 200, dtype=torch.uint8))
    y = augment_batch(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    # without the brightness/contrast change, a plain image must stay exactly
    # plain after rotation: the corners get the edge colour, not a fill value
    y = augment_batch(x, light=(1, 1))
    assert torch.allclose(y, x, atol=1e-4)


def test_batches_cover_every_sample_once():
    from legorec.shape.batches import Batches

    x = np.zeros((10, 8, 8), np.uint8)
    seen = torch.cat([y for _, y in Batches(x, np.arange(10), "cpu", 4, True)])
    assert sorted(seen.tolist()) == list(range(10))
    assert len(Batches(x, np.arange(10), "cpu", 4)) == 3


@pytest.mark.parametrize(
    "old_config",
    [
        {"size_feature": True, "contrast": "stretch", "mask": "yolo"},
        {"contrast": "none", "mask": "yolo"},
        {"contrast": "stretch", "mask": "brick"},
        {},  # the oldest checkpoints: no contrast, brick_mask
    ],
)
def test_older_checkpoints_are_refused(tmp_path, old_config):
    from legorec.shape.classifier import ShapeClassifier
    from legorec.shape.model import usable_checkpoints

    path = tmp_path / "old.pt"
    config = {"size": 96, "width": 8, "dropout": 0.3} | old_config
    torch.save(
        {"model": {}, "parts": ["3001"], "names": {}, "cfg": config}, path
    )  # no "format": written by the code before v1-legacy
    with pytest.raises(ValueError, match="v1-legacy"):
        ShapeClassifier(path)
    assert usable_checkpoints([path]) == []


def test_older_checkpoint_of_todays_kind_still_loads(tmp_path):
    from legorec.shape.classifier import ShapeClassifier

    model = SmallCNN(2, width=8)
    path = tmp_path / "ok1tq8ks.pt"
    config = {"size": 96, "width": 8, "dropout": 0.3, "size_feature": False}
    config |= {"contrast": "stretch", "mask": "yolo"}
    torch.save(
        {
            "model": model.state_dict(),
            "parts": ["3001", "4073"],
            "names": {},
            "cfg": config,
            "split_hash": "x",
            "size_norm": {"mu": 0.0, "sd": 1.0},
        },
        path,
    )
    assert ShapeClassifier(path).parts == ["3001", "4073"]


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


def test_size_prior_skips_parts_without_bricks():
    # a part whose train photos gave no brick must not get a row: an empty
    # row is NaN, and one NaN column made every answer that part
    from legorec.shape.size_prior import SizePrior

    prior = SizePrior.fit(
        {"3001": [(32.0, 16.0), (31.5, 15.8)], "3003": [(16.0, 16.0)], "4073": []}
    )
    assert set(prior.sizes) == {"3001", "3003"}

    out = prior.log_likelihood(
        [_rectangle_crop(20, 10)], ["3001", "3003", "4073", "3622"]
    )
    assert np.all(np.isfinite(out))  # also the parts without a row


def test_size_prior_refuses_a_table_with_nan(tmp_path):
    from legorec.shape.size_prior import SizePrior

    table = tmp_path / "part_sizes.csv"
    SizePrior.fit({"3001": [(32.0, 16.0)]}).save(table)
    with open(table, "a", encoding="utf-8") as file:
        file.write("4073,0,nan,nan,nan,nan,nan,nan\n")
    with pytest.raises(ValueError, match="4073"):
        SizePrior.load(table)
