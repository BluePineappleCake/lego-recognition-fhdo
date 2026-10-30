import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legorec.shape import predict as shape_predict
from legorec.shape.model import SmallCNN
from legorec.types import BrickCrop

PARTS = ["3001", "3002", "3003"]


def _save_tiny_checkpoint(path, size_feature=False):
    """An untrained model with the same file format as scripts/train_shape.py."""
    model = SmallCNN(len(PARTS), width=8, size_feature=size_feature)
    torch.save(
        {
            "model": model.state_dict(),
            "parts": PARTS,
            "names": {p: f"brick {p}" for p in PARTS},
            "cfg": {
                "size": 96,
                "width": 8,
                "dropout": 0.3,
                "size_feature": size_feature,
            },
            "size_norm": {"mu": 5.0, "sd": 0.3},
        },
        path,
    )
    return path


def _crop_with_two_bricks():
    """Like a crop from segmentation: the mask marks one brick, a neighbour
    brick is also visible in the cut-out but must be ignored."""
    image = np.full((120, 160, 3), 220, np.uint8)
    image[30:90, 20:100] = (200, 30, 30)  # our brick (red)
    image[10:40, 120:150] = (30, 30, 200)  # neighbour (blue)
    mask = np.zeros((120, 160), bool)
    mask[30:90, 20:100] = True
    return BrickCrop(image=image, mask=mask, bbox=(500, 300, 160, 120))


@pytest.fixture(autouse=True)
def _no_loaded_model():
    """Every test starts without a loaded model."""
    shape_predict._classifier = None
    yield
    shape_predict._classifier = None


def test_predict_returns_part_and_confidence(tmp_path, monkeypatch):
    checkpoint = _save_tiny_checkpoint(tmp_path / "shape.pt")
    monkeypatch.setenv("LEGOREC_SHAPE_MODEL", str(checkpoint))

    part, confidence = shape_predict.predict(_crop_with_two_bricks())

    assert part in PARTS
    assert isinstance(confidence, float) and 0 < confidence <= 1


def test_model_is_loaded_only_once(tmp_path, monkeypatch):
    checkpoint = _save_tiny_checkpoint(tmp_path / "shape.pt")
    monkeypatch.setenv("LEGOREC_SHAPE_MODEL", str(checkpoint))

    shape_predict.predict(_crop_with_two_bricks())
    first = shape_predict._classifier
    shape_predict.predict(_crop_with_two_bricks())
    assert shape_predict._classifier is first


def test_size_model_needs_real_scale(tmp_path):
    checkpoint = _save_tiny_checkpoint(tmp_path / "size.pt", size_feature=True)
    with pytest.raises(ValueError):
        shape_predict.load_model(checkpoint)

    shape_predict.load_model(checkpoint, real_scale=1.0, device="cpu")
    part, _ = shape_predict.predict(_crop_with_two_bricks())
    assert part in PARTS


def test_missing_model_gives_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGOREC_SHAPE_MODEL", str(tmp_path / "does_not_exist.pt"))
    with pytest.raises(FileNotFoundError, match="LEGOREC_SHAPE_MODEL"):
        shape_predict.predict(_crop_with_two_bricks())
