import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legorec.shape import predict as shape_predict
from legorec.shape.model import SizeNorm, SmallCNN, save_checkpoint
from legorec.types import BrickCrop

PARTS = ["3001", "3002", "3003"]


def _save_tiny_checkpoint(path, size_feature=False):
    """An untrained model, saved the same way as by scripts/train_shape.py."""
    model = SmallCNN(len(PARTS), width=8, size_feature=size_feature)
    config = {"size": 96, "width": 8, "dropout": 0.3, "size_feature": size_feature}
    names = {p: f"brick {p}" for p in PARTS}
    save_checkpoint(path, model, PARTS, names, config, "test", SizeNorm(5.0, 0.3))
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


def test_photo_without_bricks_gives_an_empty_list(tmp_path):
    from legorec.shape.model import ShapeClassifier

    classifier = ShapeClassifier(_save_tiny_checkpoint(tmp_path / "shape.pt"))
    assert classifier.predict([]) == []
    assert classifier.scores([]).shape == (0, len(PARTS))


def test_checkpoint_of_a_real_run_loads_without_pickled_code(tmp_path):
    """train_shape.py saves asdict(TrainConfig); load_checkpoint only accepts
    plain values (weights_only=True), so every setting must be one."""
    from dataclasses import asdict

    from legorec.shape.model import load_checkpoint
    from legorec.shape.train import TrainConfig

    model = SmallCNN(len(PARTS), width=8)
    path = tmp_path / "run.pt"
    names = {p: f"brick {p}" for p in PARTS}
    config = asdict(TrainConfig(width=8))
    save_checkpoint(path, model, PARTS, names, config, "test", SizeNorm(5.0, 0.3))
    assert load_checkpoint(path)["cfg"] == config


def test_size_prior_is_on_by_default_and_can_be_switched_off(tmp_path, monkeypatch):
    from legorec.shape import size_prior

    table = tmp_path / "sizes.csv"
    size_prior.SizePrior.fit({"3001": [(36.0, 18.0)] * 5}).save(table)
    monkeypatch.setattr(shape_predict, "TABLE", table)
    checkpoint = _save_tiny_checkpoint(tmp_path / "shape.pt")
    assert shape_predict.load_model(checkpoint, device="cpu").size_prior is not None
    off = shape_predict.load_model(checkpoint, device="cpu", size_prior=None)
    assert off.size_prior is None
    monkeypatch.setattr(shape_predict, "TABLE", tmp_path / "missing.csv")
    assert shape_predict.load_model(checkpoint, device="cpu").size_prior is None
