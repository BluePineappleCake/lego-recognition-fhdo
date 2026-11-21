import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legorec.shape import classifier as shape_classifier
from legorec.shape.classifier import ShapeClassifier, load_classifier
from legorec.shape.model import SmallCNN, save_checkpoint
from legorec.types import BrickCrop

PARTS = ["3001", "3002", "3003"]


def _save_tiny_checkpoint(path):
    """An untrained model, saved the same way as by scripts/train_shape.py."""
    model = SmallCNN(len(PARTS), width=8)
    config = {"size": 96, "width": 8, "dropout": 0.3}
    names = {p: f"brick {p}" for p in PARTS}
    save_checkpoint(path, model, PARTS, names, config, "test")
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


def test_predict_returns_part_and_confidence(tmp_path):
    classifier = load_classifier(_save_tiny_checkpoint(tmp_path / "shape.pt"))

    [(part, confidence)] = classifier.predict([_crop_with_two_bricks()])

    assert part in PARTS
    assert isinstance(confidence, float) and 0 < confidence <= 1


def test_top_k_and_scores_agree_with_predict(tmp_path):
    classifier = load_classifier(_save_tiny_checkpoint(tmp_path / "shape.pt"))
    crops = [_crop_with_two_bricks()] * 2

    top = classifier.top_k(crops, k=3)
    scores = classifier.scores(crops)

    assert len(top) == 2 and len(top[0]) == 3
    assert abs(sum(p for _, p in top[0]) - 1) < 1e-5
    assert scores.shape == (2, len(PARTS))
    assert top[0][0] == classifier.predict(crops)[0]
    assert PARTS[int(scores[0].argmax())] == top[0][0][0]


def test_missing_model_gives_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LEGOREC_SHAPE_MODEL", str(tmp_path / "does_not_exist.pt"))
    with pytest.raises(FileNotFoundError, match="LEGOREC_SHAPE_MODEL"):
        load_classifier()


def test_photo_without_bricks_gives_an_empty_list(tmp_path):
    classifier = ShapeClassifier(_save_tiny_checkpoint(tmp_path / "shape.pt"))
    assert classifier.predict([]) == []
    assert classifier.scores([]).shape == (0, len(PARTS))


def test_checkpoint_of_a_real_run_loads_without_pickled_code(tmp_path):
    """train_shape.py saves asdict(TrainConfig); load_checkpoint only accepts
    plain values (weights_only=True), so every setting must be one."""
    from dataclasses import asdict

    from legorec.shape.model import load_checkpoint
    from legorec.shape.training import TrainConfig

    model = SmallCNN(len(PARTS), width=8)
    path = tmp_path / "run.pt"
    names = {p: f"brick {p}" for p in PARTS}
    config = asdict(TrainConfig(width=8))
    save_checkpoint(path, model, PARTS, names, config, "test")
    assert load_checkpoint(path)["cfg"] == config


def test_size_prior_is_on_by_default_and_can_be_switched_off(tmp_path, monkeypatch):
    from legorec.shape import size_prior

    table = tmp_path / "sizes.csv"
    size_prior.SizePrior.fit({"3001": [(36.0, 18.0)] * 5}).save(table)
    monkeypatch.setattr(shape_classifier, "TABLE", table)
    checkpoint = _save_tiny_checkpoint(tmp_path / "shape.pt")
    assert load_classifier(checkpoint, device="cpu").size_prior is not None
    off = load_classifier(checkpoint, device="cpu", size_prior=None)
    assert off.size_prior is None
    monkeypatch.setattr(shape_classifier, "TABLE", tmp_path / "missing.csv")
    assert load_classifier(checkpoint, device="cpu").size_prior is None


def test_pipeline_names_every_brick_it_finds(tmp_path, monkeypatch):
    from legorec import pipeline

    crops = [_crop_with_two_bricks()] * 3
    monkeypatch.setattr(pipeline, "bricks_on_sheet", lambda photo, model=None: crops)
    classifier = ShapeClassifier(_save_tiny_checkpoint(tmp_path / "shape.pt"))

    found = pipeline.find_and_name(np.zeros((10, 10, 3), np.uint8), classifier)

    assert len(found) == 3 and found[0].crop is crops[0]
    assert (found[0].part, found[0].confidence) == classifier.predict(crops)[0]
