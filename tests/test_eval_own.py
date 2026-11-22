import importlib.util

import numpy as np
import pytest

pytest.importorskip("torch")

from legorec.paths import REPO_ROOT
from legorec.shape.model import SmallCNN, save_checkpoint
from legorec.types import BrickCrop


def _eval_own():
    spec = importlib.util.spec_from_file_location(
        "eval_own", REPO_ROOT / "scripts" / "eval_own.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_photo_without_bricks_counts_as_wrong(tmp_path):
    eval_own = _eval_own()
    checkpoint = tmp_path / "shape_test.pt"
    config = {"size": 96, "width": 8, "dropout": 0.3}
    save_checkpoint(checkpoint, SmallCNN(1, width=8), ["3001"], {}, config, "test")
    mask = np.zeros((60, 60), bool)
    mask[10:50, 15:45] = True
    crop = BrickCrop(np.full((60, 60, 3), 200, np.uint8), mask, (0, 0, 60, 60))
    brick = eval_own.Brick("a.jpg#1", "3001", "3001 brick 2x4", "3001_a", "s")

    # one part only: the photo with a brick is always right
    result = eval_own.evaluate(checkpoint, [brick], [crop], "cpu", empty=["3001_b"])

    assert result["photo_right"] == {"3001_a": True, "3001_b": False}
    assert result["m"]["photo_top1"] == 0.5
