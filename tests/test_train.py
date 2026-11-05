import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legorec.paths import REPO_ROOT
from legorec.shape.loaders import Batches
from legorec.shape.model import SizeNorm, SmallCNN
from legorec.shape.train import TrainConfig, evaluate, overfit_one_batch, train_epoch


def test_config_file_overrides_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("width: 16\nsize_feature: true\n", encoding="utf-8")
    # None = option not given on the command line, keep the file/default value
    config = TrainConfig.from_yaml(path, seed=3, epochs=None)
    assert (config.width, config.size_feature, config.seed) == (16, True, 3)
    assert config.epochs == TrainConfig().epochs


def test_typo_in_config_file_is_an_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("epoch: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="epoch"):
        TrainConfig.from_yaml(path)


def test_all_config_files_in_the_repo_are_valid():
    paths = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert paths
    for path in paths:
        TrainConfig.from_yaml(path)


def _random_batches(n=32, batch_size=8, shuffle=True):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (n, 32, 32), dtype=np.uint8)
    z = rng.normal(size=n).astype(np.float32)
    y = rng.integers(0, 3, n)
    return Batches(x, z, y, "cpu", batch_size, shuffle)


def _optimizer_and_schedule(model, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 3e-3, total_steps=steps)
    return optimizer, scheduler


def test_one_epoch_and_evaluation_run():
    torch.manual_seed(0)
    model = SmallCNN(3, width=8, size_feature=True)
    batches = _random_batches()
    optimizer, scheduler = _optimizer_and_schedule(model, len(batches))
    config = TrainConfig(size_feature=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss, accuracy = train_epoch(
        model, batches, optimizer, scheduler, loss_fn, config, SizeNorm(0.0, 1.0)
    )
    assert np.isfinite(loss) and 0 <= accuracy <= 1

    scores, labels, val_loss = evaluate(model, _random_batches(shuffle=False))
    assert scores.shape == (32, 3) and labels.shape == (32,) and np.isfinite(val_loss)


def test_model_can_memorise_one_batch():
    torch.manual_seed(0)
    model = SmallCNN(3, width=8)
    one_batch = [next(iter(_random_batches(batch_size=16)))] * 100
    optimizer, scheduler = _optimizer_and_schedule(model, len(one_batch))
    loss = overfit_one_batch(
        model,
        one_batch,
        optimizer,
        scheduler,
        torch.nn.CrossEntropyLoss(),
        TrainConfig(),
        SizeNorm(0.0, 1.0),
    )
    assert loss < 0.1
