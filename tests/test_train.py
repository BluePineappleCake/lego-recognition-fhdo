import numpy as np
import pytest

torch = pytest.importorskip("torch")

from legorec.paths import REPO_ROOT
from legorec.shape.batches import Batches
from legorec.shape.model import SmallCNN
from legorec.shape.training import TrainConfig, evaluate, overfit_one_batch, train_epoch


def test_config_file_overrides_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("width: 16\nhflip: true\n", encoding="utf-8")
    # None = option not given on the command line, keep the file/default value
    config = TrainConfig.from_yaml(path, seed=3, epochs=None)
    assert (config.width, config.hflip, config.seed) == (16, True, 3)
    assert config.epochs == TrainConfig().epochs


def test_typo_in_config_file_is_an_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("epoch: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="epoch"):
        TrainConfig.from_yaml(path)


def test_removed_settings_are_an_error(tmp_path):
    # old configs (size_feature, contrast, mask, Kaggle) need the tag v1-legacy
    path = tmp_path / "config.yaml"
    path.write_text("size_feature: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size_feature"):
        TrainConfig.from_yaml(path)


def test_all_config_files_in_the_repo_are_valid():
    paths = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert paths
    for path in paths:
        TrainConfig.from_yaml(path)


def _random_batches(n=32, batch_size=8, shuffle=True):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (n, 32, 32), dtype=np.uint8)
    y = rng.integers(0, 3, n)
    return Batches(x, y, "cpu", batch_size, shuffle)


def _optimizer_and_schedule(model, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 3e-3, total_steps=steps)
    return optimizer, scheduler


def test_one_epoch_and_evaluation_run():
    torch.manual_seed(0)
    model = SmallCNN(3, width=8)
    batches = _random_batches()
    optimizer, scheduler = _optimizer_and_schedule(model, len(batches))
    loss_fn = torch.nn.CrossEntropyLoss()

    loss, accuracy = train_epoch(
        model, batches, optimizer, scheduler, loss_fn, TrainConfig()
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
    )
    assert loss < 0.1


def test_photos_are_repeated_and_appended():
    from legorec.shape.training_data import TrainData, combine_train

    renders = (np.zeros((5, 4, 4), np.uint8), [0] * 5)
    photos = (np.full((2, 4, 4), 7, np.uint8), [1, 2])
    sources = {
        "internet_renders": {"train": renders},
        "internet_photos": {"train": photos},
    }
    data = TrainData(sources, ["a", "b", "c"], {}, {})
    x, y = combine_train(data, ["internet_renders", "internet_photos"], repeat=3)

    assert x.shape == (11, 4, 4) and len(y) == 11
    assert y == [0] * 5 + [1, 2] * 3
    assert (x[5:] == 7).all() and (x[:5] == 0).all()


def test_missing_internet_folder_is_skipped(tmp_path):
    from legorec.shape.training_data import load_source

    assert load_source("internet_photos", TrainConfig(), tmp_path) is None
    assert load_source("internet_renders", TrainConfig(), tmp_path) is None


@pytest.mark.parametrize(
    "settings",
    [
        {"train_on": ["kaggle_renders"]},  # removed source
        {"train_on": ["internet_renders", "own_photos"]},  # which parts?
        {"train_on": ["internet_renders"], "own_photo_parts": ["4073"]},
        {"train_on": ["internet_renders"], "select_on": "own_renders"},
    ],
)
def test_settings_that_do_not_fit_together_are_an_error(settings):
    with pytest.raises(ValueError):
        TrainConfig(**settings).check()


def test_part_numbers_from_yaml_are_text(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "train_on: [internet_renders, own_photos]\nown_photo_parts: [4073]\n",
        encoding="utf-8",
    )
    assert TrainConfig.from_yaml(path).own_photo_parts == ["4073"]


def _write_renders(root, files):
    from PIL import Image

    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((40, 30), 220, np.uint8)
        image[10:30, 8:22] = 60
        Image.fromarray(image).save(path)


def _no_yolo(monkeypatch):
    """Training images get their mask from YOLO; in tests nothing is found, so
    every image takes the brick_mask route (no model file needed)."""
    from legorec.segment import bricks, single

    monkeypatch.setattr(bricks, "brick_model_id", lambda model=None: "test")
    monkeypatch.setattr(
        single, "single_brick_masks", lambda images, *a, **k: [None] * len(images)
    )


def test_classes_come_from_the_training_sources(tmp_path, monkeypatch):
    from legorec.data import renders
    from legorec.shape.training_data import load_training_data

    _no_yolo(monkeypatch)
    monkeypatch.setenv("LEGOREC_PROCESSED", str(tmp_path / "cache"))
    monkeypatch.setattr(
        renders,
        "SPLIT_FILES",
        {s: tmp_path / f"{s}.csv" for s in ("internet_renders", "own_renders")},
    )
    internet = tmp_path / "internet"
    own = tmp_path / "own"
    _write_renders(
        internet / "renders",
        [f"3001 brick 2x4/3001_Red_0_{i}.jpeg" for i in range(20)]
        + [f"3005 brick 1x1/3005_Red_0_{i}.jpeg" for i in range(20)],
    )
    _write_renders(
        own, [f"4073 round plate 1x1/4073_Red_0_{i}.jpeg" for i in range(20)]
    )
    config = TrainConfig(train_on=["internet_renders", "own_renders"])
    roots = {
        "internet_renders": internet,
        "internet_photos": internet,
        "own_renders": own,
    }
    data = load_training_data(config, roots)

    assert data.parts == ["3001", "3005", "4073"]
    assert set(data.sources) == {"internet_renders", "own_renders"}  # no photos/
    n = sum(len(split[1]) for split in data.sources["own_renders"].values())
    assert n == 20 and set(data.sources["own_renders"]["train"][1]) == {2}
    assert (tmp_path / "own_renders.csv").exists()
    assert list((tmp_path / "cache").glob("own_renders_train_*_yolotest.npz"))


def test_deleting_a_render_builds_a_new_cache(tmp_path, monkeypatch):
    from legorec.data import renders
    from legorec.shape.training_data import load_source

    _no_yolo(monkeypatch)
    monkeypatch.setenv("LEGOREC_PROCESSED", str(tmp_path / "cache"))
    monkeypatch.setattr(renders, "SPLIT_FILES", {"own_renders": tmp_path / "own.csv"})
    own = tmp_path / "own"
    _write_renders(own, [f"3001 brick 2x4/3001_Red_0_{i}.jpeg" for i in range(20)])
    config = TrainConfig(train_on=["own_renders"])
    load_source("own_renders", config, own)

    # the split file still lists it: only the file list tells the caches apart
    next((own / "3001 brick 2x4").iterdir()).unlink()
    data = load_source("own_renders", config, own)

    assert all(len(x) == len(parts) for x, parts in data.splits.values())
    assert sum(len(parts) for _, parts in data.splits.values()) == 19
