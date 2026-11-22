import importlib.util

import pytest

from legorec.paths import REPO_ROOT


def _export_yolo():
    spec = importlib.util.spec_from_file_location(
        "export_yolo", REPO_ROOT / "scripts" / "export_yolo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_an_old_export_is_deleted(tmp_path):
    export_yolo = _export_yolo()
    photos = tmp_path / "photos"
    (photos / "3001 brick 2x4").mkdir(parents=True)
    with pytest.raises(SystemExit):
        export_yolo.empty_export(photos)  # a wrong --out
    assert (photos / "3001 brick 2x4").exists()

    old = tmp_path / "export"
    export_yolo.empty_export(old)
    export_yolo.write_data_yaml(old)
    (old / "images" / "train" / "stale.jpg").write_bytes(b"")
    export_yolo.empty_export(old)
    assert not (old / "images" / "train" / "stale.jpg").exists()
