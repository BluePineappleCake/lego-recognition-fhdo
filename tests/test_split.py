from pathlib import Path

from PIL import Image

from legorec.data.renders import (
    FILE_NAME_PATTERN,
    RenderImage,
    index_renders,
    load_or_create_split,
    load_split,
    make_split,
    save_split,
)


def _items(parts=("3001", "3002"), poses=20):
    out = []
    for p in parts:
        for i in range(poses):
            for v in "LR":
                name = f"{p} brick"
                out.append(
                    RenderImage(f"{name} {i:03d}{v}.png", p, name, f"{p}_{i:03d}")
                )
    return out


def test_name_pattern():
    m = FILE_NAME_PATTERN.match("14719 flat tile corner 2x2 000L")
    assert m["name"] == "14719 flat tile corner 2x2"
    assert m["pose"] == "000"
    assert m["camera"] == "L"


def test_both_views_of_a_pose_end_up_in_the_same_split(tmp_path):
    # camera L and R of one pose are near-copies: index_renders must give them
    # the same group, otherwise one could land in train and the other in test
    renders = tmp_path / "renders"
    renders.mkdir()
    for pose in range(20):
        for camera in "LR":
            file = renders / f"3001 brick 2x4 {pose:03d}{camera}.png"
            Image.new("L", (4, 4)).save(file)

    images = index_renders(renders)
    assert len(images) == 40
    assert len({image.group for image in images}) == 20

    images_per_split, _ = load_or_create_split(images, tmp_path / "split.csv")
    split_of_file = {
        Path(image.path).name: split
        for split, images_in_split in images_per_split.items()
        for image in images_in_split
    }
    for pose in range(20):
        name = f"3001 brick 2x4 {pose:03d}"
        assert split_of_file[f"{name}L.png"] == split_of_file[f"{name}R.png"]


def test_split_is_stratified_and_deterministic():
    items = _items(poses=50)
    a, b = make_split(items), make_split(items)
    assert a == b
    for part in ("3001", "3002"):
        counts = {s: 0 for s in ("train", "val", "test")}
        for g, s in a.items():
            if g.startswith(part + "_"):
                counts[s] += 1
        assert counts == {"train": 40, "val": 5, "test": 5}


def test_csv_roundtrip(tmp_path):
    assign = make_split(_items())
    f = tmp_path / "renders.csv"
    save_split(assign, f)
    assert load_split(f) == assign
    assert f.read_bytes().endswith(b"\n") and b"\r" not in f.read_bytes()
