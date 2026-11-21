from pathlib import Path

import pytest
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


def test_kaggle_3062_is_3062b(tmp_path):
    # the Kaggle renders show the hollow-stud brick but call it "3062"; the
    # part must match our photos (3062b), the group must stay as in renders.csv
    Image.new("L", (4, 4)).save(tmp_path / "3062 Round Brick 1x1 007L.png")
    [image] = index_renders(tmp_path)
    assert image.part == "3062b"
    assert image.name == "3062b Round Brick 1x1"
    assert image.group == "3062_007"


def test_extending_a_split_keeps_the_old_groups(tmp_path):
    from dataclasses import dataclass

    from legorec.data.renders import load_or_create_split, load_split

    @dataclass(frozen=True)
    class Item:
        part: str
        group: str

    path = tmp_path / "split.csv"
    old = [Item("4073", f"4073_{i}") for i in range(30)]
    _, first_hash = load_or_create_split(old, path)
    before = load_split(path)

    new = old + [Item("4073", f"4073_{i}") for i in range(30, 60)]
    with pytest.raises(ValueError):
        load_or_create_split(new, path)  # fixed datasets: missing groups are an error
    per_split, second_hash = load_or_create_split(new, path, extend=True)
    after = load_split(path)

    assert {g: after[g] for g in before} == before  # nothing moved
    assert len(after) == 60 and first_hash != second_hash
    assert sum(len(items) for items in per_split.values()) == 60
    # the same new groups land in the same split on every computer
    assert load_or_create_split(new, path, extend=True)[1] == second_hash


def test_cut_renders_are_grouped_by_pose_and_checked(tmp_path):
    from legorec.data.cut_renders import index_cut_renders

    folder = tmp_path / "3005 brick 1x1"
    folder.mkdir()
    for name in [
        "3005_Aqua_0_100.jpeg",
        "3005_Red_0_100.jpeg",
        "3005_Light Grey_1_200.jpeg",
    ]:
        Image.new("L", (4, 4)).save(folder / name)
    renders = index_cut_renders(tmp_path)
    assert [r.group for r in renders] == ["3005_100", "3005_200", "3005_100"]
    assert {r.name for r in renders} == {"3005 brick 1x1"}

    Image.new("L", (4, 4)).save(folder / "3062b_Red_0_5.jpeg")  # wrong folder
    with pytest.raises(ValueError, match="another part"):
        index_cut_renders(tmp_path)
    # the internet set keeps mould variants in the folder of the brick
    variant = [
        r for r in index_cut_renders(tmp_path, strict=False) if "3062b" in r.path
    ]
    assert variant[0].part == "3005" and variant[0].group == "3062b_5"
