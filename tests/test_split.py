from dataclasses import dataclass

import pytest
from PIL import Image

from legorec.data.splits import load_or_create_split, load_split, make_split, save_split


@dataclass(frozen=True)
class Item:
    part: str
    group: str


def _items(parts=("3001", "3002"), groups=20):
    # two images per group, like two cameras of one capture
    return [Item(p, f"{p}_{i:03d}") for p in parts for i in range(groups) for _ in "LR"]


def test_both_images_of_a_group_end_up_in_the_same_split(tmp_path):
    items_per_split, _ = load_or_create_split(_items(), tmp_path / "split.csv")
    split_of_group = {}
    for split, items in items_per_split.items():
        for item in items:
            assert split_of_group.setdefault(item.group, split) == split
    assert len(split_of_group) == 40


def test_split_is_stratified_and_deterministic():
    items = _items(groups=50)
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


def test_extending_a_split_keeps_the_old_groups(tmp_path):
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


def test_renders_are_grouped_by_pose_and_checked(tmp_path):
    from legorec.data.renders import index_renders

    folder = tmp_path / "3005 brick 1x1"
    folder.mkdir()
    for name in [
        "3005_Aqua_0_100.jpeg",
        "3005_Red_0_100.jpeg",
        "3005_Light Grey_1_200.jpeg",
    ]:
        Image.new("L", (4, 4)).save(folder / name)
    renders = index_renders(tmp_path)
    assert [r.group for r in renders] == ["3005_100", "3005_200", "3005_100"]
    assert {r.name for r in renders} == {"3005 brick 1x1"}

    Image.new("L", (4, 4)).save(folder / "3062b_Red_0_5.jpeg")  # wrong folder
    with pytest.raises(ValueError, match="another part"):
        index_renders(tmp_path)
    # the internet set keeps mould variants in the folder of the brick
    variant = [r for r in index_renders(tmp_path, strict=False) if "3062b" in r.path]
    assert variant[0].part == "3005" and variant[0].group == "3062b_5"
