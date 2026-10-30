from legorec.data.renders import (
    FILE_NAME_PATTERN,
    RenderImage,
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


def test_both_views_of_a_pose_in_the_same_split():
    items = _items()
    assign = make_split(items)
    # the split is defined per group, so L and R cannot be separated
    assert len(assign) == len({it.group for it in items})


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
