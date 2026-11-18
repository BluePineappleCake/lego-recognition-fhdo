import pytest
from PIL import Image

from legorec.data.internet import (
    index_photos,
    load_or_create_photo_split,
    only_parts,
    render_parts,
)
from legorec.data.renders import save_split


def _photo(folder, file_name):
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(folder / file_name)


def _dataset(root, part_names=("3001 brick 2x4", "2654 plate 2x2"), captures=20):
    """Every capture photographed by cameras c0..c2."""
    for name in part_names:
        for capture in range(captures):
            for camera in range(3):
                _photo(
                    root / "photos" / name,
                    f"c{camera}_4_ab{capture:02d}_original_{1609710000000 + capture}.jpg",
                )


def test_part_comes_from_the_folder_and_cameras_share_a_group(tmp_path):
    _dataset(tmp_path, part_names=("3001 brick 2x4",), captures=2)
    photos = index_photos(tmp_path)

    assert len(photos) == 6
    assert {p.part for p in photos} == {"3001"}
    assert {p.name for p in photos} == {"3001 brick 2x4"}
    assert len({p.group for p in photos}) == 2
    assert {p.camera for p in photos} == {"0", "1", "2"}


def test_root_may_be_the_photos_folder_itself(tmp_path):
    _dataset(tmp_path, captures=1)
    assert len(index_photos(tmp_path / "photos")) == len(index_photos(tmp_path))


def test_all_cameras_of_a_capture_end_up_in_the_same_split(tmp_path):
    _dataset(tmp_path)
    photos = index_photos(tmp_path)
    photos_per_split, _ = load_or_create_photo_split(photos, tmp_path / "split.csv")

    split_of_group = {}
    for split, photos_in_split in photos_per_split.items():
        for photo in photos_in_split:
            assert split_of_group.setdefault(photo.group, split) == split


def test_split_is_70_15_15_per_part(tmp_path):
    _dataset(tmp_path)
    photos_per_split, _ = load_or_create_photo_split(
        index_photos(tmp_path), tmp_path / "split.csv"
    )
    for part in ("3001", "2654"):
        captures = {
            split: {p.group for p in photos if p.part == part}
            for split, photos in photos_per_split.items()
        }
        assert {s: len(g) for s, g in captures.items()} == {
            "train": 14,
            "val": 3,
            "test": 3,
        }


def test_unknown_file_name_is_an_error(tmp_path):
    _dataset(tmp_path, captures=1)
    _photo(tmp_path / "photos" / "3001 brick 2x4", "IMG_0001.jpg")
    with pytest.raises(ValueError, match="name pattern"):
        index_photos(tmp_path)


def test_only_parts_keeps_the_render_classes(tmp_path):
    _dataset(tmp_path, captures=1)
    render_split = tmp_path / "renders.csv"
    save_split(
        {"3001_000": "train", "3001_001": "test", "3002_000": "val"}, render_split
    )

    known = render_parts(render_split)
    assert known == {"3001", "3002"}
    kept = only_parts(index_photos(tmp_path), known)
    assert {p.part for p in kept} == {"3001"}
