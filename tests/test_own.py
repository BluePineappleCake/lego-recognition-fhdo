import pytest
from PIL import Image

from legorec.data.own import index_own_crops


def _crop(folder, file_name):
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(folder / file_name)


def test_crops_of_one_photo_share_a_group_and_the_folder_is_the_label(tmp_path):
    _crop(tmp_path / "3001 brick 2x4", "IMG_3415_b01.png")
    _crop(tmp_path / "3001 brick 2x4", "IMG_3415_b02.png")
    _crop(tmp_path / "3009 brick 1x6", "IMG_3480.jpg")
    (tmp_path / "labels.csv").write_text("file\n")  # ignored

    crops = index_own_crops(tmp_path)

    assert len(crops) == 3
    assert {c.part for c in crops} == {"3001", "3009"}
    assert len({c.group for c in crops if c.part == "3001"}) == 1


def test_a_jump_in_photo_numbers_starts_a_new_session(tmp_path):
    for name in ("IMG_3415.jpg", "IMG_3430.jpg", "IMG_8817.jpg"):
        _crop(tmp_path / "3001 brick 2x4", name)

    sessions = {c.group: c.session for c in index_own_crops(tmp_path)}

    assert sessions["IMG_3415"] == sessions["IMG_3430"] == "IMG_3415-3430"
    assert sessions["IMG_8817"] == "IMG_8817-8817"


def test_unknown_file_names_are_not_silently_dropped(tmp_path):
    _crop(tmp_path / "3001 brick 2x4", "foto 1.png")
    with pytest.raises(ValueError):
        index_own_crops(tmp_path)
