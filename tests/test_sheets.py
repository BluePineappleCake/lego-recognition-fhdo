import pytest
from PIL import Image

from legorec.data.sheets import SheetPhoto, index_sheet_photos, photo_sessions


def test_sheet_photos_get_the_part_of_their_top_folder(tmp_path):
    for rel in ("3001 brick 2x4/IMG_1.jpg", "MIXED 3009+3710/sub/IMG_2.jpg"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(tmp_path / rel)

    photos = {p.group: p for p in index_sheet_photos(tmp_path)}

    assert photos["3001_IMG_1"].part == "3001"
    assert photos["MIXED_IMG_2"].folder == "MIXED 3009+3710"


def test_a_copy_of_a_photo_in_another_folder_is_skipped(tmp_path):
    for rel in ("3001 brick 2x4/IMG_1.jpg", "lego-real-photos/by_part/3001/IMG_1.jpg"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), "red").save(tmp_path / rel)

    photos = index_sheet_photos(tmp_path)

    assert [p.group for p in photos] == ["3001_IMG_1"]


def test_the_upper_case_part_folder_wins_over_a_lower_case_copy(tmp_path):
    for rel in ("MIXED 3009+3710/IMG_2.jpg", "lego-real-photos/mixed/IMG_2.jpg"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), "blue").save(tmp_path / rel)

    assert [p.group for p in index_sheet_photos(tmp_path)] == ["MIXED_IMG_2"]


def test_a_typo_in_a_folder_name_is_an_error(tmp_path):
    (tmp_path / "3001brick 2x4").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "3001brick 2x4" / "IMG_1.jpg")
    with pytest.raises(ValueError, match="3001brick"):
        index_sheet_photos(tmp_path)


def test_a_jump_in_photo_numbers_starts_a_new_session():
    groups = ["3001_IMG_3415", "3002_IMG_3430", "3001_IMG_8817", "3001_PXL_2026"]
    photos = [SheetPhoto("", g.split("_")[0], "", g) for g in groups]

    sessions = photo_sessions(photos)

    assert sessions["3001_IMG_3415"] == sessions["3002_IMG_3430"] == "IMG_3415-3430"
    assert sessions["3001_IMG_8817"] == "IMG_8817-8817"
    assert sessions["3001_PXL_2026"] == "other"  # another phone's naming
