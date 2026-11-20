from PIL import Image

from legorec.data.sheets import index_sheet_photos


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
