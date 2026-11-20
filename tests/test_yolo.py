import numpy as np
import pytest
from PIL import Image

from legorec.segment.yolo import (
    label_line,
    mask_to_polygon,
    read_label_file,
    select_photos,
)


def test_a_rectangle_becomes_a_four_point_polygon(tmp_path):
    mask = np.zeros((100, 200), bool)
    mask[10:30, 50:150] = True

    polygon = mask_to_polygon(mask)
    line = label_line(polygon, width=200, height=100)
    (tmp_path / "a.txt").write_text(line + "\n")
    (read,) = read_label_file(tmp_path / "a.txt")

    assert line.startswith("0 ")
    assert len(polygon) == 4
    assert read.min(axis=0) == pytest.approx([50 / 200, 10 / 100])
    assert read.max(axis=0) == pytest.approx([149 / 200, 29 / 100])


def test_an_empty_mask_has_no_polygon():
    assert mask_to_polygon(np.zeros((10, 10), bool)) is None


def test_select_photos_reads_review_csv_or_takes_all(tmp_path):
    for split, name in (("train", "a"), ("val", "b")):
        (tmp_path / "images" / split).mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(tmp_path / "images" / split / f"{name}.jpg")
    (tmp_path / "review.csv").write_text("photo,split\nb,val\n")

    assert list(select_photos(tmp_path)) == ["b"]
    assert sorted(select_photos(tmp_path, [])) == ["a", "b"]
    with pytest.raises(ValueError):
        select_photos(tmp_path, ["c"])
