import numpy as np
import pytest
from PIL import Image

from legorec.shape.preprocess import brick_mask, preprocess_crop, preprocess_image
from legorec.types import BrickCrop


def _rect_image(w=120, h=40, size=400, bg=200, fg=60):
    a = np.full((size, size), bg, np.uint8)
    y0, x0 = (size - h) // 2, (size - w) // 2
    a[y0 : y0 + h, x0 : x0 + w] = fg
    return a


def test_output_shape_and_measured_size():
    x, px = preprocess_image(Image.fromarray(_rect_image()), size=96)
    assert x.shape == (96, 96) and x.dtype == np.uint8
    assert px == 120  # longest side of the brick, before resizing


def test_size_is_lost_in_the_image_but_kept_in_px():
    # same aspect ratio, different size -> (almost) identical images, different px
    x1, px1 = preprocess_image(_rect_image(120, 40))
    x2, px2 = preprocess_image(_rect_image(90, 30))
    assert np.abs(x1.astype(int) - x2.astype(int)).mean() < 5
    assert px1 / px2 == 120 / 90


def test_crop_with_mask_matches_plain_background_version():
    a = _rect_image()
    rgb = np.repeat(a[..., None], 3, 2)
    crop = BrickCrop(image=rgb, mask=a < 128, bbox=(0, 0, 400, 400))
    x_mask, px_mask = preprocess_crop(crop)
    x_plain, px_plain = preprocess_image(a)
    assert px_mask == px_plain
    assert np.abs(x_mask.astype(int) - x_plain.astype(int)).max() <= 2


def test_crop_ignores_clutter_outside_the_mask():
    a = _rect_image()
    noisy = a.copy()
    noisy[:30, :30] = 0  # dark clutter in a corner, not part of the brick
    rgb = np.repeat(noisy[..., None], 3, 2)
    _, px = preprocess_crop(BrickCrop(image=rgb, mask=a < 128, bbox=(0, 0, 400, 400)))
    assert px == 120


# --- robust mask (brick_mask) ---------------------------------------------


def test_mask_keeps_exact_bounding_box():
    a = _rect_image()
    ys, xs = np.nonzero(brick_mask(np.repeat(a[..., None], 3, 2)))
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (180, 219, 140, 259)


def test_dark_stripe_at_the_edge_is_ignored():
    # the WhatsApp case: a thin dark line along the bottom edge
    a = _rect_image()
    a[-3:, :] = 30
    _, px = preprocess_image(a)
    assert px == 120


def test_small_speck_is_ignored():
    a = _rect_image()
    a[20:26, 20:26] = 40
    _, px = preprocess_image(a)
    assert px == 120


def test_highlights_inside_the_brick_do_not_split_it():
    a = _rect_image()
    a[195:205, 150:250:20] = 200  # bright spots with background colour
    a[180:220, 199:201] = 200  # a bright line that cuts the brick in two
    _, px = preprocess_image(a)
    assert px == 120


def test_white_brick_on_blue_background():
    rgb = np.zeros((400, 400, 3), np.uint8)
    rgb[:] = (40, 90, 200)  # blue card
    rgb[180:220, 140:260] = (235, 235, 235)  # white brick
    _, px = preprocess_image(rgb)
    assert px == 120


def test_empty_image_falls_back_to_whole_image():
    a = np.full((300, 200), 128, np.uint8)
    x, px = preprocess_image(a)
    assert x.shape == (96, 96) and px == 300


# --- contrast option --------------------------------------------------------


def _dark_red_brick():
    rgb = np.zeros((400, 400, 3), np.uint8)
    rgb[:] = (40, 90, 200)  # blue card
    rgb[180:220, 140:260] = (150, 20, 20)  # dark red brick ...
    for x0 in range(150, 250, 24):  # ... with five lighter 10 x 10 px studs
        rgb[195:205, x0 : x0 + 10] = (200, 60, 60)
    return rgb


def test_contrast_none_is_the_default():
    a = _rect_image()
    x1, px1 = preprocess_image(a)
    x2, px2 = preprocess_image(a, contrast="none")
    assert px1 == px2 and np.array_equal(x1, x2)


def test_stretch_uses_full_range_inside_the_brick():
    x_none, _ = preprocess_image(_dark_red_brick(), size=96)
    x_str, _ = preprocess_image(_dark_red_brick(), size=96, contrast="stretch")
    centre = (slice(40, 56), slice(20, 76))  # inside the brick
    assert np.ptp(x_str[centre]) > 2 * np.ptp(x_none[centre])
    assert x_str[0, 0] == 128  # fixed background value


def test_stretch_same_for_photo_and_segmentation_crop():
    rgb = _dark_red_brick()
    mask = np.zeros(rgb.shape[:2], bool)
    mask[180:220, 140:260] = True
    x_img, px_img = preprocess_image(rgb, contrast="stretch")
    crop = BrickCrop(image=rgb, mask=mask, bbox=(0, 0, 400, 400))
    x_crop, px_crop = preprocess_crop(crop, contrast="stretch")
    assert px_img == px_crop
    assert np.abs(x_img.astype(int) - x_crop.astype(int)).max() <= 2


def test_flat_brick_and_unknown_mode():
    a = _rect_image()  # one grey value only: nothing to stretch, no crash
    x, _ = preprocess_image(a, contrast="stretch")
    assert x.shape == (96, 96)
    with pytest.raises(ValueError):
        preprocess_image(a, contrast="equalise")
