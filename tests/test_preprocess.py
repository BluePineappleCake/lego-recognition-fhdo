import numpy as np
from PIL import Image

from legorec.shape.preprocess import preprocess_crop, preprocess_image
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
