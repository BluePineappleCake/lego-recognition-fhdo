import numpy as np
from PIL import Image

from legorec.segment.single import _to_rgb, brick_mask
from legorec.shape.preprocess import _to_grey, preprocess_crop
from legorec.types import BrickCrop


def preprocess_image(image, size=96):
    """One brick on a plain background, mask by brick_mask, as the training
    images where YOLO finds nothing."""
    rgb = _to_rgb(image)
    h, w = rgb.shape[:2]
    return preprocess_crop(BrickCrop(rgb, brick_mask(rgb), (0, 0, w, h)), size)


def _rect_image(w=120, h=40, size=400, bg=200, fg=60):
    a = np.full((size, size), bg, np.uint8)
    y0, x0 = (size - h) // 2, (size - w) // 2
    a[y0 : y0 + h, x0 : x0 + w] = fg
    return a


def test_output_shape():
    x = preprocess_image(Image.fromarray(_rect_image()), size=96)
    assert x.shape == (96, 96) and x.dtype == np.uint8


def test_size_is_lost_in_the_image():
    # same aspect ratio, different size -> (almost) identical images; the size
    # in mm comes back through the size prior (legorec.shape.size_prior)
    x1 = preprocess_image(_rect_image(120, 40))
    x2 = preprocess_image(_rect_image(90, 30))
    assert np.abs(x1.astype(int) - x2.astype(int)).mean() < 5


def test_crop_with_mask_matches_plain_background_version():
    a = _rect_image()
    rgb = np.repeat(a[..., None], 3, 2)
    crop = BrickCrop(image=rgb, mask=a < 128, bbox=(0, 0, 400, 400))
    x_mask = preprocess_crop(crop)
    x_plain = preprocess_image(a)
    assert np.abs(x_mask.astype(int) - x_plain.astype(int)).max() <= 2


def test_crop_ignores_clutter_outside_the_mask():
    a = _rect_image()
    noisy = a.copy()
    noisy[:30, :30] = 0  # dark clutter in a corner, not part of the brick
    rgb = np.repeat(noisy[..., None], 3, 2)
    clean = np.repeat(a[..., None], 3, 2)
    x_noisy = preprocess_crop(BrickCrop(image=rgb, mask=a < 128, bbox=(0, 0, 400, 400)))
    x_clean = preprocess_crop(
        BrickCrop(image=clean, mask=a < 128, bbox=(0, 0, 400, 400))
    )
    assert np.array_equal(x_noisy, x_clean)


def test_brick_at_the_image_edge_gives_the_same_crop_as_in_the_middle():
    # the square around a brick in the corner reaches outside the image; that
    # part must be filled with background, not shifted back into the image
    corner = np.full((400, 400), 200, np.uint8)
    corner[:40, :120] = 60  # the brick of _rect_image(), moved to the corner
    brick_corner = preprocess_image(corner) != 128
    brick_middle = preprocess_image(_rect_image()) != 128
    # not pixel-exact: the contrast stretch erodes the mask, but not at the
    # image edge
    overlap = (brick_corner & brick_middle).sum() / (brick_corner | brick_middle).sum()
    assert overlap > 0.95


# --- robust mask (brick_mask) ---------------------------------------------


def test_mask_keeps_exact_bounding_box():
    a = _rect_image()
    ys, xs = np.nonzero(brick_mask(np.repeat(a[..., None], 3, 2)))
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (180, 219, 140, 259)


def test_dark_stripe_at_the_edge_is_ignored():
    # the WhatsApp case: a thin dark line along the bottom edge
    a = _rect_image()
    a[-3:, :] = 30
    assert np.array_equal(preprocess_image(a), preprocess_image(_rect_image()))


def test_small_speck_is_ignored():
    a = _rect_image()
    a[20:26, 20:26] = 40
    assert np.array_equal(preprocess_image(a), preprocess_image(_rect_image()))


def test_highlights_inside_the_brick_do_not_split_it():
    a = _rect_image()
    a[195:205, 150:250:20] = 200  # bright spots with background colour
    a[180:220, 199:201] = 200  # a bright line that cuts the brick in two
    ys, xs = np.nonzero(brick_mask(np.repeat(a[..., None], 3, 2)))
    assert (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) == (120, 40)


def test_white_brick_on_blue_background():
    rgb = np.zeros((400, 400, 3), np.uint8)
    rgb[:] = (40, 90, 200)  # blue card
    rgb[180:220, 140:260] = (235, 235, 235)  # white brick
    ys, xs = np.nonzero(brick_mask(rgb))
    assert (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) == (120, 40)


def test_empty_image_falls_back_to_whole_image():
    a = np.full((300, 200), 128, np.uint8)
    assert preprocess_image(a).shape == (96, 96)


# --- contrast stretch -------------------------------------------------------


def _dark_red_brick():
    rgb = np.zeros((400, 400, 3), np.uint8)
    rgb[:] = (40, 90, 200)  # blue card
    rgb[180:220, 140:260] = (150, 20, 20)  # dark red brick ...
    for x0 in range(150, 250, 24):  # ... with five lighter 10 x 10 px studs
        rgb[195:205, x0 : x0 + 10] = (200, 60, 60)
    return rgb


def test_stretch_uses_full_range_inside_the_brick():
    x = preprocess_image(_dark_red_brick(), size=96)
    centre = (slice(40, 56), slice(20, 76))  # inside the brick
    assert np.ptp(x[centre]) > 200  # dark red alone would be ~40 grey values
    assert x[0, 0] == 128  # fixed background value


def test_stretch_same_for_photo_and_segmentation_crop():
    rgb = _dark_red_brick()
    mask = np.zeros(rgb.shape[:2], bool)
    mask[180:220, 140:260] = True
    x_img = preprocess_image(rgb)
    x_crop = preprocess_crop(BrickCrop(image=rgb, mask=mask, bbox=(0, 0, 400, 400)))
    assert np.abs(x_img.astype(int) - x_crop.astype(int)).max() <= 2


def test_flat_brick_does_not_crash():
    x = preprocess_image(_rect_image())  # one grey value only: nothing to stretch
    assert x.shape == (96, 96)


def test_single_brick_crops_use_yolo_and_fall_back_to_brick_mask(monkeypatch):
    from legorec.segment import single

    image = np.full((60, 80, 3), 200, np.uint8)
    image[20:40, 20:60] = 40  # the brick
    image[40:48, 20:60] = 150  # its shadow: brick_mask takes it, YOLO not
    yolo = np.zeros((60, 80), bool)
    yolo[20:40, 20:60] = True
    monkeypatch.setattr(
        single, "single_brick_masks", lambda images, model=None: [yolo, None]
    )

    crops, fallbacks = single.single_brick_crops([image, image])
    assert fallbacks == 1  # second image: YOLO found nothing
    assert np.array_equal(crops[0].mask, yolo)
    assert np.array_equal(crops[1].mask, brick_mask(image))  # with the shadow
    assert single.single_brick_crops([]) == ([], 0)  # no YOLO needed


def test_empty_mask_gives_the_whole_image():
    crop = BrickCrop(_rect_image(), np.zeros((400, 400), bool), (0, 0, 400, 400))
    assert preprocess_crop(crop).shape == (96, 96)


def test_grey_is_the_same_for_pil_image_and_numpy_array():
    """Renders arrive as PIL images, segmentation crops as numpy arrays."""
    rgb = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert np.array_equal(_to_grey(rgb), _to_grey(Image.fromarray(rgb)))
