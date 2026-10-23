# Experiment log

Short record of what we tried, what came out, and what we concluded.
Full curves, configs and confusion matrices are in W&B (team `lego-fhdo`,
project `lego-recognition`). Newest entries at the top.

## 2026-10-23 – Shape classifier v0 on renders

**Setup:** `configs/shape_baseline.yaml` – greyscale crops 96 × 96, small CNN
(4 conv blocks, 1.2 M parameters, from scratch), 30 epochs, AdamW + OneCycle,
label smoothing 0.1, rotation/scale/light augmentation. Split by pose
(`data/splits/renders.csv`): 32,000 / 4,000 / 4,000 images, 80 test images per
class. About 6 s per epoch on an RTX 4080.

| Run (W&B) | Config | Top-1 | Top-3 | Length errors | Other errors |
|---|---|---|---|---|---|
| `wobbly-feather-2` | baseline | 93.7 % | 99.5 % | 115 | 137 |
| `silver-fog-4` | size feature | 94.8 % | 99.6 % | 70 | 138 |

Render test set, seed 0, one run each. "Length errors" = confusions between
the pairs in `legorec.eval.metrics.LENGTH_PAIRS`.

**Findings**

1. Top-3 is almost 100 %: errors are confusions between similar parts.
2. The look-alikes we expected (round tile 4150 vs 27925, roof tiles with
   different slopes, plate/tile/jumper 1×2, brick vs Technic brick) cause
   almost no errors (0–2 each).
3. Almost half of the errors (115 of 252) are the same part type in a different
   length: 1×4 vs 1×3, 2×4 vs 2×3, 1×2 vs 1×1 …
4. Cause: cropping + resizing to 96 × 96 removes the absolute size.
   `scripts/size_check.py` shows the renders keep the true size (median size
   ratios 1.22 / 1.24 / 1.55 vs. 1.24 / 1.30 / 1.58 from the geometry), so the
   render camera distance is constant.
5. Feeding the measured size as an extra input (`size_feature: true`, 5 %
   noise during training) cuts length errors by 39 % (115 → 70); all other
   errors stay the same (137 → 138). The effect appears exactly where the
   hypothesis predicts.
6. The model is underconfident (mean confidence 0.81 at 94.8 % accuracy),
   probably from label smoothing.

**Open**

- Repeat both configs with seeds 1 and 2 before calling the size effect final.
- Size on real photos needs a calibration (`real_scale` = render px/mm ÷
  photo px/mm), e.g. from the grey card with a fixed camera.
- Renders show random 3D poses, real bricks lie flat: the size distribution
  on photos will differ.
- Largest remaining pair: plate 2×2 vs plate 2×2 with 2 studs (13 errors,
  same size) → try `size: 128`.
- Calibrate confidences (temperature scaling on val) before using them to
  flag unsure bricks.

Slides: `docs/presentations/2026-10-23_progress_shape_v0.pptx`
