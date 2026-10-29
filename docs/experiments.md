# Experiment log

Short record of what we tried, what came out, and what we concluded.
Full curves, configs and confusion matrices are in W&B (team `lego-fhdo`,
project `lego-recognition`). Newest entries at the top.

## 2026-10-29 – Robust brick mask, contrast stretch, run-to-run variation

### Robust brick mask (`preprocess.brick_mask`, PR #14)

Brick = largest region whose Lab colour differs from the background, after a
morphological closing (instead of every pixel that differs in grey value).

- Renders: 5,000 checked with `scripts/check_preprocess.py`, 0 change the
  measured size by more than 2 %, max 1.3 % → no regression.
- Catalog image with a thin dark line at the bottom edge: measured size
  652 px (= image width) before, 305 px now (same as the cropped image).
- The training cache name now contains `PREPROCESS_VERSION` (`mask2`).

### All render results so far (render test set, 4,000 images)

| Run | Config | Code | Top-1 | Length errors | Other errors |
|---|---|---|---|---|---|
| `wobbly-feather-2` | baseline | first script | 93.70 % | 115 | 137 |
| `silver-fog-4` | size feature | first script | 94.80 % | 70 | 138 |
| `o8jomt0u` | baseline | repo before #13 | 94.60 % | 81 | 135 |
| `hki5v89j` | baseline | current (`mask2`) | 94.45 % | 83 | 139 |
| `56b289he` | contrast stretch | current (`mask2`) | 92.35 % | 113 | 193 |

All seed 0. `hki5v89j` is confirmed in W&B; for `o8jomt0u` and `56b289he`
the mapping of the exported confusion matrices to run ids was inferred from
the order of the export - check `render_test/top1` in W&B.

**Correction of 2026-10-23, finding 5.** Three baseline runs with the same
config made 115, 81 and 83 length errors: runs of one config differ by up to
34 errors (and ~0.9 pp top-1). The "-39 % length errors" compared the size
feature against the worst of them. Against the newer baselines it is closer to
-15 %. The size feature probably helps, but the size of the effect is open
until several seeds per config have been run.

### Contrast stretch (`contrast: stretch`)

Grey values inside the (slightly shrunk) brick mask stretched to 0..255,
background set to 128. Motivation: in greyscale a red brick is almost black.

- Renders: 92.35 % vs 94.45 % for the baseline (-2.1 pp, 84 more errors),
  both error categories go up; new frequent confusion: tile 1x2 vs roof tile
  1x2 (3069/85984). Hypothesis: slopes are recognised by the shading of their
  faces, which the per-crop stretch changes.
- One catalog image of a red 3002 brick 2x3: baseline predicts 3001 (52 %,
  wrong), contrast model 3002 (22 %, right, but 3001 at 21 %). All three
  contrast-model candidates are plausible bricks/plates, the baseline's are not.
- Decision: keep as an option (default `none`), decide on real photos.

### Open

- Seeds 0-3 for `shape_baseline` and `shape_size_feature` on the current code:

  | Config | Top-1 (mean ± range) | Length errors (mean ± range) |
  |---|---|---|
  | baseline | tbd | tbd |
  | size feature | tbd | tbd |

- First real photo set (exploration, not the frozen test set): compare
  baseline, contrast and size feature on the same photos.
- Training is ~8.5 s per epoch since #13 (no mixed precision), ~4.5 min per run.

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
   **Overstated, see the correction on 2026-10-29:** single runs of the same
   config differ by up to 34 length errors.
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

Slides: `docs/presentations/2026-10-23_progress_shape_v0.pptx` (slide 6 updated
on 2026-10-29 with the correction)
