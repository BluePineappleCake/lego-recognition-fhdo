# Experiment log

What we tried, what came out, and what we concluded. So far all entries are
about the shape classifier. Newest entries are at the top.

Full curves, configs and confusion matrices are in W&B (team `lego-fhdo`,
project `lego-recognition`, link in the [README](../README.md#links)).
Unknown term? See [How to read this log](#how-to-read-this-log).

**Contents:** [Where we stand](#where-we-stand) ·
[How to read this log](#how-to-read-this-log) ·
[2026-11-05](#2026-11-05-model-files-first-real-photos-default-model) ·
[2026-10-29](#2026-10-29-robust-brick-mask-contrast-stretch-run-to-run-variation) ·
[2026-10-23](#2026-10-23-shape-classifier-v0-on-renders) ·
[How to add an entry](#how-to-add-an-entry)

## Where we stand

State after the entry of 2026-11-05. Please update this section when you add
an entry.

**What we know**

1. Baseline and size-feature runs reach 93.7–94.8 % top-1 on the render test
   set. Top-3 is almost 100 % (99.5–99.6 % on 2026-10-23): the mistakes are
   confusions between similar parts.
2. Two runs with the same config can differ by ~0.9 pp top-1 and by up to 34
   length errors. One run per variant is not enough to compare variants.
3. In the first baseline run, almost half of the errors (115 of 252) were the
   same part in a different length, because resizing to 96 × 96 removes the
   absolute size. Giving the network the measured size probably helps, but
   how much is still open: -39 % length errors against the worst baseline run,
   about -15 % against the newer ones.
4. Contrast stretch is worse on renders (-2.1 pp) but did better on the first
   real photos. It is the default model for now.
5. On renders the model is underconfident: mean confidence 0.81 at 94.8 %
   accuracy.

**Open**

- [ ] Seeds 0–3 for baseline, size feature and contrast stretch on the
      current code (planned since 2026-10-23).
- [ ] Fill in the missing numbers (tbd): real photos on 2026-11-05, run
      `lydt7czi` on 2026-10-29.
- [ ] Real photos: compare baseline, contrast stretch and size feature on the
      same photos; paired test (McNemar); find out which colours the gain
      comes from.
- [ ] Size feature on real photos: calibrate `real_scale` (e.g. grey card +
      fixed camera). The size distribution will differ: renders show random
      3D poses, real bricks lie flat.
- [ ] Plate 2×2 vs plate 2×2 with 2 studs (13 errors, same size): try
      `size: 128`.
- [ ] Calibrate the confidences (temperature scaling on val) before using them
      to flag unsure bricks.
- [ ] Store `PREPROCESS_VERSION` in the checkpoints and warn on a mismatch.

## How to read this log

- **Data:** unless an entry says otherwise, all numbers are on the **render
  test set**: 4,000 rendered images (80 per part) that the model never sees
  during training. Results on real photos are marked as such.
- **Top-1:** share of test images where the model's first guess is right.
  **pp** = percentage points.
- **Length errors:** test images where the model mixed up two parts that
  differ mainly in length, like 1×3 and 1×4 (the pairs are listed in
  `LENGTH_PAIRS` in [`metrics.py`](../src/legorec/eval/metrics.py)).
  **Other errors:** all other wrong answers. Both are counts out of 4,000
  images. Example: 115 + 137 = 252 errors means 93.7 % top-1.
- **Run:** the W&B run. Newer entries use the run id (`hki5v89j`), older ones
  the random run name (`wobbly-feather-2`). The checkpoint of a run is
  `models/shape_<run id>.pt`.
- **Config:** baseline, size feature or contrast stretch, i.e. the files in
  [`configs/`](../configs/).
- **Code:** which version of the code trained the model, see the table below.
  Preprocessing and training changed in between, so runs from different code
  versions are not perfectly comparable.
- **More terms** (seed, macro-F1, calibration …):
  [Glossary in the README](../README.md#glossary).
- **Corrections:** we never delete old results. When a conclusion turns out to
  be wrong, we add a correction and mark the old text (example: finding 5 of
  2026-10-23).

Code versions in the column *Code*:

| Code | Meaning |
|---|---|
| first script | the first training script, before the code moved into `src/legorec` |
| repo before #13 | shape classifier v0 in the repo ([#11](https://github.com/BluePineappleCake/lego-recognition-fhdo/pull/11)) |
| after #13 | training code simplified, mixed precision removed ([#13](https://github.com/BluePineappleCake/lego-recognition-fhdo/pull/13)) |
| mask2 | robust brick mask, `PREPROCESS_VERSION = "mask2"` ([#14](https://github.com/BluePineappleCake/lego-recognition-fhdo/pull/14)) |
| mask2-pad | current code: the crop square is padded at the image edge instead of shifted ([#18](https://github.com/BluePineappleCake/lego-recognition-fhdo/pull/18)). No run in this log uses it yet. |

## 2026-11-05: Model files, first real photos, default model

### Which checkpoint file belongs to which run?

The run ids were confirmed from the config that is stored in each checkpoint.

| File | Run | Config | Code |
|---|---|---|---|
| `shape_o8jomt0u.pt` | `o8jomt0u` | baseline | repo before #13 |
| `shape_hki5v89j.pt` | `hki5v89j` | baseline | mask2 |
| `shape_56b289he.pt` | `56b289he` | contrast stretch | mask2 |
| `shape_lydt7czi.pt` | `lydt7czi` | size feature | after #13, before the contrast option |

- `lydt7czi` was missing from this log. It is now in the table of 2026-10-29
  (render test results still tbd). Its W&B tag `size-feature` is missing.
- `shape.pt` was a byte-identical copy of `hki5v89j`.

### First real photos

Exploration only: these photos are **not** the frozen test set.

| Model | Top-1 | n photos |
|---|---|---|
| baseline `hki5v89j` | tbd | tbd |
| contrast stretch `56b289he` | tbd | tbd |

**Result:** contrast stretch is better than the baseline on real photos,
although it is worse on renders (-2.1 pp). This fits the hypothesis of
2026-10-29: in greyscale, coloured bricks are too dark.

**Not checked yet:** a paired comparison (McNemar test), which colours the
gain comes from, more than one seed.

### Default model

`models/shape.pt` = `56b289he` (W&B name `valiant-pond-9`, artifact
`shape-cnn:v2` with the alias `default`; contrast stretch, seed 0, mask2).
This is provisional until seeds 0–3 and a real val set confirm it.
Previous default: `hki5v89j` (baseline).

### Open

- Checkpoints do not store `PREPROCESS_VERSION`. So models trained with
  `mask2` silently run with the newer preprocessing since
  [#18](https://github.com/BluePineappleCake/lego-recognition-fhdo/pull/18).
  Store it and warn on a mismatch.
- Retrain baseline and contrast stretch on the current code, seeds 0–3.

## 2026-10-29: Robust brick mask, contrast stretch, run-to-run variation

### Robust brick mask ([#14](https://github.com/BluePineappleCake/lego-recognition-fhdo/pull/14))

**Problem:** the old brick detection counted every pixel whose grey value
differed from the background. A thin dark line at the edge of a catalog image
made the "brick" as wide as the whole image.

**Change** (`preprocess.brick_mask`): brick = the largest region whose colour
differs from the background, measured in Lab colour space, after a
morphological closing (which fills small gaps in the mask).

**Result:**

- Renders: 5,000 images checked with `scripts/check_preprocess.py`. None
  changes the measured size by more than 2 % (max. 1.3 %), so nothing got
  worse on renders.
- The catalog image with the thin dark line at the bottom edge: measured size
  652 px (= image width) before, 305 px now (the same as for the cropped
  image).
- The name of the training cache now contains `PREPROCESS_VERSION` (`mask2`),
  so an old cache is not reused by mistake.

### All render results so far

| Run | Config | Code | Top-1 | Length errors | Other errors |
|---|---|---|---|---|---|
| `wobbly-feather-2` | baseline | first script | 93.70 % | 115 | 137 |
| `silver-fog-4` | size feature | first script | 94.80 % | 70 | 138 |
| `o8jomt0u` | baseline | repo before #13 | 94.60 % | 81 | 135 |
| `lydt7czi` | size feature | after #13 | tbd | tbd | tbd |
| `hki5v89j` | baseline | mask2 | 94.45 % | 83 | 139 |
| `56b289he` | contrast stretch | mask2 | 92.35 % | 113 | 193 |

All seed 0, render test set (4,000 images). `hki5v89j` is confirmed in W&B.
`o8jomt0u` and `56b289he` were first inferred from the order of the export and
confirmed from the checkpoint configs on 2026-11-05. `lydt7czi` was added on
2026-11-05.

> **Correction of finding 5 from 2026-10-23.** Three baseline runs with the
> same config made 115, 81 and 83 length errors: runs of one config differ by
> up to 34 length errors (and ~0.9 pp top-1). The "-39 % length errors"
> compared the size feature against the worst of them. Against the newer
> baselines it is closer to -15 %. The size feature probably helps, but how
> much is open until several seeds per config have been run.

### Contrast stretch (`contrast: stretch`, [#15](https://github.com/BluePineappleCake/lego-recognition-fhdo/pull/15))

**Why:** in greyscale a red brick is almost black.

**Change:** the grey values inside the (slightly shrunk) brick mask are
stretched to 0..255, the background is set to 128. Config:
[`configs/shape_contrast.yaml`](../configs/shape_contrast.yaml).

**Result:**

- Renders: 92.35 % vs 94.45 % for the baseline (-2.1 pp, 84 more errors).
  Both error types go up. A new frequent confusion is tile 1×2 vs roof tile
  1×2 (3069 / 85984). Hypothesis: the network recognises slopes by the shading
  of their faces, and the stretch per crop changes this shading.
- One catalog image of a red brick 2×3 (3002): the baseline predicts 3001
  (52 %, wrong). The contrast model predicts 3002 (22 %, right, but 3001 at
  21 %). All three top candidates of the contrast model are plausible bricks
  or plates; the baseline's are not.

**Decision:** keep it as an option (default `none`) and decide on real
photos. → [2026-11-05](#2026-11-05-model-files-first-real-photos-default-model):
better on the first real photos, now the default model.

### Open

- First real photo set (exploration, not the frozen test set): compare
  baseline, contrast stretch and size feature on the same photos.
  → Partly done on 2026-11-05 (baseline and contrast stretch).
- Seeds 0–3 for `shape_baseline` and `shape_size_feature` on the current code.
  The results go into this table:

| Config | Top-1 (mean ± range) | Length errors (mean ± range) |
|---|---|---|
| baseline | tbd | tbd |
| size feature | tbd | tbd |

**Speed:** since #13 (no mixed precision) one epoch takes ~8.5 s, one run
~4.5 min.

## 2026-10-23: Shape classifier v0 on renders

**Question:** how well does a small CNN, trained from scratch on renders,
recognise the 50 parts? And what kind of mistakes does it make?

**Setup:**

- Config: [`configs/shape_baseline.yaml`](../configs/shape_baseline.yaml)
- Input: greyscale crops, 96 × 96 pixels
- Model: small CNN from scratch, 4 conv blocks, 1.2 M parameters
- Training: 30 epochs, AdamW + OneCycle, label smoothing 0.1, augmentation
  (rotation, scale, light)
- Data: split by pose ([`data/splits/renders.csv`](../data/splits/renders.csv)):
  32,000 / 4,000 / 4,000 images for train / val / test, 80 test images per
  class
- Speed: about 6 s per epoch on an RTX 4080

**Result** (render test set, seed 0, one run each):

| Run (W&B) | Config | Top-1 | Top-3 | Length errors | Other errors |
|---|---|---|---|---|---|
| `wobbly-feather-2` | baseline | 93.7 % | 99.5 % | 115 | 137 |
| `silver-fog-4` | size feature | 94.8 % | 99.6 % | 70 | 138 |

**Findings:**

1. Top-3 is almost 100 %. When the model is wrong, the right part is nearly
   always among its three best guesses: the errors are confusions between
   similar parts.
2. The look-alikes we expected (round tile 4150 vs 27925, roof tiles with
   different slopes, plate/tile/jumper 1×2, brick vs Technic brick) cause
   almost no errors (0–2 each).
3. Almost half of the errors (115 of 252) are the same part type in a
   different length: 1×4 vs 1×3, 2×4 vs 2×3, 1×2 vs 1×1 …
4. Cause: cropping and resizing to 96 × 96 removes the absolute size – a 1×3
   and a 1×4 brick both fill the crop. `scripts/size_check.py` shows that the
   renders keep the true size (median size ratios 1.22 / 1.24 / 1.55 vs.
   1.24 / 1.30 / 1.58 from the geometry), so the render camera distance is
   constant.
5. Feeding the measured size as an extra input (`size_feature: true`, 5 %
   noise during training) cuts length errors by 39 % (115 → 70); all other
   errors stay the same (137 → 138). The effect appears exactly where the
   hypothesis predicts.
   **Overstated, see the
   [correction on 2026-10-29](#2026-10-29-robust-brick-mask-contrast-stretch-run-to-run-variation):**
   single runs of the same config differ by up to 34 length errors.
6. The model is underconfident (mean confidence 0.81 at 94.8 % accuracy),
   probably because of label smoothing.

**Open:**

- Repeat both configs with seeds 1 and 2 before calling the size effect
  final. → Still open, now planned as seeds 0–3 (see 2026-10-29).
- Size on real photos needs a calibration (`real_scale` = render px/mm ÷
  photo px/mm), e.g. from the grey card with a fixed camera.
- Renders show random 3D poses, real bricks lie flat: the size distribution
  on photos will differ.
- Largest remaining pair: plate 2×2 vs plate 2×2 with 2 studs (13 errors,
  same size) → try `size: 128`.
- Calibrate the confidences (temperature scaling on val) before using them to
  flag unsure bricks.

**Slides:**
[`docs/presentations/2026-11-19_progress_shape_v0.pptx`](presentations/2026-11-19_progress_shape_v0.pptx)
– first called `2026-10-23_progress_shape_v0.pptx`; slide 6 was updated on
2026-10-29 with the correction.

## How to add an entry

Put new entries at the top, directly below the code-version table in
[How to read this log](#how-to-read-this-log). Use this template:

```markdown
## YYYY-MM-DD: Short title

**Question:** What did we want to find out, and why?

**Setup:** config, code version (PR), seeds, data (renders or real photos),
W&B run ids

**Result:** a table with the numbers; say on which data

**Conclusion:** what we learned or decided, and how sure we are

**Open:** what comes next
```

Then update [Where we stand](#where-we-stand). Never delete or quietly change
old results: add a correction instead.
