# Experiment log

What we tried, what came out, and what we concluded: the shape classifier
and, since 2026-11-20, finding the bricks (YOLO-seg). Newest entries are at
the top.

Full curves, configs and confusion matrices are in W&B (team `lego-fhdo`,
project `lego-recognition`, link in the [README](../README.md#links)).
Unknown term? See [How to read this log](#how-to-read-this-log).

**Contents:** [Where we stand](#where-we-stand) ·
[How to read this log](#how-to-read-this-log) ·
[2026-11-20](#2026-11-20-finding-bricks-with-yolo-seg-one-class-brick) ·
[2026-11-19](#2026-11-19-shape-models-on-our-own-photos) ·
[2026-11-18 (2)](#2026-11-18-2-training-with-renders--internet-photos) ·
[2026-11-18](#2026-11-18-internet-photos-first-large-real-photo-evaluation) ·
[2026-11-11](#2026-11-11-seeds-0-to-3-for-baseline-and-size-feature) ·
[2026-11-05](#2026-11-05-model-files-first-real-photos-default-model) ·
[2026-10-29](#2026-10-29-robust-brick-mask-contrast-stretch-run-to-run-variation) ·
[2026-10-23](#2026-10-23-shape-classifier-v0-on-renders) ·
[How to add an entry](#how-to-add-an-entry)

## Where we stand

State after the entry of 2026-11-20. Please update this section when you add
an entry.

**What we know**

1. Baseline and size-feature runs reach 93.7–95.5 % top-1 on the render test
   set. Top-3 is almost 100 % (99.5–99.6 % on 2026-10-23): the mistakes are
   confusions between similar parts.
2. Two runs with the same config can differ by ~0.9 pp top-1 and by up to 34
   length errors. One run per variant is not enough to compare variants.
3. In the first baseline run, almost half of the errors (115 of 252) were the
   same part in a different length, because resizing to 96 × 96 removes the
   absolute size. Giving the network the measured size probably helps, but
   how much is still open: -39 % length errors against the worst baseline run,
   about -15 % against the newer ones.
   Seeds 0–3 on the current code
   (2026-11-11): 78.75 → 70.0 length errors on average (about -11 %) and
   +0.5 pp top-1, but the ranges overlap, so the effect is small and not
   final yet.
4. Contrast stretch is worse on renders (-2.1 pp) but much better on real
   photos: 25.8 % vs 2.5 % top-1 on the internet photos (val), significant on
   captures (McNemar 24 / 0, p < 0.001). It stays the default model.
5. On renders the model is underconfident: mean confidence 0.81 at 94.8 %
   accuracy. On real photos it is overconfident: 0.46 at 25.8 %.
6. The render-to-real gap is huge: ~92 % on renders vs 25.8 % on real photos
   for the default model; the greyscale baseline is at chance level (2.5 %,
   chance = 2 %). Part of it is preprocessing (shadows, light bricks on light
   background: about +6 pp with a stricter mask threshold), most of it is not.
7. Adding the internet photos to training (`shape_contrast_internet`) lifts
   internet val from 25.2 % to 93.4 % (3 seeds each) and costs nothing
   measurable on renders (92.2 % vs 92.0 %). Train and val photos come from
   the same camera rig, so this is an upper bound. On our own photos
   (2026-11-19): 49 % top-1 per crop vs 13 % for renders only.
8. The most frequent error on our own photos is a brick vs a plate with the
   same footprint (15–18 % of all crops): from above the height is hard to
   see.
9. Finding the bricks: `find_bricks` with a local background colour no
   longer finds shadows and light gradients (0 of 175 thrown-away crops). A
   YOLO-seg model trained from scratch finds every brick of val and test
   (recall 1.0, test mAP50-95 0.916), but its labels come from `find_bricks`:
   whether it separates touching bricks better is still open.

**Open**

- [ ] Seeds 0–3 for contrast stretch on the current code. Contrast stretch:
      seeds 0–2 done on 2026-11-18 (2).
- [x] Seeds 0–3 for baseline and size feature on the current code
      (2026-11-11).
- [ ] Seeds 0–3 for contrast stretch on the current code.
- [ ] Sync the offline runs of 2026-11-11 to W&B.
- [ ] Fill in the missing numbers (tbd): real photos on 2026-11-05, run
      `lydt7czi` on 2026-10-29.
- [x] Real photos: compare baseline and contrast stretch on the same photos
      with a paired test (McNemar). → 2026-11-18 (internet photos).
- [ ] Find out which colours the contrast-stretch gain comes from.
- [x] Train on renders + `internet/train`, evaluate on `internet/val`.
      → 2026-11-18 (2).
- [x] Evaluate the renders-only and the mixed models on our own photos.
      → 2026-11-19 (all photos, exploration; a frozen test session is
      still missing). The new default model is not set yet.
- [ ] Class list renders ∪ internet (18 parts), see 2026-11-19.
- [ ] YOLO vs `find_bricks` on photos with touching bricks and piles
      (2026-11-20).
- [ ] Shadows: `brick_mask` counts the shadow as brick. Fix it, then check on
      renders that nothing gets worse (`scripts/check_preprocess.py`).
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
  differ only in length, like brick 1×3 and 1×4: same family and width,
  different length (`PART_GEOMETRY` / `LENGTH_PAIRS` in
  [`metrics.py`](../src/legorec/eval/metrics.py)). **Changed on 2026-11-21:**
  before, the list was collected from the errors of the first baseline run
  (17 pairs, some of them look-alikes such as brick 1×1 / slope 2×1). All
  length-error numbers in entries up to 2026-11-20 use that old list and are
  not comparable with newer ones.
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
| mask2-pad | current code: the crop square is padded at the image edge instead of shifted ([#18](https://github.com/BluePineappleCake/lego-recognition-fhdo/pull/18)). First runs: 2026-11-11. |

## 2026-11-20: Finding bricks with YOLO-seg (one class "brick")

**Question:** Can a YOLO segmentation model, trained by us on our own sheet
photos, find every brick? It should replace `find_bricks` (colour
threshold), which returns bricks that touch as one blob. YOLO only finds
bricks; which part a brick is stays the job of the shape CNN.

**Setup:**

- Photos: our 246 sheet photos in `data/raw/photos/` (the copy in
  `lego-real-photos/` is skipped as duplicates), warped to the brick area at
  8 px/mm (1080 × 1240 px). Split by photo, per part 70 / 15 / 15
  ([`data/splits/sheets.csv`](../data/splits/sheets.csv), fingerprint
  `3325911c`): 162 / 28 / 56 photos with 515 / 90 / 177 bricks.
- Labels: drafts from `find_bricks` with the new local background (next
  section), written by [`scripts/export_yolo.py`](../scripts/export_yolo.py).
  8 photos corrected by hand in CVAT
  ([`scripts/cvat_labels.py`](../scripts/cvat_labels.py), stored in
  `data/labels/sheets/`): 2 with two touching 2×4 bricks, 3 with a white
  brick on white paper, 3 with a dark brown brick on altrosa paper.
- Model: `yolo26n-seg.yaml` **from scratch, no COCO weights**, Ultralytics
  8.4.216, `imgsz=1024`, 100 epochs, everything else default. 8 min on the
  RTX 4080. Output: `runs/segment/train/weights/best.pt`.

**find_bricks: one background colour vs local background.** The first crops
(2026-11-16) compared every pixel with one colour, the median of the area's
border. Light gradients, dark corners and paper texture became "bricks".
Now the background colour is the median in a 41 × 41 mm window around every
pixel (`background="local"`, the default; the old method is still there as
`background="border"` and gives exactly the old crops). Checked against our
hand-checked crops, all 246 photos:

| | border (old) | local (new) |
|---|---|---|
| crops we had thrown away (shadows, gradients) still found | 175 of 175 | 0 of 175 |
| kept bricks found | – | 594 of 604 |
| mixed photos with the right number of bricks (`labels_mixed.csv`) | 5 of 7 | 7 of 7 |
| photos to check by hand (`review.csv`) | 111 | 16 (7 false alarms) |

The 10 "missed" kept bricks were merged with a shadow blob in the old crops;
the local method finds them on their own. Still weak: bricks that touch
(one mask), a white brick on white paper (only its shadow is found).

**Result** (`best.pt`, masks; boxes are almost the same):

| Split | Photos | Bricks | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| val | 28 | 90 | 0.998 | 1.000 | 0.995 | 0.924 |
| test | 56 | 177 | 0.993 | 1.000 | 0.995 | 0.916 |

Speed: about 5 ms per photo. The test split was evaluated once
(2026-11-20) and stays frozen until the model is final.

**Conclusion:**

- YOLO finds separated bricks as reliably as `find_bricks`: every brick of
  val and test, about one false alarm in 177. Val and test agree, so this is
  not tuned to the 28 val photos.
- No pretrained weights are needed for this step.
- In the pipeline (`segment.bricks_on_sheet`, `scripts/find_bricks.py`) a
  mask that lies more than half inside a surer one is dropped (the extra
  piece of a long 1×6 brick). With that, YOLO finds exactly 177 bricks on
  the test split, the same number as `find_bricks` on every test photo.
- **Not shown yet: that YOLO is better than `find_bricks`.** Apart from 8
  photos, all labels (val and test too) come from `find_bricks` itself, and
  touching bricks occur only on the 2 corrected train photos. The numbers
  measure "YOLO does what `find_bricks` does, on separated bricks".

**Open:**

- 10 new sheet photos with touching bricks and small piles, true count by
  hand. Compare the counting errors of YOLO and `find_bricks`. This decides
  whether YOLO replaces `find_bricks` in the pipeline.
- Pipeline: crops for the shape CNN from YOLO masks; check that they look
  like the crops of `find_bricks` (both include the shadow).

## 2026-11-19: Shape models on our own photos

**Question:** How good are the shape models on our own photos, and does
training with the internet photos help there too?

**Setup:**

- Data: [`scripts/eval_own.py`](../scripts/eval_own.py) on
  `photos_own_cutout`: 638 crops from 168 photos (cut with the old
  `find_bricks`, checked by hand), only the 13 parts that are render classes.
  6 of our 19 parts (2654, 3009, 3666, 3710, 4073, 6215) are not render
  classes: the model cannot answer them.
- All photos, no split yet: an exploration, not the final test.
- Crops of one photo are not independent (same part, same light): counted per
  crop and per photo (mean probability over the crops of a photo); McNemar
  per photo.
- Models: all checkpoints in `models/` except size feature.

**Result:**

| Variant | Runs | Top-1 per crop | Top-1 per photo |
|---|---|---|---|
| renders only, baseline | – | 6 % | 7 % |
| renders only, contrast stretch | 4 | 13 % | 17 % |
| renders + internet, contrast stretch | 3 (seeds 0–2) | 49 % | 61 % |

- The 3 seeds of renders + internet are within 0.5 pp of each other;
  renders + internet vs renders only: p < 0.001 for every seed pair
  (McNemar per photo).
- The most frequent errors are a brick and a plate with the same footprint:
  3003 → 3022 (34), 3004 → 3023 (33), 3001 → 3020 (20); 15–18 % of all
  crops. If all of them were fixed, top-1 per crop would be 64–67 %.
- AdaBN (BatchNorm statistics re-estimated on our crops, 2-fold by photo):
  +1–2 pp, within the noise.

**Conclusion:** Training with the internet photos helps on our own photos as
well (13 % → 49 % per crop), but half of the bricks are still wrong. The
largest single error is brick vs plate: on top-down photos the height of a
brick is hard to see.

**Careful:** most parts were photographed in only one session (one phone).
A model fine-tuned on these photos could learn the session instead of the
part; a frozen test set needs a new session with all parts on both phones.

**Open:**

- Class list renders ∪ internet: the internet photos and renders of 2654,
  3009, 3666, 3710 and 6215 are dropped at the moment (`only_parts` in
  `load_internet_data`).
- A/B test: the same bricks from above vs at ~45°, paired per photo; does the
  brick/plate error go down?
- Fine-tuning on our own photos, after a frozen test session exists.

## 2026-11-11: Seeds 0 to 3 for baseline and size feature

**Question:** Does the size feature really reduce length errors, or was the
difference between single runs luck? Open since 2026-10-23.

**Setup:** configs `shape_baseline` and `shape_size_feature`, seeds 0–3,
code mask2-pad (`main` at 3f46094), Kaggle GPU T4 x2, render test set.
Run by Sami with `WANDB_MODE=offline`: the runs are not in W&B yet, the
numbers are from the `[render test]` line of the script output.
Run ids (seeds 0–3): baseline `trpurok9`, `3o8pk2ul`, `7v5xgsk1`, `6xpj1iq9`;
size feature `xbzi8q4v`, `d1sq4h9f`, `yt6ml3qe`, `k1c17b38`.

**Result** (render test set, 4,000 images)

| Config | Seed | Top-1 | Top-3 | Length errors | Other errors |
|---|---|---|---|---|---|
| baseline | 0 | 94.55 % | 99.48 % | 78 | 140 |
| baseline | 1 | 95.03 % | 99.48 % | 80 | 119 |
| baseline | 2 | 94.40 % | 99.42 % | 75 | 149 |
| baseline | 3 | 94.40 % | 99.50 % | 82 | 142 |
| size feature | 0 | 95.43 % | 99.58 % | 72 | 111 |
| size feature | 1 | 94.50 % | 99.60 % | 83 | 137 |
| size feature | 2 | 95.05 % | 99.52 % | 68 | 130 |
| size feature | 3 | 95.45 % | 99.62 % | 57 | 125 |

Mean (min–max) over seeds 0–3:

| Config | Top-1 | Length errors | Other errors |
|---|---|---|---|
| baseline | 94.59 % (94.40–95.03) | 78.75 (75–82) | 137.5 (119–149) |
| size feature | 95.11 % (94.50–95.45) | 70.0 (57–83) | 125.75 (111–137) |

**Conclusion**

1. The size feature lowers the length errors on average from 78.75 to 70.0
   (about -11 %) and raises top-1 by about 0.5 pp.
2. The ranges overlap: size feature seed 1 has 83 length errors, more than
   every baseline run. The effect is probably real but small; 4 seeds are
   not enough to call it final.
3. On the current code the baseline is stable (75–82 length errors). The 115
   of `wobbly-feather-2` came from the first script.
4. Unexpected: the other errors also go down (137.5 → 125.75). The
   hypothesis was that the size only helps the length pairs.

**Open:** sync the offline runs to W&B; seeds 0–3 for contrast stretch; more
seeds or a paired test if a decision depends on the size feature; why do the
other errors go down?

- Handle shadows in `brick_mask`, check on renders with
  `scripts/check_preprocess.py`.
- Retrain on the current code (`mask2-pad`) before comparing further.

## 2026-11-18 (2): Training with renders + internet photos

**Question:** Do real photos in the training set close the render-to-real
gap? And what do they cost on renders?

**Setup:**

- Configs: [`configs/shape_contrast.yaml`](../configs/shape_contrast.yaml)
  (renders only) and
  [`configs/shape_contrast_internet.yaml`](../configs/shape_contrast_internet.yaml):
  the same plus the 1,901 photos of `internet/train` (13 parts), each used 8
  times per epoch (`real_repeat: 8`): 47,208 training images per epoch, about
  a third real photos.
- Code: `mask2-pad`, seeds 0, 1, 2 for each config. Model selection as
  always on the render val set.
- `internet/val` is now logged after every epoch (`internet_val/*` in W&B)
  for every run. Final numbers below with `scripts/eval_internet.py`.
- Speed on the RTX 4080: ~9 s per epoch renders only, ~11–14 s with photos.

**Result:**

| Run | Config | Seed | Render test top-1 | Length / other errors | Internet val top-1 | Top-3 | Per capture | Conf |
|---|---|---|---|---|---|---|---|---|
| `efgo8od0` | renders only | 0 | 92.03 % | 113 / 206 | 23.3 % | 38.8 % | 29.2 % | 0.43 |
| `czendws5` | renders only | 1 | 92.10 % | 120 / 196 | 24.2 % | 39.6 % | 31.2 % | 0.44 |
| `7igjkjkp` | renders only | 2 | 92.45 % | 113 / 189 | 28.1 % | 43.5 % | 32.3 % | 0.44 |
| `rydhagpu` | + internet | 0 | 91.80 % | 107 / 221 | 93.3 % | 98.3 % | 95.8 % | 0.83 |
| `x32vqyv2` | + internet | 1 | 92.47 % | 106 / 195 | 93.5 % | 98.0 % | 95.8 % | 0.82 |
| `p29f3lck` | + internet | 2 | 91.70 % | 105 / 227 | 93.5 % | 98.6 % | 97.9 % | 0.83 |

Mean (range) over the 3 seeds:

| Config | Render test top-1 | Length errors | Other errors | Internet val top-1 | Per capture |
|---|---|---|---|---|---|
| renders only | 92.2 % (92.0–92.5) | 115 | 197 | 25.2 % (23.3–28.1) | 30.9 % |
| + internet | 92.0 % (91.7–92.5) | 106 | 214 | 93.4 % (93.3–93.5) | 96.5 % |

- McNemar on captures, every renders-only model vs every mixed model: 0 vs
  61–66 captures, p < 0.001. Between seeds of the same config: p ≥ 0.5.
- Image-level McNemar is misleading here: two renders-only seeds differ with
  p = 0.02 on images but p = 0.55 on captures. Only use the capture test.
- Macro-F1 on internet val (0.81–0.92 for the mixed models) is not reliable
  with 13 parts: one stray prediction of a part that is not in the photos
  adds a class with F1 = 0 to the average. Use top-1 per capture.

**Curves:** renders-only models jump between 4 % and 31 % on internet val from
epoch to epoch; the transfer is luck, not learned. With photos, internet val
rises steadily to ~93 % by epoch 20 and stays there, so model selection on
render val also picks a good epoch for the photos.

**Conclusion:**

1. Real photos in training close the gap on the internet photos almost
   completely (25 % → 93 %) and make the confidences sensible again (0.83 at
   93 %).
2. Cost on renders: none measurable. Top-1 -0.2 pp, within the seed range;
   fewer length errors, slightly more other errors.
3. This is an upper bound: train and val photos come from the same rig (same
   light, background, cropping, partly the same bricks). The model may have
   learned the rig rather than real bricks. The real test is our own photos.
4. `models/shape.pt` stays `56b289he` until that test is done.

**Open:**

- Our own photos: freeze `test.csv`, evaluate all six models.
- Fewer repeats (`real_repeat: 2, 4`) and fewer photos: how many real photos
  are needed?

## 2026-11-18: Internet photos, first large real-photo evaluation

**Question:** How large is the render-to-real gap on many real photos? How
much of it comes from the preprocessing and how much from the model?

**Setup:**

- Data: internet photos ([data/README.md](../data/README.md), section 2b),
  split `data/splits/internet.csv` (`e4434e94`), **val**, only the 13 parts
  that are also render classes: 356 photos from 96 captures (one capture = one
  brick, seen by 2–6 cameras).
- Models: `hki5v89j` (baseline, mask2), `o8jomt0u` (baseline, repo before
  #13), `56b289he` (contrast stretch, mask2). `lydt7czi` skipped: size
  feature, and the photos are tightly cropped, so the brick size in pixels
  says nothing.
- Preprocessing: current code (`mask2-pad`), although the checkpoints were
  trained with older versions (open item `PREPROCESS_VERSION`).
- Script: `scripts/eval_internet.py`.

**Result** (internet val, 356 photos / 96 captures):

| Run | Config | Top-1 | Top-3 | Top-1 among the 13 | Top-1 per capture | Conf | Length errors | Other errors |
|---|---|---|---|---|---|---|---|---|
| `hki5v89j` | baseline | 2.5 % | 9.3 % | 12.4 % | 3.1 % | 0.58 | 2 | 345 |
| `o8jomt0u` | baseline | 1.7 % | 8.1 % | 12.1 % | 2.1 % | 0.56 | 2 | 348 |
| `56b289he` | contrast stretch | 25.8 % | 40.4 % | 38.2 % | 28.1 % | 0.46 | 45 | 219 |

- *Top-1 among the 13:* best guess among the parts that occur in these
  photos. *Per capture:* one answer per capture, mean probability over its
  cameras.
- McNemar `hki5v89j` vs `56b289he`: images 2 / 85, captures 0 / 24
  (only baseline right / only contrast right), p < 0.001 for both. The test on
  captures is the honest one: cameras of one capture are not independent.
- `hki5v89j` vs `o8jomt0u`: no difference (captures 2 / 1, p = 1.0).

Mask threshold `thr` of `brick_mask` (exploration on val, not a setting we
keep):

| `thr` | 12 (default) | 20 | 30 | 45 |
|---|---|---|---|---|
| `56b289he` | 25.8 % | 30.6 % | 31.7 % | 28.1 % |
| `hki5v89j` | 2.5 % | 2.5 % | 3.1 % | 3.4 % |

**What the model sees** (one photo per capture, photo | 96 × 96 crop,
green = right):
[crops 1](figures/2026-11-18_internet_val_crops_1.jpg),
[crops 2](figures/2026-11-18_internet_val_crops_2.jpg)

1. In most crops the brick is found, but about half show a light halo: the
   shadow is counted as brick and the stretch makes it bright. The renders
   have no shadows.
2. About 8 of 96 crops are unusable: white or beige bricks on a light
   background, only fragments of the brick are left.
3. Many errors are on clean crops: 3005 (brick 1×1) is wrong in all 57 photos,
   mostly as 3063 (curved brick); 3062 (round brick 1×1) 0 / 18; 3003 → 99301
   / 3046 (roof tiles); dark green 3021 → 3020.
4. 3020 (plate 2×4) is right 22 / 22, but it is also the most frequent answer
   (65 times): partly a bias, not only recognition.

**Conclusion:**

1. The gap is huge: ~92 % on renders vs 25.8 % on real photos for the default
   model. Without the contrast stretch the model is at chance level.
2. Contrast stretch is clearly the better default on real photos, now on
   enough photos for a significant paired test.
3. The preprocessing explains a part (+6 pp with a stricter threshold), most
   of the gap remains: the model itself does not transfer.
4. Limits: the internet photos are not our setup (tightly cropped, other
   camera, other light). They show the size of the problem, the number that
   counts stays the one on our own photos. `internet/test` is untouched.

**Open:**

- Train with renders + `internet/train` (new config), evaluate on
  `internet/val` and on our own photos.
- Handle shadows in `brick_mask`, check on renders with
  `scripts/check_preprocess.py`.
- Retrain on the current code (`mask2-pad`) before comparing further.

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
