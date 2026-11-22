# LEGO Brick Recognition

Take a photo of loose LEGO bricks, get a list of what is on it (part, colour,
count), and find out which official LEGO sets you can (almost) build with them.

Data Science group project, FH Dortmund, winter semester 2026/27.

> **Project rule:** we do not use ready-made tools or models that recognise
> LEGO for us. We build and train the recognition ourselves.

**New here?** Read [How it works](#how-it-works) and [Where we are](#where-we-are),
then follow [Setup](#setup). Words like *render*, *top-1* or *seed* are
explained in the [Glossary](#glossary).

**Contents:** [How it works](#how-it-works) ·
[Where we are](#where-we-are) ·
[Links](#links) ·
[Setup](#setup) ·
[Get the data](#get-the-data) ·
[Train the shape classifier](#train-the-shape-classifier) ·
[Find and name the bricks on a sheet photo](#find-and-name-the-bricks-on-a-sheet-photo) ·
[Use the pipeline in code](#use-the-pipeline-in-code) ·
[Code tour](#code-tour) ·
[Working together](#working-together) ·
[Glossary](#glossary)

## How it works

```
Photo → Find bricks → Shape (CNN) + Colour → Inventory → "What can I build?"
```

| Step | What happens | Approach | Status |
|---|---|---|---|
| Capture | We photograph the bricks | Bricks on a printed capture sheet (template v2) with 4 ArUco markers, so every photo can be warped flat to the same scale (8 px/mm) | **in use**: 246 sheet photos of 19 parts |
| Find bricks | Each brick is found and cut out | A YOLO-seg model with one class "brick", trained by us from scratch (the old colour threshold is in the Git tag `v1-legacy`) | **in use** on separated bricks; touching bricks and piles still open |
| Shape | Each cut-out gets a part number, e.g. 3001 = brick 2×4 | CNN on greyscale cut-outs, trained on internet photos and renders plus our own renders; the size of the brick in mm (known from the sheet) weighs the answer | **in use**: 84 % of the bricks on our test photos right, see [Where we are](#where-we-are) |
| Colour | Each brick gets a LEGO colour | White balance with the grey card, then the nearest measured LEGO colour | planned |
| Inventory | Everything is counted | Part, colour, count + confidence | planned |
| Sets | We compare the inventory with official sets | Coverage and missing parts per set, using Rebrickable part lists | planned |

Why a grey card? Its colour is a known neutral grey. That lets us correct the
colours of every photo (white balance) before we decide which LEGO colour a
brick has.

**Core experiment:** the tool has to work on our own sheet photos, but almost
all training images are not ours: internet photos and renders. How much
accuracy do we lose between the training images and our photos (the
*render-to-real gap*), and how do we win it back? At first (Kaggle renders
only) the gap was between renders and real photos. Since 2026-11-21 the
larger gap is between the internet photos and our photos: the team's model
gets about 95 % top-1 on internet photos (val) but about 67 % on our test
photos without the size prior (84 % with it), see the
[experiment log](docs/experiments.md), 2026-11-21.

## Where we are

From the [experiment log](docs/experiments.md) (last entry: 2026-11-21):

- **Whole pipeline on our own photos:** on sheet photos that no model has
  seen (54 photos, 171 bricks of 19 parts), YOLO finds the bricks and the
  shape model names 84 % of them correctly (92 % per photo; mean of 3
  training runs of `shape_internet_own`, with the size prior). Default model:
  `ok1tq8ks`, the best of the three on val.
- **The sheet gives the size:** every brick is measured in mm (long and short
  side). Weighing the CNN's answer with that size lifts our photos from about
  56 % to 82–84 %, without any training.
- **Training data:** internet photos and renders plus our own renders
  (LDraw, `scripts/render_ldraw.py`). The training images are cut out by the
  same YOLO model as the sheet photos, which was better than the old mask in
  all 3 seeds.
- **Older experiments** (Kaggle renders, size feature, models without
  contrast stretch or YOLO masks): their code is in the Git tag
  [`v1-legacy`](https://github.com/BluePineappleCake/lego-recognition-fhdo/tree/v1-legacy);
  `main` only keeps the way the team's model is trained.
- **Main error left:** a brick vs a plate with the same footprint (3003 vs
  3022, 3004 vs 3023). From above they look and measure almost the same.
- **Careful:** test and training photos come from the same photo sessions. A
  new session (other day, other light) as a frozen test is the next step.
- **Not started:** colour, inventory, sets.

Open questions and next steps are at the top of the
[experiment log](docs/experiments.md).

## Links

| What | Where | Access |
|---|---|---|
| Experiment tracking | [W&B project `lego-fhdo/lego-recognition`](https://forge.coreweave.com/wandb/lego-fhdo/lego-recognition) | team members – ask for an invite to the W&B team `lego-fhdo` |
| Training data (photos, renders) | [TU Gdańsk: LEGO bricks for training classification network](https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,202309140842198941751-0) | public |
| Our photos | private Kaggle dataset `gonsterenjoyer/lego-sheet-photos`, see [`data/README.md`](data/README.md) | team members (added as collaborators) |
| Part and set lists | [Rebrickable downloads](https://rebrickable.com/downloads/) | public |
| Experiment log | [`docs/experiments.md`](docs/experiments.md) | – |
| Where the data goes | [`data/README.md`](data/README.md) | – |

## Setup

You need Python 3.10 or newer and Git. The commands below are for Windows
(PowerShell). Comments show the macOS/Linux version where it is different.

**1. Get the code and create a virtual environment.** A virtual environment is
a separate Python installation for this project, so its packages do not mix
with your other projects.

```powershell
git clone https://github.com/BluePineappleCake/lego-recognition-fhdo.git
cd lego-recognition-fhdo
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. If PowerShell does not allow the
activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once and try again.

**2. Only if you want to train on your own NVIDIA GPU:** install PyTorch with
CUDA *now*, before step 3. Copy the install command from
[pytorch.org](https://pytorch.org) (*Get Started*). Reason: `pip install torch`
from PyPI gives you a CPU-only version on Windows. Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

**3. Install the project:**

```powershell
pip install -e ".[dev,train]"   # our package + tools for tests and training
pre-commit install              # checks the code style before every commit
```

`-e` (editable) means that changes in `src/` work immediately, without a new
install. The words in brackets are optional extras:

| Extra | Adds | You need it for |
|---|---|---|
| `dev` | pytest, ruff, pre-commit, nbstripout, scikit-learn | tests and code style (everyone) |
| `train` | PyTorch, torchvision, W&B, Ultralytics | training and using the model (YOLO cuts out the training images) |
| `detect` | Ultralytics | finding the bricks with YOLO-seg (`find_bricks.py`, `eval_own.py`, `export_yolo.py`) |
| `docs` | matplotlib | the notebooks in `notebooks/` |

**4. Check that it works:**

```powershell
pytest   # all tests should pass; without PyTorch a few are skipped. No dataset needed.
```

## Get the data

The data is **not** in this repository. [`data/README.md`](data/README.md)
shows the folder layout and how to download everything.

- **Training data for the team's model:** photos and renders of 18 parts
  from the [TU Gdańsk LEGO set](https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,202309140842198941751-0)
  into `data/raw/internet/`, and our own renders into `data/raw/own_renders/`.
  Which folders and how to name them: [`data/README.md`](data/README.md),
  sections 2b and 2d.
- **Our own photos (finding bricks and evaluation):** sheet photos in a
  private Kaggle dataset. The shape model trains on them only for parts
  without internet data (at the moment 4073, `own_photo_parts`). YOLO trains
  on their train split and picks its best epoch on val; the size table
  (`data/reference/part_sizes.csv`) is measured on train. Only their test
  split is unseen.
- **Splits:** the files `data/splits/*.csv` say which image belongs to train,
  val or test. They are in Git, so everyone uses exactly the same split.
  - Never regenerate a split file: results can only be compared on the same
    split.
  - Never train or tune anything on the test split of our sheet photos
    (`data/splits/sheets.csv`): decisions are made on val. It is **not a
    frozen test set yet**: its photos come from the same photo sessions as
    train and val. A new session as a frozen test is the next step.

## Train the shape classifier

The shape classifier is a small CNN. It looks at the grey 96 × 96 cut-out of
one brick and answers with a part number. Which images it learns from is set
by `train_on` in the config; the classes are the parts that have training
images:

| Source (`train_on`) | Folder | Parts |
|---|---|---|
| `internet_photos` | `data/raw/internet/photos/` | 18 |
| `internet_renders` | `data/raw/internet/renders/` | 17 |
| `own_renders` | `data/raw/own_renders/` (`scripts/render_ldraw.py`) | 3 so far |
| `own_photos` | bricks on our sheet photos, only the parts in `own_photo_parts` | e.g. 4073 |

The team's model uses **`configs/shape_internet_own.yaml`**: internet photos
and renders, our renders, and our photos of 4073 (19 parts). Its training
images are cut out by the YOLO brick finder, the same way as the sheet photos
later (where YOLO finds nothing, by `brick_mask`).

**What one training run does** (`scripts/train_shape.py`):

1. It reads the config and loads the images of every source in `train_on`,
   plus the internet photos for evaluation. The first run preprocesses all
   images once (a few minutes, YOLO runs over each of them)
   and caches them in `data/processed/`.
2. It starts a run in W&B and logs a few training images, so you can see what
   the network sees.
3. It trains for 30 epochs. After every epoch it checks the val split of
   every source and keeps the checkpoint with the best macro-F1 on the val
   split of `select_on` (the internet photos for `shape_internet_own`).
4. It tests this best checkpoint once on the test splits of the render
   sources and on the internet photos (val split). The test splits of the
   photos are left for `scripts/eval_own.py` and `scripts/eval_internet.py`.

Result: the checkpoint `models/shape_<run id>.pt` on your disk, the same file as
W&B artifact `shape-cnn`, and all curves and numbers in W&B.

### On your PC (NVIDIA GPU)

**Once:** log in to W&B. You need an invite to the W&B team `lego-fhdo`.

```powershell
wandb login
```

**In every new terminal:** tell W&B to save your runs in the team project and
not in your private account:

```powershell
$env:WANDB_ENTITY="lego-fhdo"   # cmd: set WANDB_ENTITY=lego-fhdo   macOS/Linux: export WANDB_ENTITY=lego-fhdo
```

**Sanity check first.** It trains on a single batch until the loss is almost 0.
If that does not work, something is broken, and a long run would be wasted.

```powershell
python scripts/train_shape.py --config configs/shape_internet_own.yaml --sanity
# last line: "[sanity] loss on that batch after training 0.0... (should be ~0)"
```

**Real runs:**

```powershell
python scripts/train_shape.py --config configs/shape_internet_own.yaml           # the team's model
python scripts/train_shape.py --config configs/shape_internet_own.yaml --seed 1  # the same, seed 1
```

One run of `shape_internet_own` takes about 9 minutes on an RTX 4080 (30
epochs of ~60,000 images). New renders in `data/raw/own_renders/` or new
sheet photos are added to their split files automatically; commit those files
afterwards.

| Option | Meaning |
|---|---|
| `--config` | YAML file in `configs/` (required) |
| `--internet-data` | folder with `photos/` and `renders/` of the internet dataset (default: `data/raw/internet/`) |
| `--seed` | random seed, e.g. `--seed 1` (default: 0) |
| `--epochs` | number of epochs, overrides the config |
| `--sanity` | run only the sanity check |

**Settings:** all settings and their defaults are in `TrainConfig`
(`src/legorec/shape/training.py`). A config file only lists what it changes. A
key that is not a setting there (e.g. the typo `epoch: 5`) stops the run with
an error. This is on purpose: a typo must not quietly change an experiment.

Before you compare variants, read the
[rules for experiments](#rules-for-experiments). Write what you found into
[`docs/experiments.md`](docs/experiments.md).

### Without a local GPU

Training needs the internet set and our renders; neither is on Kaggle or
Colab. To train there, upload `data/raw/internet/` and `data/raw/own_renders/`
as a private dataset, set `LEGOREC_DATA` to the folder that contains `raw/`,
and `LEGOREC_PROCESSED` to a writable folder (e.g. `/kaggle/working/processed`).
The older Kaggle-render runs and their notebook setup are in the Git tag
`v1-legacy`.

## Find and name the bricks on a sheet photo

`scripts/find_bricks.py` runs the first steps of the pipeline on photos of the
capture sheet: warp the photo flat, find every brick, and give each one a part
number with the shape model.

```powershell
python scripts/find_bricks.py data/raw/own_photos/MIXED*/IMG_8844.jpg
python scripts/find_bricks.py photo.jpg --no-shape           # only find the bricks
```

- It writes the photo with numbered outlines and one crop per brick to
  `data/processed/find_bricks/`, and prints one line per brick.
- Finding the bricks needs the `detect` extra and the YOLO model in
  `models/bricks_yolo.pt` (or the file in `LEGOREC_BRICK_MODEL`).
- The part numbers come from `models/shape.pt` (or `--shape-model`), weighed
  by the measured size of every brick in mm (the **size prior**, table
  `data/reference/part_sizes.csv`). `--no-size-prior` switches it off. After
  new sheet photos, rebuild the table with `python scripts/size_check.py`
  and commit it.

To evaluate shape models on real photos:

```powershell
python scripts/eval_own.py --size-prior --models models/shape_ok1tq8ks.pt  # whole pipeline, our test photos
python scripts/eval_internet.py                                             # internet photos, val split
```

Both explain at the top what they report. `eval_own.py` uses the test split of
our sheet photos by default; comparing several models is only fair on the
parts that all of them know, and the script does exactly that.

## Use the pipeline in code

This is the interface for the other pipeline steps (colour, inventory):

```python
import cv2
from legorec.pipeline import find_and_name

for brick in find_and_name(cv2.imread("IMG_8844.jpg")):
    print(brick.part, brick.confidence, brick.crop.bbox)  # e.g. 3001 0.93 (...)
```

- `brick.crop` is a `BrickCrop`: the image, the mask of the brick and its
  position on the sheet (8 px/mm). The colour step starts from it.
- `confidence` is the model's probability for its answer (0 to 1). It is **not
  calibrated** yet: on real photos the model can be wrong with 70–80 %
  confidence.
- Only the shape step, for crops you already have:
  `load_classifier().predict(crops)` (`legorec.shape.classifier`). The size
  prior is on there, so the crops must come from a sheet photo; for other
  images use `load_classifier(size_prior=None)`.
- The model is loaded from `models/shape.pt`, or from the file named in the
  environment variable `LEGOREC_SHAPE_MODEL`.
- Checkpoints of older experiments (Kaggle, size feature, ...) are refused
  with a message: they need the Git tag `v1-legacy`.

**Get the team's default model.** It is the W&B artifact `shape-cnn:default`
(which run it is and why: [experiment log](docs/experiments.md)). Download it
once and copy it to `models/shape.pt`:

```powershell
python -c "import wandb; wandb.Api().artifact('lego-fhdo/lego-recognition/shape-cnn:default').download('models')"
copy models\shape_ok1tq8ks.pt models\shape.pt   # macOS/Linux: cp models/shape_ok1tq8ks.pt models/shape.pt
```

The downloaded file is called `shape_<run id>.pt`; `ok1tq8ks` is the run of the
current default (`shape_internet_own`, chosen on 2026-11-21).

## Code tour

```
src/legorec/     reusable code: data/, segment/, shape/, eval/
                 (planned: colour, inventory, sets)
configs/         one YAML file per training experiment
data/splits/     versioned train/val/test file lists
data/labels/     versioned labels corrected by hand (YOLO)
notebooks/       exploration only, named NN_initials_topic.ipynb
scripts/         command-line tools: training, evaluation, labelling, figures
tests/           pytest tests; they also show how each function is called
docs/            experiment log and progress presentations
```

Every file starts with a docstring that explains the idea behind it.

### Where to start reading

Read these five files in this order; together they are the whole pipeline:

1. `src/legorec/pipeline.py`: photo in, named bricks out, in 20 lines
2. `src/legorec/segment/bricks.py`: how YOLO finds the bricks on the sheet
3. `src/legorec/shape/preprocess.py`: one brick → the 96 × 96 grey image the CNN sees
4. `src/legorec/shape/classifier.py`: the trained CNN plus the size prior → part number
5. `src/legorec/shape/training.py` and `scripts/train_shape.py`: how the CNN is trained

All files, by pipeline step:

| File | What it does |
|---|---|
| `src/legorec/pipeline.py` | `find_and_name(photo)`: the whole pipeline on one sheet photo |
| `src/legorec/types.py` | `BrickCrop`, the data passed from one step to the next |
| **Finding the bricks** | |
| `src/legorec/segment/sheet.py` | finds the 4 ArUco markers, warps the brick area flat (`brick_area`) |
| `src/legorec/segment/bricks.py` | `bricks_on_sheet(photo)`: one `BrickCrop` per brick, found by YOLO-seg |
| `src/legorec/segment/single.py` | the same for images with one brick each (training images): YOLO, `brick_mask` as fallback |
| `src/legorec/segment/labels.py` | YOLO label files (one class "brick"), for training YOLO |
| **Naming the part** | |
| `src/legorec/shape/preprocess.py` | one `BrickCrop` → 96 × 96 grey image (the same code for training and use) |
| `src/legorec/shape/model.py` | the CNN and its checkpoint files |
| `src/legorec/shape/classifier.py` | `ShapeClassifier`: a trained CNN used on new crops, with the size prior |
| `src/legorec/shape/size_prior.py` | the size prior: how well the measured size in mm fits each part |
| **Training** | |
| `src/legorec/data/splits.py` | train / val / test files in `data/splits/`, the same for the whole team |
| `src/legorec/data/renders.py`, `internet.py`, `sheets.py` | the data sources: index, split, sessions |
| `src/legorec/shape/training_data.py` | the training sources (`train_on`), preprocessed and cached |
| `src/legorec/shape/training.py` | `TrainConfig` (all settings), one training epoch, evaluation |
| `src/legorec/shape/augment.py`, `batches.py` | random rotation, zoom and lighting per batch; batches on the GPU |
| `src/legorec/eval/metrics.py` | top-1, top-3, macro-F1, length errors, McNemar test |

**Scripts** (each explains its options at the top):

| Script | What it does |
|---|---|
| `find_bricks.py` | sheet photo → bricks with outlines and part numbers |
| `train_shape.py` | train the shape CNN |
| `eval_internet.py`, `eval_own.py` | shape models on the internet photos / on our sheet photos (whole pipeline) |
| `export_yolo.py`, `cvat_labels.py`, `show_labels.py` | YOLO labels: export, correct by hand in CVAT, check by eye |
| `render_ldraw.py` | our own renders from the LDraw library (Blender) |
| `size_check.py` | the size of every part in mm: the table of the size prior |

## Working together

**Team:**

| Name | Main area |
|---|---|
| Togzhan | TBD |
| Ainur | TBD |
| Sami | TBD |
| David | TBD |

**How we work:**

1. Tasks live on the GitHub project board. We discuss on Discord.
2. One branch per task, e.g. `shape/baseline-cnn`.
3. A pull request into `main`; one teammate reviews it before it is merged.

- Code that is used in more than one notebook moves into `src/`.
- Every experiment logs its config to Weights & Biases (`train_shape.py` does
  this for you). Results and conclusions go into
  [`docs/experiments.md`](docs/experiments.md).

### Rules for experiments

| Rule | Why |
|---|---|
| One change per experiment | Otherwise we cannot tell which change caused the effect. |
| Compare over several seeds, not single runs | Runs with the same settings differ by up to ~2.4 pp top-1 on our test photos (83.6–86.0 %, 3 seeds of `shape_internet_own` with the size prior; see the [log](docs/experiments.md), 2026-11-21). One lucky run can fool us. |
| Never train or tune on the real-photo test set | The test set must stay unseen until the end. Otherwise our final number is too optimistic. |

## Glossary

| Term | Meaning |
|---|---|
| Render | A computer-generated image of a 3D LEGO model. Training data: the internet set and our own LDraw renders. |
| Real photo | A photo of real bricks that we take ourselves. The tool has to work on these in the end. |
| Render-to-real gap | How much accuracy a model loses between the images it is trained on (renders, internet photos) and our own photos. Measuring and closing it is our core experiment. |
| Part number | LEGO's number for a part shape, e.g. 3001 = brick 2×4. These are our classes. |
| Pose | One orientation of a brick in the renders. The same pose comes in several colours, so there are several almost identical images of it. |
| Group | Images that must stay in one split: the colours of one render pose, the cameras of one internet capture, the bricks of one sheet photo. Otherwise a near-copy of a training image would end up in the test set, and the test would be too easy. |
| Train / val / test | *Train*: the model learns from it. *Val*: we use it to choose (best epoch, settings). *Test*: only for the final number. |
| Frozen test set | A test set that never changes and is never used for decisions. We do not have one yet: the test split of `sheets.csv` comes from the same sessions as train and val. |
| Crop | The cut-out of one brick, scaled to 96 × 96 grey pixels. This is what the CNN sees. |
| Baseline | The team's config (`configs/shape_internet_own.yaml`). Every variant is compared against it. |
| Top-1 / top-3 accuracy | Share of images where the right part is the model's first guess / among its three best guesses. |
| Macro-F1 | The F1 score of every part, averaged, so every part counts the same. Used to pick the best epoch. |
| pp | Percentage points: from 94.45 % to 92.35 % is -2.1 pp. |
| Seed | Start value of the random number generator. Same config with another seed → a slightly different result. |
| Run | One training run. W&B gives it an id (e.g. `56b289he`) and a random name (e.g. `valiant-pond-9`). |
| Config | A YAML file in `configs/` with the settings that differ from the defaults. |
| Checkpoint | A saved model: weights plus settings, e.g. `models/shape_56b289he.pt`. |
| W&B, artifact | Weights & Biases, the website where we log our runs. An *artifact* is a versioned file stored there, e.g. a checkpoint (`shape-cnn:v2`). An *alias* such as `default` points to one version. |
| Sanity check | Training on one single batch until the loss is ~0. It shows that the model and the training loop can learn at all. |
| Augmentation | Random changes to the training images (rotation, zoom, shift, lighting), so the model learns the brick and not the exact picture. |
| OneCycle | A learning-rate schedule: the learning rate first goes up, then down to almost 0 at the end of training. |
| Label smoothing | The training target for the right part is about 0.9 instead of 1 (setting `label_smoothing: 0.1`). |
| Length errors | Confusions between parts that differ only in length, like brick 1×3 and 1×4. The pairs follow from `PART_GEOMETRY` in `src/legorec/eval/metrics.py` (same family and width, different length). |
| Size prior | Resizing to 96 × 96 removes the real size of a brick. On a sheet photo it is measured in mm and added to the scores: how well does that size fit each part? (`src/legorec/shape/size_prior.py`) |
| Contrast stretch | The grey values of each brick are spread over the full range 0–255. Idea: in greyscale a red brick is almost black. |
| Calibrated | A model is calibrated if its confidence matches how often it is right: answers with 80 % confidence are right in about 80 % of cases. *Underconfident*: the confidence is lower than the accuracy. |
| Temperature scaling | A simple way to calibrate a model: divide its outputs by one number that is fitted on the val set. |
| McNemar test | A statistical test that compares two models on the *same* images. It only looks at the images where exactly one of the two models is right. |

## License

MIT
