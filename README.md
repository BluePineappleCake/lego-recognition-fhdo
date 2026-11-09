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
[Try a model on your own photos](#try-a-model-on-your-own-photos) ·
[Use the shape model in code](#use-the-shape-model-in-code) ·
[Code tour](#code-tour) ·
[Working together](#working-together) ·
[Glossary](#glossary)

## How it works

```
Photo → Find bricks → Shape (CNN) + Colour → Inventory → "What can I build?"
```

| Step | What happens | Approach | Status |
|---|---|---|---|
| Capture | We photograph the bricks | Fixed camera, even light, grey card in every photo | planned |
| Find bricks | Each brick is found and cut out | OpenCV contours for separated bricks; YOLO-seg for piles (stretch goal: only if time allows) | planned |
| Shape | Each cut-out gets a part number, e.g. 3001 = brick 2×4 | CNN on greyscale cut-outs, trained on renders, fine-tuned on real photos | **v0 trained**, see [Where we are](#where-we-are) |
| Colour | Each brick gets a LEGO colour | White balance with the grey card, then the nearest measured LEGO colour | planned |
| Inventory | Everything is counted | Part, colour, count + confidence | planned |
| Sets | We compare the inventory with official sets | Coverage and missing parts per set, using Rebrickable part lists | planned |

Why a grey card? Its colour is a known neutral grey. That lets us correct the
colours of every photo (white balance) before we decide which LEGO colour a
brick has.

**Core experiment:** we train on *renders* (computer-generated images), but in
the end the tool has to work on real photos. How much accuracy do we lose
between renders and real photos (the *render-to-real gap*), and how do we win
it back?

## Where we are

From the [experiment log](docs/experiments.md) (last entry: 2026-11-05):

- **The shape classifier v0 works on renders:** about 94.5 % top-1, i.e. about
  945 of 1,000 test renders (images it has never seen) get the right part
  number. Two runs with the same settings can differ by about 1 percentage
  point, so smaller differences between two variants do not mean anything yet.
- **First real photos:** the *contrast stretch* variant did better than the
  baseline, although it is worse on renders. This was a first exploration,
  not the final test.
- **Default model:** the contrast-stretch model, for now.
- **Everything else** (finding bricks, colour, inventory, sets) is planned.

Open questions and next steps are at the top of the
[experiment log](docs/experiments.md).

## Links

| What | Where | Access |
|---|---|---|
| Experiment tracking | [W&B project `lego-fhdo/lego-recognition`](https://forge.coreweave.com/wandb/lego-fhdo/lego-recognition) | team members – ask for an invite to the W&B team `lego-fhdo` |
| Training data (renders) | [Kaggle: Images of LEGO Bricks](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images) | public |
| Our photos | private Kaggle dataset (link follows) | team members |
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
| `train` | PyTorch, torchvision, timm, W&B | training and using the model (already installed on Kaggle) |
| `docs` | matplotlib | the class overview figures (the PDF table also needs `pip install reportlab`) |
| `app` | Streamlit | the app (planned) |

**4. Check that it works:**

```powershell
pytest   # all tests should pass; without PyTorch a few are skipped. No dataset needed.
```

## Get the data

The data is **not** in this repository. [`data/README.md`](data/README.md)
shows the folder layout and how to download everything.

- **Renders (training):**
  [Images of LEGO Bricks](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images)
  by J. Hazelzet, 40,000 images of 50 part types. Unzip them into
  `data/raw/renders/`. The code expects the folder that *directly* contains
  the 40,000 PNGs: `data/raw/renders/dataset/`. The download also contains an
  older version of the dataset ("LEGO brick images v1"). Do not use that one.
- **Real photos (fine-tuning and test):** our own photos, in a private Kaggle
  dataset.
- **Splits:** the files `data/splits/*.csv` say which image belongs to train,
  val or test. They are in Git, so everyone uses exactly the same split.
  - Never regenerate `renders.csv`: render results can only be compared on the
    same split.
  - The test split of our real photos is frozen: never train or tune anything
    on it.

**Which bricks are in the renders?** `scripts/class_overview.py` draws all 50
classes with example renders (PNG/PDF/CSV). `scripts/class_overview_pdf.py`
makes a PDF table with notes on look-alikes. Both need the `docs` extra, the
PDF table also needs `reportlab`:

```powershell
pip install -e ".[docs]" reportlab
python scripts/class_overview_pdf.py --data data/raw/renders/dataset --out docs/figures/classes_table.pdf
```

`docs/figures/` is generated and not committed.

## Train the shape classifier

The shape classifier is a small CNN. It looks at the grey 96 × 96 cut-out of
one brick and answers with a part number. So far it is trained on renders only.

**What one training run does** (`scripts/train_shape.py`):

1. It reads the config and loads the preprocessed renders. The very first run
   preprocesses all 40,000 images once (a few minutes) and caches them in
   `data/processed/`.
2. It starts a run in W&B and logs a few training images, so you can see what
   the network sees.
3. It trains for 30 epochs. After every epoch it checks the validation set and
   keeps the checkpoint with the best validation macro-F1.
4. It tests this best checkpoint once on the render test set.

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
python scripts/train_shape.py --config configs/shape_baseline.yaml --sanity
# last line: "[sanity] loss on that batch after training 0.0... (should be ~0)"
```

**Real runs:**

```powershell
python scripts/train_shape.py --config configs/shape_baseline.yaml               # baseline
python scripts/train_shape.py --config configs/shape_size_feature.yaml --seed 1  # + brick size, seed 1
python scripts/train_shape.py --config configs/shape_contrast.yaml               # contrast stretch per crop
```

One run takes about 4–5 minutes on an RTX 4080.

| Option | Meaning |
|---|---|
| `--config` | YAML file in `configs/` (required) |
| `--data` | folder that directly contains the 40,000 PNGs (default: `data/raw/renders/dataset/`) |
| `--seed` | random seed, e.g. `--seed 1` (default: 0) |
| `--epochs` | number of epochs, overrides the config |
| `--sanity` | run only the sanity check |

**Settings:** all settings and their defaults are in `TrainConfig`
(`src/legorec/shape/train.py`). A config file only lists what it changes. A
key that is not a setting there (e.g. the typo `epoch: 5`) stops the run with
an error. This is on purpose: a typo must not quietly change an experiment.

Before you compare variants, read the
[rules for experiments](#rules-for-experiments). Write what you found into
[`docs/experiments.md`](docs/experiments.md).

### On Kaggle (no local GPU needed)

1. *Create → New Notebook*. In the right panel: *+ Add Input* →
   "Images of LEGO Bricks" (joosthazelzet). *Settings* → Accelerator **GPU**,
   Internet **on** (needs a phone-verified Kaggle account).
2. *Add-ons → Secrets*: add `WANDB_API_KEY` (your key from
   [wandb.ai/authorize](https://wandb.ai/authorize)) and tick it for this
   notebook.
3. Run:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
os.environ["WANDB_ENTITY"] = "lego-fhdo"
os.environ["LEGOREC_PROCESSED"] = "/kaggle/working/processed"  # /kaggle/input is read-only

!git clone https://github.com/BluePineappleCake/lego-recognition-fhdo.git
%cd lego-recognition-fhdo
!pip install -q -e ".[train]"   # torch is preinstalled and stays as is
!python scripts/train_shape.py --config configs/shape_baseline.yaml \
    --data /kaggle/input/datasets/joosthazelzet/lego-brick-images/dataset
```

- `--data` must point to `.../dataset`. The input also contains the older
  version "LEGO brick images v1".
- For full training runs use *Save Version → Save & Run All*. Interactive
  sessions stop when you are idle.

## Try a model on your own photos

`scripts/predict_shape.py` runs a trained model on photos with **one brick per
photo**: plain background, no strong shadow (a shadow counts as part of the
brick).

```powershell
python scripts/predict_shape.py --checkpoint models/shape_<run id>.pt photo1.jpg photo2.jpg
python scripts/predict_shape.py --checkpoint models/shape_<run id>.pt my_photos/
```

- The script writes `predictions.png`: the 96 × 96 crop the model actually saw,
  next to its answer. Look at it first – a bad crop explains most bad
  predictions.
- If a folder has subfolders named after part numbers (`my_photos/3001/a.jpg`
  …), they are used as labels and the script reports top-1 and top-3
  accuracy.
- Size-feature models also need `--real_scale`. The docstring at the top of the
  script explains a rough calibration.

## Use the shape model in code

This is the interface for the other pipeline steps:

```python
from legorec.shape.predict import predict

# crop: a BrickCrop from the segmentation step
part, confidence = predict(crop)  # e.g. ("3001", 0.93)
```

- `confidence` is the model's probability for its answer (0 to 1). It is **not
  calibrated** yet: on real photos the model can be wrong with 70–80 %
  confidence.
- The model is loaded from `models/shape.pt`, or from the file named in the
  environment variable `LEGOREC_SHAPE_MODEL`.
- The checkpoint stores its preprocessing settings (e.g. `contrast`), so
  `predict` uses the same preprocessing as in training.
- Size-feature models need `load_model(path, real_scale=...)` before the first
  `predict`, see `src/legorec/shape/predict.py`.

**Get the team's default model.** It is the W&B artifact `shape-cnn:default`
(which run it is and why: [experiment log](docs/experiments.md)). Download it
once and copy it to `models/shape.pt`:

```powershell
python -c "import wandb; wandb.Api().artifact('lego-fhdo/lego-recognition/shape-cnn:default').download('models')"
copy models\shape_56b289he.pt models\shape.pt   # macOS/Linux: cp models/shape_56b289he.pt models/shape.pt
```

The downloaded file is called `shape_<run id>.pt`; `56b289he` is the run of the
current default.

## Code tour

```
src/legorec/     reusable code; so far data/, shape/ and eval/
                 (planned: segmentation, colour, inventory, sets)
configs/         one YAML file per experiment
data/splits/     versioned train/val/test file lists
notebooks/       exploration only, named NN_initials_topic.ipynb
scripts/         training, evaluation and overview figures
tests/           pytest tests; they also show how each function is called
docs/            experiment log and progress presentations
```

### Where to start reading

The shape step, in the order the data flows through it:

| File | What it does |
|---|---|
| `src/legorec/data/renders.py` | finds the Kaggle renders, splits them by pose, caches the preprocessed images |
| `src/legorec/shape/preprocess.py` | one brick image → 96 × 96 grey crop + measured size (the same code for renders and photos) |
| `src/legorec/shape/augment.py` | random rotation, zoom, shift and lighting for every training batch |
| `src/legorec/shape/model.py` | the CNN with the optional size input, checkpoints, `ShapeClassifier` for new images |
| `src/legorec/shape/train.py` | `TrainConfig` (all settings), one training epoch, evaluation |
| `scripts/train_shape.py` | puts it together: data → W&B run → training → test |
| `src/legorec/eval/metrics.py` | top-1, top-3, macro-F1, length errors |
| `src/legorec/shape/predict.py` | `predict(crop)`, the interface for the other pipeline steps |

Every file starts with a docstring that explains the idea behind it.

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
| Compare over several seeds, not single runs | Runs with the same settings differ by up to ~1 pp top-1 (see the [log](docs/experiments.md)). One lucky run can fool us. |
| Never train or tune on the real-photo test set | The test set must stay unseen until the end. Otherwise our final number is too optimistic. |

## Glossary

| Term | Meaning |
|---|---|
| Render | A computer-generated image of a 3D LEGO model. Our training data (Kaggle). |
| Real photo | A photo of real bricks that we take ourselves. The tool has to work on these in the end. |
| Render-to-real gap | How much accuracy a model loses when it is trained on renders and used on real photos. Measuring and closing it is our core experiment. |
| Part number | LEGO's number for a part shape, e.g. 3001 = brick 2×4. These are our classes (the renders have 50). |
| Pose | One orientation of a brick in the renders. Every pose was rendered by two cameras (L and R), so there are two almost identical images of it. |
| Split by pose | Both images of a pose always go into the same split. Otherwise a near-copy of a training image would end up in the test set, and the test would be too easy. |
| Train / val / test | *Train*: the model learns from it. *Val*: we use it to choose (best epoch, settings). *Test*: only for the final number. |
| Frozen test set | A test set that never changes and is never used for decisions. |
| Crop | The cut-out of one brick, scaled to 96 × 96 grey pixels. This is what the CNN sees. |
| Baseline | The standard config (`configs/shape_baseline.yaml`). Every variant is compared against it. |
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
| Length errors | Confusions between parts that differ mainly in length, like 1×3 and 1×4. The pairs are listed in `LENGTH_PAIRS` in `src/legorec/eval/metrics.py`. |
| Size feature | Resizing to 96 × 96 removes the real size of a brick. With `size_feature: true`, the measured size (in pixels) is an extra input to the network. |
| `real_scale` | Converts brick sizes from photo pixels to render pixels. Only needed for size-feature models on photos. |
| Contrast stretch | The option `contrast: stretch`: the grey values of each brick are spread over the full range 0–255. Idea: in greyscale a red brick is almost black. |
| Calibrated | A model is calibrated if its confidence matches how often it is right: answers with 80 % confidence are right in about 80 % of cases. *Underconfident*: the confidence is lower than the accuracy. |
| Temperature scaling | A simple way to calibrate a model: divide its outputs by one number that is fitted on the val set. |
| McNemar test | A statistical test that compares two models on the *same* images. It only looks at the images where exactly one of the two models is right. |

## License

MIT
