# LEGO Brick Recognition

Take a photo of loose LEGO bricks, get an inventory (part, colour, count),
and find out which official sets you can (almost) build with them.

Data Science group project, FH Dortmund, winter semester 2026/27.

> **Rule of the project:** no ready-made tools or models that recognise LEGO
> for us. Recognition is built and trained by the team.

## How it works

```
Photo → Find bricks → Shape (CNN) + Colour → Inventory → "What can I build?"
```

| Step | Approach | Status |
|---|---|---|
| Capture | Fixed camera, even light, grey card in every photo | planned |
| Find bricks | OpenCV contours for separated bricks; YOLO-seg for piles (stretch goal) | planned |
| Shape | CNN on greyscale cut-outs, trained on renders, fine-tuned on real photos | v0: 94.8 % top-1 on renders, real photos pending |
| Colour | White balance via grey card, nearest measured LEGO colour | planned |
| Inventory | Part, colour, count + confidence | planned |
| Sets | Coverage and missing parts per set, using Rebrickable part lists | planned |

**Core experiment:** how much accuracy do we lose going from rendered
training images to real photos, and how do we win it back?

## Setup

Requires Python 3.10+ and Git.

```bash
git clone https://github.com/BluePineappleCake/lego-recognition-fhdo.git
cd lego-recognition-fhdo
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Optional extras: `.[train]` for PyTorch/timm/W&B (preinstalled on Kaggle),
`.[app]` for the Streamlit app.

> **Training locally with an NVIDIA GPU:** install PyTorch with CUDA **before**
> `pip install -e ".[dev,train]"`, using the command from
> [pytorch.org](https://pytorch.org) (Get Started). `pip install torch` from PyPI
> is CPU-only on Windows. Check with
> `python -c "import torch; print(torch.cuda.is_available())"`.

## Data

Data is **not** stored in this repository.

- **Renders (training):** [Images of LEGO Bricks](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images)
  by J. Hazelzet, 40,000 images, 50 part types
- **Real photos (fine-tuning + test):** our own photos, private Kaggle dataset
- **Splits:** `data/splits/*.csv` lists which photo belongs to which split.
  The test split is frozen: never train or tune on it.

See `data/README.md` for where to put files locally.

## Training the shape classifier

```bash
wandb login                                   # once; runs go to the team "lego-fhdo"
set WANDB_ENTITY=lego-fhdo                    # PowerShell: $env:WANDB_ENTITY="lego-fhdo"

python scripts/train_shape.py --config configs/shape_baseline.yaml --sanity   # loss -> ~0
python scripts/train_shape.py --config configs/shape_baseline.yaml
python scripts/train_shape.py --config configs/shape_size_feature.yaml --seed 1
```

Renders are read from `data/raw/renders/dataset/` (or `--data <folder>`, the
folder that directly contains the 40,000 PNGs). The first run preprocesses
everything once into `data/processed/`. Checkpoints go to `models/` and to W&B.

Results and conclusions so far: [`docs/experiments.md`](docs/experiments.md).

Rules: one change per experiment, compare over several seeds, never tune on
the real-photo test set.

## Repository structure

```
src/legorec/     reusable code (segmentation, shape, colour, inventory, sets)
configs/         one YAML per experiment
data/splits/     versioned train/test file lists
notebooks/       exploration only, named NN_initials_topic.ipynb
scripts/         training and evaluation entry points
app/             Streamlit app
tests/
docs/            experiment log and progress presentations
```

## Team

| Name | Main area |
|---|---|
| Togzhan | TBD |
| Ainur | TBD |
| Sami | TBD |
| David | TBD |

## Working together

- Tasks live on the GitHub project board, discussion on Discord
- Branch per task (`shape/baseline-cnn`), pull request into `main`, one review
- Code used in more than one notebook moves into `src/`
- Every experiment logs its config to Weights & Biases

## License

MIT
