"""Where data lives. Override with environment variables, never hard-code paths.

LEGOREC_DATA       root of the (unversioned) data folder, default: <repo>/data
LEGOREC_PROCESSED  writable folder for caches, default: $LEGOREC_DATA/processed
                   (on Kaggle /kaggle/input is read-only -> set it to /kaggle/working)
LEGOREC_SHAPE_MODEL  checkpoint used by legorec.shape.predict,
                   default: <repo>/models/shape.pt
LEGOREC_BRICK_MODEL  YOLO-seg model that finds the bricks (legorec.segment),
                   default: <repo>/models/bricks_yolo.pt
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# versioned in Git, therefore always relative to the repository
SPLITS_DIR = REPO_ROOT / "data" / "splits"
LABELS_DIR = REPO_ROOT / "data" / "labels"  # labels corrected by hand
MODELS_DIR = REPO_ROOT / "models"


def data_dir() -> Path:
    return Path(os.environ.get("LEGOREC_DATA", REPO_ROOT / "data"))


def renders_dir() -> Path:
    """Folder that directly contains the 40,000 render PNGs."""
    return data_dir() / "raw" / "renders" / "dataset"


def own_crops_dir() -> Path:
    """Single-brick crops of our sheet photos, checked by hand (eval_own.py)."""
    return data_dir() / "raw" / "own_crops"


def own_photos_dir() -> Path:
    """Our own sheet photos, one subfolder per part (see data/README.md)."""
    return data_dir() / "raw" / "own_photos"


def internet_dir() -> Path:
    """Folder of the internet dataset; contains photos/ (and renders/)."""
    return data_dir() / "raw" / "internet"


def processed_dir() -> Path:
    return Path(os.environ.get("LEGOREC_PROCESSED", data_dir() / "processed"))


def shape_model_path() -> Path:
    """Checkpoint of the shape classifier that the pipeline uses."""
    return Path(os.environ.get("LEGOREC_SHAPE_MODEL", MODELS_DIR / "shape.pt"))


def brick_model_path() -> Path:
    """YOLO-seg model that finds the bricks on a sheet photo."""
    return Path(os.environ.get("LEGOREC_BRICK_MODEL", MODELS_DIR / "bricks_yolo.pt"))
