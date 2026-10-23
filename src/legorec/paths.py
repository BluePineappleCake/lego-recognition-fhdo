"""Where data lives. Override with environment variables, never hard-code paths.

LEGOREC_DATA       root of the (unversioned) data folder, default: <repo>/data
LEGOREC_PROCESSED  writable folder for caches, default: $LEGOREC_DATA/processed
                   (on Kaggle /kaggle/input is read-only -> set it to /kaggle/working)
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# versioned in Git, therefore always relative to the repository
SPLITS_DIR = REPO_ROOT / "data" / "splits"
MODELS_DIR = REPO_ROOT / "models"


def data_dir() -> Path:
    return Path(os.environ.get("LEGOREC_DATA", REPO_ROOT / "data"))


def renders_dir() -> Path:
    """Folder that directly contains the 40,000 render PNGs."""
    return data_dir() / "raw" / "renders" / "dataset"


def processed_dir() -> Path:
    return Path(os.environ.get("LEGOREC_PROCESSED", data_dir() / "processed"))
