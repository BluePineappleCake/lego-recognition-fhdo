"""Correct YOLO labels by hand in CVAT, without losing them on the next export.

    python scripts/cvat_labels.py pack                     # photos of review.csv
    python scripts/cvat_labels.py pack 3001_IMG_3481 3004_IMG_8877
    python scripts/cvat_labels.py unpack path/to/task_export.zip

pack writes to <data>/cvat/:

    images/        the photos: upload these when you create the CVAT task
    upload.zip     the current labels: in the task "Upload annotations",
                   format "Ultralytics YOLO Segmentation 1.0"

unpack reads the zip that CVAT exports ("Export task dataset", same format,
images not needed) and stores every label file in FIXED_DIR
(data/labels/sheets/<photo>.txt). That folder is versioned: commit the
files, so the whole team gets the corrections. export_yolo.py takes a label
from there instead of the automatic one, so corrections survive a new
export. unpack also copies them into the current export, so you can train
right away.
"""

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

from legorec.paths import LABELS_DIR, processed_dir
from legorec.segment.yolo import NAMES

FIXED_DIR = LABELS_DIR / "sheets"  # versioned in Git, see data/README.md


def pack(data: Path, names: list[str]) -> None:
    images = {p.stem: p for p in data.glob("images/*/*.jpg")}
    if not names:
        with open(data / "review.csv", newline="", encoding="utf-8") as file:
            names = [row["photo"] for row in csv.DictReader(file)]
    unknown = [n for n in names if n not in images]
    if unknown:
        raise SystemExit(f"not in {data}/images: {unknown}")

    out = data / "cvat"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    yaml = "path: ./\ntrain: train.txt\nnames:\n" + "".join(
        f"  {k}: {v}\n" for k, v in NAMES.items()
    )
    with zipfile.ZipFile(out / "upload.zip", "w") as archive:
        archive.writestr("data.yaml", yaml)
        archive.writestr("train.txt", "".join(f"images/train/{n}.jpg\n" for n in names))
        for name in names:
            image = images[name]
            shutil.copy(image, out / "images" / image.name)
            label = data / "labels" / image.parent.name / f"{name}.txt"
            archive.write(label, f"labels/train/{name}.txt")
    print(f"[pack] {len(names)} photos -> {out}")
    print(
        "  1. CVAT: create a task, label 'brick' (polygon), upload the files in images/"
    )
    print(
        "  2. task -> Actions -> Upload annotations -> Ultralytics YOLO Segmentation 1.0"
    )
    print(f"     -> {out / 'upload.zip'}")


def unpack(data: Path, archive_path: Path) -> None:
    labels = {p.stem: p for p in data.glob("labels/*/*.txt")}
    FIXED_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            path = Path(member)
            if path.suffix != ".txt" or "labels" not in path.parts:
                continue
            text = archive.read(member).decode("utf-8")
            (FIXED_DIR / path.name).write_text(text, encoding="utf-8")
            if path.stem in labels:  # the current export, to train right away
                labels[path.stem].write_text(text, encoding="utf-8")
            n += 1
    print(f"[unpack] {n} corrected label files -> {FIXED_DIR} (and the current export)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pack", "unpack"])
    parser.add_argument("args", nargs="*", help="pack: photo names; unpack: CVAT zip")
    parser.add_argument("--data", default=str(processed_dir() / "yolo_bricks"))
    options = parser.parse_args()
    data = Path(options.data)
    if options.action == "pack":
        pack(data, options.args)
    else:
        if len(options.args) != 1:
            raise SystemExit("unpack needs exactly one zip file")
        unpack(data, Path(options.args[0]))


if __name__ == "__main__":
    main()
