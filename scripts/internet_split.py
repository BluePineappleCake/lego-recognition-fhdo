"""Index the internet photos, create (or check) data/splits/internet.csv, print an overview.

    python scripts/internet_split.py [--data <folder with photos/>]

Default folder: $LEGOREC_DATA/raw/internet. The first run creates
data/splits/internet.csv -> commit it. Later runs only load and check it:
if photos were added that are not in the file, it stops with an error.
"""

import argparse
from collections import Counter, defaultdict

from legorec.data.internet import index_photos, load_or_create_photo_split, render_parts
from legorec.paths import internet_dir

SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    photos = index_photos(args.data or internet_dir())
    photos_per_split, split_hash = load_or_create_photo_split(photos)
    known = render_parts()

    images = defaultdict(Counter)  # part -> split -> number of images
    groups = defaultdict(lambda: defaultdict(set))  # part -> split -> captures
    names = {}
    for split, photos_in_split in photos_per_split.items():
        for photo in photos_in_split:
            images[photo.part][split] += 1
            groups[photo.part][split].add(photo.group)
            names[photo.part] = photo.name

    print(f"\nsplit {split_hash}: images (captures) per split\n")
    header = f"{'part':<26}" + "".join(f"{s:>14}" for s in SPLITS) + "  in renders"
    print(header)
    print("-" * len(header))
    totals = defaultdict(Counter)
    for part in sorted(images):
        row = f"{names[part]:<26}"
        for split in SPLITS:
            n_img, n_grp = images[part][split], len(groups[part][split])
            row += f"{f'{n_img} ({n_grp})':>14}"
            subset = "known" if part in known else "unknown"
            totals[subset][split] += n_img
            totals[subset][split + "_g"] += n_grp
        print(row + ("  yes" if part in known else "  no"))

    print()
    for subset in ("known", "unknown"):
        n_parts = sum(1 for p in images if (p in known) == (subset == "known"))
        row = f"{f'{subset} parts ({n_parts})':<26}"
        for split in SPLITS:
            cell = f"{totals[subset][split]} ({totals[subset][split + '_g']})"
            row += f"{cell:>14}"
        print(row)


if __name__ == "__main__":
    main()
