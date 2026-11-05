"""
Class overview as a PDF table with pictures: one row per class with part number,
name, three cropped renders (different poses) and a "watch out" note.

    python -m pip install reportlab
    python scripts/class_overview_pdf.py --data C:\\lego\\data --out docs/figures/classes_table.pdf

Needs class_overview.py in the same folder (index, crops, grouping are reused).
"""

import argparse
import io
import random
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).parent))
from class_overview import (
    TYPE_ORDER,
    crop_to_brick,
    index,
    part_key,
    part_type,
    pick_views,
)

NOTES = {
    # bricks
    "3005": "length series 1×1 / 1×2 / 1×3 / 1×4",
    "3004": "length series",
    "3622": "length series",
    "3010": "length series",
    "3003": "length series 2×2 / 2×3 / 2×4",
    "3002": "length series",
    "3001": "length series",
    "3659": "bow, length pair with 4490",
    "4490": "bow, length pair with 3659",
    "2357": "corner: same footprint as 2420 (plate), 14719 (tile)",
    # plates
    "3024": "length series 1×1 / 1×2 / 1×3",
    "3023": "length series; same footprint as 3794, 3069",
    "3623": "length series",
    "3022": "length series 2×2 / 2×3 / 2×4; look-alike 33909",
    "3021": "length series",
    "3020": "length series",
    "3794": "jumper; same footprint as 3023, 3069",
    "33909": "look-alike 3022 (largest remaining pair in v0)",
    "2420": "corner: same footprint as 2357 (brick), 14719 (tile)",
    # tiles
    "3069": "same footprint as 3023, 3794; length pair with 3070",
    "3070": "length pair with 3069",
    "4150": "SAME NAME as 27925",
    "27925": "SAME NAME as 4150",
    "14719": "corner: same footprint as 2357 (brick), 2420 (plate)",
    # slopes
    "3038": "SAME NAME as 3298",
    "3298": "SAME NAME as 3038",
    "15672": "SAME NAME as 85984",
    "85984": "SAME NAME as 15672",
    "3045": "outside corner, pair with 3046",
    "3046": "inside corner, pair with 3045",
    "3675": "outside 3×3, pair with 99301",
    "99301": "inside 3×3, pair with 3675",
    # technic
    "6632": "length pair with 41677",
    "41677": "length pair with 6632",
    "18654": "length pair with 43857 (1M / 2M)",
    "43857": "length pair with 18654",
}

THUMB = 110  # px per render in the strip


def strip(paths):
    """Cropped renders side by side on white, as an in-memory PNG."""
    out = PILImage.new(
        "RGB", (THUMB * len(paths) + 6 * (len(paths) - 1), THUMB), "white"
    )
    for i, p in enumerate(paths):
        im = PILImage.fromarray(crop_to_brick(PILImage.open(p))).resize(
            (THUMB, THUMB), PILImage.LANCZOS
        )
        out.paste(im, (i * (THUMB + 6), 0))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    buf.seek(0)
    return buf, out.size


def clean(name):
    n = re.sub(r"(\d)[xX](?=\d)", "\\1×", name)
    return n[0].upper() + n[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="classes_table.pdf")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()
    rng = random.Random(cfg.seed)

    classes = index(cfg.data)
    dup = Counter(c["name"].lower() for c in classes.values())
    same_name = {p for p, c in classes.items() if dup[c["name"].lower()] > 1}
    n_img = sum(len(v) for c in classes.values() for v in c["poses"].values())

    ss = getSampleStyleSheet()
    title = ParagraphStyle(
        "t", parent=ss["Title"], fontSize=18, alignment=0, spaceAfter=2
    )
    sub = ParagraphStyle(
        "s",
        parent=ss["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#555555"),
    )
    grp = ParagraphStyle(
        "g",
        parent=ss["Heading3"],
        fontSize=11.5,
        textColor=colors.HexColor("#4C72B0"),
        spaceBefore=8,
        spaceAfter=3,
    )
    cell = ParagraphStyle("c", parent=ss["Normal"], fontSize=8.5, leading=10.5)
    note = ParagraphStyle("n", parent=cell, textColor=colors.HexColor("#444444"))

    by_type = {
        t: sorted(
            (p for p in classes if part_type(classes[p]["name"]) == t), key=part_key
        )
        for t in TYPE_ORDER
    }
    used = [t for t in TYPE_ORDER if by_type[t]]
    story = [
        Paragraph("Kaggle “Images of LEGO Bricks”: the classes", title),
        Paragraph(
            f"{len(classes)} classes, {n_img:,} rendered images. Source: J. Hazelzet, Kaggle. "
            f"{cfg.views} random poses per class, each cropped to the brick (not to scale). "
            "Highlighted rows: two part numbers share the same name in the dataset.",
            sub,
        ),
        Spacer(1, 4),
    ]
    summary = [
        used + ["Total"],
        [str(len(by_type[t])) for t in used] + [str(len(classes))],
    ]
    st = Table(summary, colWidths=[174 * mm / len(summary[0])] * len(summary[0]))
    st.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, 1), "Helvetica", 13),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F8")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
                ("TOPPADDING", (0, 1), (-1, 1), 5),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
            ]
        )
    )
    story.append(st)

    img_w = 72 * mm
    for t in used:
        parts = by_type[t]
        rows = [["Part", "Name", "Renders", "Watch out"]]
        style = [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F8")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
        for i, p in enumerate(parts, 1):
            buf, (w, h) = strip(pick_views(classes[p]["poses"], cfg.views, rng))
            n = NOTES.get(p, "")
            if p in same_name:
                style.append(
                    ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FDF0E3"))
                )
            rows.append(
                [
                    Paragraph(f"<b>{p}</b>", cell),
                    Paragraph(clean(classes[p]["name"]), cell),
                    Image(buf, width=img_w, height=img_w * h / w),
                    Paragraph(n, note),
                ]
            )
        tb = Table(
            rows, colWidths=[16 * mm, 38 * mm, img_w + 4 * mm, 48 * mm], repeatRows=1
        )
        tb.setStyle(TableStyle(style))
        story += [Paragraph(f"{t}  ({len(parts)})", grp), tb]

    def footer(c, d):
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.grey)
        c.drawString(
            18 * mm,
            10 * mm,
            "LEGO Brick Recognition, Gruppe 2, FH Dortmund WiSe 2026/27",
        )
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page))

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        cfg.out,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title="Kaggle LEGO dataset: classes",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"{len(classes)} classes -> {cfg.out}")


if __name__ == "__main__":
    main()
