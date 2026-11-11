"""Inventory and integrity checks for YOLO-format detection datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(split_dir: Path) -> list[Path]:
    """Return all image files below ``split_dir``, sorted."""
    return sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(image_path: Path) -> Path:
    """Map an image path to its YOLO label path.

    Handles both ``.../images/x.jpg -> .../labels/x.txt`` and labels
    stored next to the image.
    """
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_label_file(path: Path) -> tuple[list[tuple], int]:
    """Parse a YOLO label file.

    Returns:
        A tuple (rows, n_malformed). Rows are (cls, cx, cy, w, h) tuples.
        Lines without exactly 5 fields (for example polygons) count as
        malformed.
    """
    if not path.exists():
        return [], 0
    rows, bad = [], 0
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            bad += 1
            continue
        rows.append((int(float(fields[0])), *map(float, fields[1:])))
    return rows, bad


def build_box_table(split: str, image_paths: list[Path]) -> pd.DataFrame:
    """Build a table with one row per box, or one empty row per unlabeled image."""
    records = []
    for img in image_paths:
        with Image.open(img) as im:  # reads the header only
            w_img, h_img = im.size
        lab = label_path_for(img)
        rows, bad = parse_label_file(lab)
        base = {
            "split": split,
            "image": str(img),
            "img_w": w_img,
            "img_h": h_img,
            "label_exists": lab.exists(),
            "n_malformed": bad,
        }
        if not rows:
            records.append({**base, "cls": None})
        for cls, cx, cy, w, h in rows:
            records.append(
                {
                    **base,
                    "cls": cls,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "area_rel": w * h,
                    "px_w": w * w_img,
                    "px_h": h * h_img,
                    "aspect": (w * w_img) / max(h * h_img, 1e-9),
                }
            )
    return pd.DataFrame.from_records(records)
