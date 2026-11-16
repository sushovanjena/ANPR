"""Extract plate crops from YOLO-labelled images for OCR work."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd


def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    pad: float = 0.0,
) -> tuple[int, int, int, int]:
    """Convert a normalised YOLO box to clipped pixel corners.

    Args:
        cx: Box centre x as a fraction of image width.
        cy: Box centre y as a fraction of image height.
        w: Box width as a fraction of image width.
        h: Box height as a fraction of image height.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        pad: Extra margin on each side, as a fraction of the box size.

    Returns:
        Integer corners (x1, y1, x2, y2) clipped to the image.
    """
    bw, bh = w * img_w, h * img_h
    x1 = cx * img_w - bw / 2 - pad * bw
    x2 = cx * img_w + bw / 2 + pad * bw
    y1 = cy * img_h - bh / 2 - pad * bh
    y2 = cy * img_h + bh / 2 + pad * bh
    return (
        max(0, round(x1)),
        max(0, round(y1)),
        min(img_w, round(x2)),
        min(img_h, round(y2)),
    )


def extract_crops(
    boxes: pd.DataFrame, out_dir: Path, pad: float = 0.04
) -> pd.DataFrame:
    """Save one PNG per plate box and return the table with crop paths.

    Args:
        boxes: One row per box with columns image, cx, cy, w, h, img_w, img_h.
        out_dir: Folder that receives the crops.
        pad: Relative margin added around each box.

    Returns:
        Copy of the table with crop_id and crop_path columns. Boxes whose crop
        would be empty or unreadable are dropped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (_, r) in enumerate(boxes.reset_index(drop=True).iterrows()):
        im = cv2.imread(str(r.image))
        if im is None:
            continue
        x1, y1, x2, y2 = yolo_to_xyxy(
            r.cx, r.cy, r.w, r.h, int(r.img_w), int(r.img_h), pad
        )
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop_id = f"p{k:05d}"
        path = out_dir / f"{crop_id}.png"
        cv2.imwrite(str(path), im[y1:y2, x1:x2])
        rows.append({**r.to_dict(), "crop_id": crop_id, "crop_path": str(path)})
    return pd.DataFrame(rows)
