"""Single-class detection metrics: AP over IoU thresholds and recall by object size."""

from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
SIZE_BINS = (
    (0.0, 0.005, "<0.5%"),
    (0.005, 0.02, "0.5-2%"),
    (0.02, 0.10, "2-10%"),
    (0.10, 1.01, ">=10%"),
)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N, 4) and b (M, 4) in xyxy format."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _match(preds, gts, iou_thr, score_thr=0.0):
    """Greedy matching per image; returns scores, tp flags and per-GT matched flags."""
    scores, tps, gt_hit = [], [], []
    for (pb, ps), gb in zip(preds, gts):
        keep = np.asarray(ps) >= score_thr
        pb, ps = np.asarray(pb).reshape(-1, 4)[keep], np.asarray(ps)[keep]
        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]
        ious = iou_matrix(pb, gb)
        taken = np.zeros(len(gb), dtype=bool)
        for i in range(len(pb)):
            tp = False
            if len(gb):
                cand = np.where(~taken, ious[i], -1.0)
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    taken[j] = True
                    tp = True
            scores.append(ps[i])
            tps.append(tp)
        gt_hit.append(taken)
    return np.array(scores), np.array(tps, dtype=bool), gt_hit


def average_precision(preds, gts, iou_thr: float = 0.5) -> float:
    """COCO-style 101-point AP for one class."""
    n_gt = sum(len(g) for g in gts)
    scores, tps, _ = _match(preds, gts, iou_thr)
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[order])
    fp = np.cumsum(~tps[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    vals = np.where(
        idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0
    )
    return float(vals.mean())


def evaluate(preds, gts, image_areas, score_thr: float = 0.25) -> dict:
    """AP50, AP75, AP50-95, precision, recall and recall by size bin.

    Args:
        preds: per image, a tuple (boxes (N, 4) xyxy in pixels, scores (N,)).
        gts: per image, ground-truth boxes (M, 4) xyxy in pixels.
        image_areas: per image, width * height in pixels (for size bins).
        score_thr: confidence threshold for precision, recall and size-bin recall.
    """
    gts = [np.asarray(g, dtype=np.float64).reshape(-1, 4) for g in gts]
    aps = [average_precision(preds, gts, t) for t in IOU_THRESHOLDS]
    scores, tps, gt_hit = _match(preds, gts, 0.5, score_thr)
    n_gt = sum(len(g) for g in gts)
    n_hit = int(sum(h.sum() for h in gt_hit))
    out = {
        "n_images": len(gts),
        "n_gt": n_gt,
        "AP50": round(aps[0], 4),
        "AP75": round(aps[5], 4),
        "AP50-95": round(float(np.mean(aps)), 4),
        "precision": round(float(tps.sum()) / max(len(tps), 1), 4),
        "recall": round(n_hit / max(n_gt, 1), 4),
        "score_thr": score_thr,
    }
    max_rec = sum(int(h.sum()) for h in _match(preds, gts, 0.5, 0.0)[2]) / max(n_gt, 1)
    out["recall_any_conf"] = round(max_rec, 4)
    frac = (
        np.concatenate(
            [
                (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) / a
                for g, a in zip(gts, image_areas)
            ]
        )
        if n_gt
        else np.zeros(0)
    )
    hit = np.concatenate([h for h in gt_hit]) if n_gt else np.zeros(0, dtype=bool)
    for lo, hi, name in SIZE_BINS:
        m = (frac >= lo) & (frac < hi)
        out["recall " + name] = (
            round(float(hit[m].mean()), 3) if m.any() else None,
            int(m.sum()),
        )
    return out
