"""Associate plate detections with the vehicles that carry them."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def containment(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Fraction of each ``inner`` box that lies inside each ``outer`` box.

    Args:
        inner: ``(N, 4)`` xyxy boxes (plates).
        outer: ``(M, 4)`` xyxy boxes (vehicles).

    Returns:
        ``(N, M)`` array with values in ``[0, 1]``.
    """
    inner = np.asarray(inner, dtype=float).reshape(-1, 4)
    outer = np.asarray(outer, dtype=float).reshape(-1, 4)
    if len(inner) == 0 or len(outer) == 0:
        return np.zeros((len(inner), len(outer)))
    lt = np.maximum(inner[:, None, :2], outer[None, :, :2])
    rb = np.minimum(inner[:, None, 2:], outer[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = (inner[:, 2] - inner[:, 0]) * (inner[:, 3] - inner[:, 1])
    return inter / np.maximum(area[:, None], 1e-9)


def _plausibility(
    plate: np.ndarray,
    vehicle: np.ndarray,
    area_range: tuple[float, float] = (0.002, 0.25),
) -> np.ndarray:
    """Soft prior in ``[0, 1]`` on plate size and position relative to a vehicle.

    Plates cover a small, bounded fraction of the vehicle box (the range is
    wide because a two-wheeler's plate is a larger share of its box than a
    bus's). They also sit in the lower half and near the horizontal centre of
    the box. Each factor decays smoothly rather than cutting off, so a
    mis-sized box lowers the match score without discarding the plate.

    Args:
        plate: ``(N, 4)`` xyxy plate boxes.
        vehicle: ``(M, 4)`` xyxy vehicle boxes.
        area_range: Plausible plate-to-vehicle area ratio.

    Returns:
        ``(N, M)`` prior values.
    """
    pw = np.clip(plate[:, 2] - plate[:, 0], 1e-6, None)
    ph = np.clip(plate[:, 3] - plate[:, 1], 1e-6, None)
    vw = np.clip(vehicle[:, 2] - vehicle[:, 0], 1e-6, None)
    vh = np.clip(vehicle[:, 3] - vehicle[:, 1], 1e-6, None)
    ratio = (pw * ph)[:, None] / (vw * vh)[None, :]
    lo, hi = area_range
    size = np.where(
        ratio < lo,
        ratio / lo,
        np.where(ratio > hi, hi / np.maximum(ratio, 1e-9), 1.0),
    )
    pcx = (plate[:, 0] + plate[:, 2]) / 2
    pcy = (plate[:, 1] + plate[:, 3]) / 2
    rel_x = (pcx[:, None] - vehicle[None, :, 0]) / vw[None, :]
    rel_y = (pcy[:, None] - vehicle[None, :, 1]) / vh[None, :]
    horiz = np.exp(-(((rel_x - 0.5) / 0.35) ** 2))
    vert = np.clip((rel_y - 0.2) / 0.3, 0.0, 1.0)
    return size * (0.6 + 0.4 * horiz) * (0.6 + 0.4 * vert)


def affinity(plate_boxes: np.ndarray, vehicle_boxes: np.ndarray) -> np.ndarray:
    """Plate-to-vehicle match strength: containment times plausibility prior."""
    plate_boxes = np.asarray(plate_boxes, dtype=float).reshape(-1, 4)
    vehicle_boxes = np.asarray(vehicle_boxes, dtype=float).reshape(-1, 4)
    if len(plate_boxes) == 0 or len(vehicle_boxes) == 0:
        return np.zeros((len(plate_boxes), len(vehicle_boxes)))
    return containment(plate_boxes, vehicle_boxes) * _plausibility(
        plate_boxes, vehicle_boxes
    )


def associate(
    plate_boxes: np.ndarray,
    plate_scores: np.ndarray,
    vehicle_boxes: np.ndarray,
    min_containment: float = 0.8,
    min_affinity: float = 0.05,
) -> dict[int, int]:
    """One-to-one assignment of plates to vehicles.

    The affinity matrix (containment times a size and position prior, weighted
    by detector confidence) is solved with the Hungarian algorithm, so a plate
    that fits two overlapping vehicles goes to the one it explains best and no
    vehicle receives two plates. Plates that fit no vehicle are dropped, which
    removes stickers and signboards.

    Args:
        plate_boxes: ``(N, 4)`` xyxy plate boxes.
        plate_scores: ``(N,)`` plate confidences.
        vehicle_boxes: ``(M, 4)`` xyxy vehicle boxes.
        min_containment: Minimum fraction of the plate inside the vehicle.
        min_affinity: Minimum affinity for an assignment to be kept.

    Returns:
        Mapping ``vehicle index -> plate index``.
    """
    plate_boxes = np.asarray(plate_boxes, dtype=float).reshape(-1, 4)
    plate_scores = np.asarray(plate_scores, dtype=float).reshape(-1)
    vehicle_boxes = np.asarray(vehicle_boxes, dtype=float).reshape(-1, 4)
    if len(plate_boxes) == 0 or len(vehicle_boxes) == 0:
        return {}
    cont = containment(plate_boxes, vehicle_boxes)
    aff = affinity(plate_boxes, vehicle_boxes) * (0.5 + 0.5 * plate_scores[:, None])
    aff = np.where(cont >= min_containment, aff, 0.0)
    rows, cols = linear_sum_assignment(-aff)
    return {
        int(c): int(r)
        for r, c in zip(rows, cols, strict=True)
        if aff[r, c] >= min_affinity
    }
