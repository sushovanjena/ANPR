"""Lightweight ByteTrack-style multi-object tracker (numpy and scipy only).

Follows the two-stage association of ByteTrack (Zhang et al., ECCV 2022):
high-confidence detections are matched to existing tracks first, then the
remaining tracks get a second chance against low-confidence detections, which
recovers vehicles that are briefly occluded or blurred. Motion is predicted
with a constant-velocity Kalman filter on ``(cx, cy, w, h)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N, 4)`` and ``(M, 4)`` xyxy box arrays."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _xyxy_to_state(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)


def _state_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class _Kalman:
    """Constant-velocity Kalman filter over ``(cx, cy, w, h)`` and velocities."""

    def __init__(self, box: np.ndarray) -> None:
        self.x = np.concatenate([_xyxy_to_state(box), np.zeros(4)])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0, 1000.0])
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01])
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _state_to_xyxy(self.x)

    def update(self, box: np.ndarray) -> None:
        z = _xyxy_to_state(box)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(8) - k @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return _state_to_xyxy(self.x)


@dataclass
class Track:
    """A tracked object with a stable integer ``track_id``."""

    track_id: int
    kalman: _Kalman
    score: float
    cls: int
    hits: int = 1
    misses: int = 0
    age: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def box(self) -> np.ndarray:
        """Current xyxy box estimate."""
        return self.kalman.box


class ByteTracker:
    """Multi-object tracker with two-stage (high then low score) matching.

    Args:
        high_thr: Detections at or above this score start tracks and are matched
            first.
        low_thr: Detections between ``low_thr`` and ``high_thr`` are used only in
            the second matching stage.
        match_iou: Minimum IoU for a first-stage match.
        low_match_iou: Minimum IoU for a second-stage match.
        max_age: Frames a track may go unmatched before it is dropped.
        min_hits: Matches before a track is reported as confirmed.
    """

    def __init__(
        self,
        high_thr: float = 0.5,
        low_thr: float = 0.1,
        match_iou: float = 0.3,
        low_match_iou: float = 0.5,
        max_age: int = 30,
        min_hits: int = 2,
    ) -> None:
        self.high_thr = high_thr
        self.low_thr = low_thr
        self.match_iou = match_iou
        self.low_match_iou = low_match_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1

    @staticmethod
    def _assign(
        tracks: list[Track], boxes: np.ndarray, min_iou: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian matching on IoU. Returns matches and unmatched indices."""
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        iou = iou_matrix(np.stack([t.box for t in tracks]), boxes)
        rows, cols = linear_sum_assignment(-iou)
        matches = [
            (r, c) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou
        ]
        matched_r = {r for r, _ in matches}
        matched_c = {c for _, c in matches}
        un_t = [i for i in range(len(tracks)) if i not in matched_r]
        un_d = [j for j in range(len(boxes)) if j not in matched_c]
        return matches, un_t, un_d

    def update(
        self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray | None = None
    ) -> list[Track]:
        """Advance one frame.

        Args:
            boxes: ``(N, 4)`` xyxy detections.
            scores: ``(N,)`` confidences.
            classes: Optional ``(N,)`` class ids carried on the track.

        Returns:
            Confirmed tracks that were matched in this frame.
        """
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        classes = (
            np.zeros(len(boxes), dtype=int)
            if classes is None
            else np.asarray(classes, dtype=int).reshape(-1)
        )
        for t in self.tracks:
            t.kalman.predict()
            t.age += 1
            t.misses += 1

        high = scores >= self.high_thr
        low = (scores >= self.low_thr) & ~high
        hi_idx, lo_idx = np.flatnonzero(high), np.flatnonzero(low)

        matches, un_t, un_d = self._assign(self.tracks, boxes[hi_idx], self.match_iou)
        matched_now: set[int] = set()
        for ti, di in matches:
            self._apply(self.tracks[ti], boxes[hi_idx[di]], scores[hi_idx[di]])
            matched_now.add(ti)

        remaining = [self.tracks[i] for i in un_t]
        matches2, _, _ = self._assign(remaining, boxes[lo_idx], self.low_match_iou)
        for ti, di in matches2:
            self._apply(remaining[ti], boxes[lo_idx[di]], scores[lo_idx[di]])
            matched_now.add(self.tracks.index(remaining[ti]))

        for di in un_d:
            k = hi_idx[di]
            trk = Track(
                self._next_id, _Kalman(boxes[k]), float(scores[k]), int(classes[k])
            )
            trk.misses = 0
            self._next_id += 1
            self.tracks.append(trk)
            matched_now.add(len(self.tracks) - 1)

        out = [
            t
            for i, t in enumerate(self.tracks)
            if i in matched_now and t.hits >= self.min_hits
        ]
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return out

    @staticmethod
    def _apply(track: Track, box: np.ndarray, score: float) -> None:
        track.kalman.update(box)
        track.score = float(score)
        track.hits += 1
        track.misses = 0
