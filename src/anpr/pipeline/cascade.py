"""End-to-end cascade: vehicles, tracks, plates, OCR and per-track consensus.

Every model is injected as a plain callable, so the orchestration can be tested
without weights or a GPU and each stage can be swapped (YOLO or RF-DETR for the
plate detector, TensorRT engines on Jetson Orin).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from anpr.pipeline.association import associate
from anpr.pipeline.consensus import Consensus, TrackReads, decide
from anpr.pipeline.vehicle import Detections
from anpr.track.bytetrack import ByteTracker

VehicleFn = Callable[[np.ndarray], Detections]
PlateFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
OcrFn = Callable[[np.ndarray], tuple[str, float]]


def sharpness(gray: np.ndarray) -> float:
    """Variance of a discrete Laplacian, a cheap blur measure (higher is sharper)."""
    g = np.asarray(gray, dtype=float)
    if g.ndim == 3:
        g = g.mean(axis=2)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


@dataclass(frozen=True)
class QualityGate:
    """Reject plate crops that are too small or too blurry to read.

    Attributes:
        min_width: Minimum crop width in pixels.
        min_height: Minimum crop height in pixels.
        min_sharpness: Minimum Laplacian variance.
    """

    min_width: int = 40
    min_height: int = 12
    min_sharpness: float = 20.0

    def passes(self, crop: np.ndarray) -> bool:
        """Return True if the crop is worth sending to OCR."""
        h, w = crop.shape[:2]
        if w < self.min_width or h < self.min_height:
            return False
        return sharpness(crop) >= self.min_sharpness


@dataclass
class TrackInfo:
    """Lifetime bookkeeping for one vehicle track."""

    track_id: int
    cls: int
    first_frame: int
    last_frame: int
    first_ts: float
    last_ts: float
    best_plate_box: np.ndarray | None = None
    best_plate_score: float = 0.0
    emitted_text: str | None = None


@dataclass(frozen=True)
class TrackSummary:
    """A finished or settled track together with its consensus plate."""

    info: TrackInfo
    consensus: Consensus | None
    final: bool


@dataclass(frozen=True)
class FrameResult:
    """What the cascade saw for one tracked vehicle in one frame."""

    track_id: int
    vehicle_box: np.ndarray
    vehicle_cls: int
    plate_box: np.ndarray | None
    text: str | None
    conf: float | None


class Cascade:
    """Run the plate-reading cascade frame by frame.

    Args:
        vehicle_fn: Frame to vehicle detections.
        plate_fn: Vehicle crop to plate ``(boxes, scores)`` in crop coordinates.
        ocr_fn: Plate crop to ``(text, confidence)``.
        tracker: Multi-object tracker; a default one is created if omitted.
        gate: Quality gate applied before OCR.
        plate_min_score: Minimum plate detector confidence.
        settle_reads: A track with this many reads and a confident valid
            consensus is no longer re-read, which saves Orin compute.
        settle_support: Consensus support needed to settle a track.
    """

    def __init__(
        self,
        vehicle_fn: VehicleFn,
        plate_fn: PlateFn,
        ocr_fn: OcrFn,
        tracker: ByteTracker | None = None,
        gate: QualityGate | None = None,
        plate_min_score: float = 0.3,
        settle_reads: int = 5,
        settle_support: float = 0.8,
    ) -> None:
        self.vehicle_fn = vehicle_fn
        self.plate_fn = plate_fn
        self.ocr_fn = ocr_fn
        self.tracker = tracker or ByteTracker()
        self.gate = gate or QualityGate()
        self.plate_min_score = plate_min_score
        self.settle_reads = settle_reads
        self.settle_support = settle_support
        self.reads: dict[int, TrackReads] = {}
        self.vehicle_cls: dict[int, int] = {}
        self.tracks: dict[int, TrackInfo] = {}
        self._announced: set[int] = set()

    def _settled(self, track_id: int) -> bool:
        tr = self.reads.get(track_id)
        if tr is None or len(tr.reads) < self.settle_reads:
            return False
        d = decide(tr)
        return bool(d and d.valid and d.score >= self.settle_support)

    def process(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        ts: float = 0.0,
        det: Detections | None = None,
    ) -> list[FrameResult]:
        """Process one frame and return a result per confirmed vehicle track.

        Args:
            frame: BGR image.
            frame_idx: Frame sequence number.
            ts: Frame timestamp in seconds.
            det: Vehicle detections already computed for this frame, for example
                by a batched call across cameras. When None the cascade runs
                its own vehicle detector.
        """
        if det is None:
            det = self.vehicle_fn(frame)
        tracks = self.tracker.update(det.boxes, det.scores, det.classes)
        h, w = frame.shape[:2]
        results: list[FrameResult] = []
        for t in tracks:
            self.vehicle_cls[t.track_id] = t.cls
            info = self.tracks.get(t.track_id)
            if info is None:
                info = TrackInfo(t.track_id, t.cls, frame_idx, frame_idx, ts, ts)
                self.tracks[t.track_id] = info
            info.last_frame, info.last_ts = frame_idx, ts
            vb = np.clip(t.box, 0, [w, h, w, h])
            plate_box, text, conf = None, None, None
            if not self._settled(t.track_id):
                x1, y1, x2, y2 = (int(round(v)) for v in vb)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    pb, ps = self.plate_fn(crop)
                    keep = np.asarray(ps) >= self.plate_min_score
                    pb, ps = np.asarray(pb).reshape(-1, 4)[keep], np.asarray(ps)[keep]
                    own = associate(pb, ps, np.array([[0, 0, x2 - x1, y2 - y1]]))
                    if 0 in own:
                        local = pb[own[0]]
                        plate_box = local + np.array([x1, y1, x1, y1])
                        if ps[own[0]] >= info.best_plate_score:
                            info.best_plate_score = float(ps[own[0]])
                            info.best_plate_box = plate_box
                        px1, py1, px2, py2 = (int(round(v)) for v in local)
                        pcrop = crop[py1:py2, px1:px2]
                        if pcrop.size and self.gate.passes(pcrop):
                            text, conf = self.ocr_fn(pcrop)
                            self.reads.setdefault(t.track_id, TrackReads()).add(
                                text, conf, frame_idx
                            )
            results.append(FrameResult(t.track_id, vb, t.cls, plate_box, text, conf))
        return results

    def finalize(self) -> dict[int, Consensus | None]:
        """Return the consensus plate for every track that produced reads."""
        return {tid: decide(tr) for tid, tr in self.reads.items()}

    def newly_settled(self) -> list[TrackSummary]:
        """Tracks that just became confident enough to report (once each)."""
        out = []
        for tid, info in self.tracks.items():
            if tid in self._announced or not self._settled(tid):
                continue
            self._announced.add(tid)
            d = decide(self.reads[tid])
            info.emitted_text = d.text if d else None
            out.append(TrackSummary(info, d, final=False))
        return out

    def drain(
        self, frame_idx: int, max_gap: int = 45, flush: bool = False
    ) -> list[TrackSummary]:
        """Finish tracks that have not been seen for ``max_gap`` frames.

        Args:
            frame_idx: Index of the current frame.
            max_gap: Frames without a sighting after which a track is closed.
            flush: Close every open track (use at end of stream).

        Returns:
            One final summary per closed track. A track that was already
            announced with the same text is not reported again, unless it has
            no plate, in which case the vehicle itself is still reported.
        """
        done = [
            tid
            for tid, info in self.tracks.items()
            if flush or frame_idx - info.last_frame > max_gap
        ]
        out = []
        for tid in done:
            info = self.tracks.pop(tid)
            tr = self.reads.pop(tid, None)
            d = decide(tr) if tr else None
            was_announced = tid in self._announced
            self._announced.discard(tid)
            if was_announced and d and d.text == info.emitted_text:
                continue
            out.append(TrackSummary(info, d, final=True))
        return out
