"""Plate events emitted by the streaming pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from anpr.ocr import grammar
from anpr.pipeline.cascade import TrackSummary
from anpr.pipeline.vehicle import COCO_VEHICLES


@dataclass(frozen=True)
class PlateEvent:
    """One plate decision for one vehicle.

    Attributes:
        camera_id: Identifier of the source camera.
        track_id: Vehicle track id (unique within one run and camera).
        event: ``"confirmed"`` (reported as soon as the read is confident) or
            ``"final"`` (reported when the vehicle leaves the scene).
        vehicle_class: ``car``, ``motorcycle``, ``bus``, ``truck`` or ``other``.
        plate: Plate string, or None when no readable plate was found.
        confidence: Consensus support in ``[0, 1]``.
        valid: True if the plate fits the Indian plate grammar. False means
            the read should be reviewed, not trusted.
        n_reads: Number of frames that contributed to the decision.
        first_seen: Timestamp of the first frame of the track (seconds).
        last_seen: Timestamp of the last frame of the track (seconds).
        state: Two-letter state or union-territory code, when valid.
        plate_box: Best plate box ``[x1, y1, x2, y2]`` in frame pixels.
    """

    camera_id: str
    track_id: int
    event: str
    vehicle_class: str
    plate: str | None
    confidence: float
    valid: bool
    n_reads: int
    first_seen: float
    last_seen: float
    state: str | None = None
    plate_box: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict:
        """Plain-dict form for JSON and SQL sinks."""
        return asdict(self)


def to_event(summary: TrackSummary, camera_id: str) -> PlateEvent:
    """Convert a cascade ``TrackSummary`` into a ``PlateEvent``."""
    info, cons = summary.info, summary.consensus
    fields = grammar.split_fields(cons.text) if cons and cons.valid else None
    box = info.best_plate_box
    return PlateEvent(
        camera_id=camera_id,
        track_id=info.track_id,
        event="final" if summary.final else "confirmed",
        vehicle_class=COCO_VEHICLES.get(info.cls, "other"),
        plate=cons.text if cons else None,
        confidence=round(cons.score, 4) if cons else 0.0,
        valid=bool(cons and cons.valid),
        n_reads=cons.n_reads if cons else 0,
        first_seen=info.first_ts,
        last_seen=info.last_ts,
        state=fields["state"] if fields else None,
        plate_box=tuple(round(float(v), 1) for v in box) if box is not None else None,
    )
