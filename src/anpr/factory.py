"""Build the cascade and stream runner from a ``Config``."""

from __future__ import annotations

import numpy as np

from anpr.config import Config
from anpr.pipeline.cascade import Cascade, QualityGate
from anpr.pipeline.vehicle import VehicleDetector
from anpr.stream.metrics import Metrics
from anpr.stream.runner import StreamRunner
from anpr.stream.sinks import JsonlSink, MultiSink, SqliteSink
from anpr.stream.source import FrameGrabber, LatestQueue
from anpr.track.bytetrack import ByteTracker


def _no_plates(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 4)), np.zeros(0)


def build_cascade(cfg: Config) -> Cascade:
    """Create the cascade with real models according to ``cfg``.

    Weights listed in ``weights/manifest.json`` are downloaded on first use when
    ``auto_fetch_weights`` is true. With ``plate.enabled`` false the plate
    detector and OCR are not loaded, so the vehicle detector and tracker can be
    tested before plate weights exist.
    """
    vehicle = VehicleDetector(cfg.vehicle.weights, cfg.vehicle.imgsz, cfg.vehicle.conf)
    if cfg.plate.enabled:
        from anpr import weights
        from anpr.models.plate_detector import PlateDetector
        from anpr.ocr.recognizer import PlateReader, RepSvtrRecognizer

        auto = cfg.auto_fetch_weights
        plate_w = weights.resolve(cfg.plate.weights, auto)
        ocr_w = weights.resolve(cfg.ocr.weights, auto)
        dict_w = weights.resolve(cfg.ocr.dict_path, auto)
        openocr = (
            weights.ensure_openocr(cfg.ocr.openocr_dir) if auto else cfg.ocr.openocr_dir
        )
        plate_fn = PlateDetector(str(plate_w), cfg.plate.kind, cfg.plate.conf)
        ocr_fn = PlateReader(
            RepSvtrRecognizer(str(openocr), str(ocr_w), str(dict_w)),
            cfg.ocr.two_line_aspect,
        )
    else:
        plate_fn, ocr_fn = _no_plates, lambda crop: ("", 0.0)
    return Cascade(
        vehicle,
        plate_fn,
        ocr_fn,
        tracker=ByteTracker(**vars(cfg.tracker)),
        gate=QualityGate(**vars(cfg.gate)),
        plate_min_score=cfg.plate.conf,
        settle_reads=cfg.consensus.settle_reads,
        settle_support=cfg.consensus.settle_support,
    )


def build_runner(cfg: Config) -> StreamRunner:
    """Create a ready-to-run ``StreamRunner`` (source, cascade and sinks)."""
    metrics = Metrics()
    queue = LatestQueue(cfg.queue_size, metrics)
    source: str | int = int(cfg.source) if cfg.source.isdigit() else cfg.source
    grabber = FrameGrabber(
        source, queue, live=cfg.live, realtime=cfg.realtime, metrics=metrics
    )
    sinks = []
    if cfg.sinks.jsonl:
        sinks.append(JsonlSink(cfg.sinks.jsonl))
    if cfg.sinks.sqlite:
        sinks.append(SqliteSink(cfg.sinks.sqlite))
    return StreamRunner(
        build_cascade(cfg),
        grabber,
        MultiSink(sinks),
        cfg.camera_id,
        metrics,
        cfg.max_gap,
        cfg.report_every,
    )
