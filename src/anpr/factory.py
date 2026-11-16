"""Build the cascade and stream runner from a ``Config``."""

from __future__ import annotations

import numpy as np

from anpr.config import Config
from anpr.deploy.multistream import Camera, MultiStreamRunner
from anpr.pipeline.cascade import Cascade, QualityGate
from anpr.pipeline.vehicle import VehicleDetector
from anpr.stream.metrics import Metrics
from anpr.stream.runner import StreamRunner
from anpr.stream.sinks import JsonlSink, MultiSink, SqliteSink
from anpr.stream.source import FrameGrabber, LatestQueue
from anpr.track.bytetrack import ByteTracker


def _no_plates(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 4)), np.zeros(0)


def build_models(cfg: Config) -> tuple:
    """Create the three models once: ``(vehicle, plate_fn, ocr_fn)``.

    Weights listed in ``weights/manifest.json`` are downloaded on first use when
    ``auto_fetch_weights`` is true. With ``plate.enabled`` false the plate
    detector and OCR are not loaded, so the vehicle detector and tracker can be
    tested before plate weights exist. Several cascades (one per camera) can
    share the returned models, so the GPU holds one copy.
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
    return vehicle, plate_fn, ocr_fn


def build_cascade(cfg: Config, models: tuple | None = None) -> Cascade:
    """Create a cascade for ``cfg`` with its own tracker and read history.

    Args:
        cfg: Pipeline configuration.
        models: Shared ``(vehicle, plate_fn, ocr_fn)`` from ``build_models``;
            built here when omitted.
    """
    vehicle, plate_fn, ocr_fn = models or build_models(cfg)
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


def build_runner(cfg: Config, on_frame=None) -> StreamRunner:
    """Create a ready-to-run ``StreamRunner`` (source, cascade and sinks).

    Args:
        cfg: Pipeline configuration.
        on_frame: Optional per-frame callback for live views (see ``StreamRunner``).
    """
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
        on_frame,
    )


def build_multistream(
    cfg: Config,
    sources: dict[str, str],
    on_frame=None,
    max_batch: int = 8,
) -> MultiStreamRunner:
    """Create a runner that serves several cameras with one set of models.

    Args:
        cfg: Pipeline configuration; ``cfg.source`` and ``cfg.camera_id`` are
            ignored in favour of ``sources``.
        sources: Camera id to RTSP URL, video file or webcam index.
        on_frame: Optional ``(camera_id, image, results, frame_index)`` callback.
        max_batch: Most frames per batched vehicle-detection call.
    """
    models = build_models(cfg)
    cameras = []
    for cam_id, src in sources.items():
        metrics = Metrics()
        queue = LatestQueue(cfg.queue_size, metrics)
        source: str | int = int(src) if str(src).isdigit() else src
        grabber = FrameGrabber(
            source, queue, live=cfg.live, realtime=cfg.realtime, metrics=metrics
        )
        cameras.append(Camera(cam_id, grabber, build_cascade(cfg, models), metrics))
    sinks = []
    if cfg.sinks.jsonl:
        sinks.append(JsonlSink(cfg.sinks.jsonl))
    if cfg.sinks.sqlite:
        sinks.append(SqliteSink(cfg.sinks.sqlite))
    vehicle = models[0]
    batch_fn = getattr(vehicle, "detect_batch", None)
    return MultiStreamRunner(
        cameras,
        MultiSink(sinks),
        batch_fn,
        max_batch,
        cfg.max_gap,
        cfg.report_every,
        on_frame,
    )
