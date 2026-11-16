"""Streaming runner: grabber thread, inference loop, events and metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from anpr.pipeline.cascade import Cascade
from anpr.stream.events import to_event
from anpr.stream.metrics import Metrics
from anpr.stream.sinks import Sink
from anpr.stream.source import FrameGrabber, LatestQueue

log = logging.getLogger("anpr.stream")


def instrument(cascade: Cascade, metrics: Metrics) -> Cascade:
    """Wrap the cascade's model callables so each stage reports its latency."""

    def timed(name: str, fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with metrics.timer(name):
                return fn(*args, **kwargs)

        return wrapper

    cascade.vehicle_fn = timed("vehicle_ms", cascade.vehicle_fn)
    cascade.plate_fn = timed("plate_ms", cascade.plate_fn)
    cascade.ocr_fn = timed("ocr_ms", cascade.ocr_fn)
    return cascade


class StreamRunner:
    """Run the cascade over a stream and emit plate events.

    Args:
        cascade: The configured cascade (instrumented automatically).
        source: Frame source (a ``FrameGrabber`` or compatible object exposing
            ``queue``, ``start`` and ``stop``).
        sink: Event sink.
        camera_id: Name attached to every event.
        metrics: Shared metrics object. The source should use the same one.
        max_gap: Frames without a sighting before a vehicle is considered gone.
        report_every: Seconds between metric log lines (0 disables).
    """

    def __init__(
        self,
        cascade: Cascade,
        source: FrameGrabber,
        sink: Sink,
        camera_id: str = "cam0",
        metrics: Metrics | None = None,
        max_gap: int = 45,
        report_every: float = 10.0,
    ) -> None:
        self.metrics = metrics or Metrics()
        self.cascade = instrument(cascade, self.metrics)
        self.source = source
        self.sink = sink
        self.camera_id = camera_id
        self.max_gap = max_gap
        self.report_every = report_every
        self.events = 0
        self._stopped = False

    def stop(self) -> None:
        """Request a graceful stop (safe to call from a signal handler)."""
        self._stopped = True

    def _emit(self, summaries) -> None:
        for s in summaries:
            self.sink.emit(to_event(s, self.camera_id))
            self.events += 1

    def warmup(self) -> None:
        """Load and warm every model on dummy input before the stream opens.

        The first call of a model pays weight loading and CUDA initialisation
        (about 16 s for the vehicle detector on a Colab T4), during which a
        live source would fill and drop its queue. Failures are logged, not
        raised, because a missing optional model should not stop the stream.
        """
        steps = (
            ("vehicle", self.cascade.vehicle_fn, np.zeros((720, 1280, 3), np.uint8)),
            ("plate", self.cascade.plate_fn, np.zeros((240, 320, 3), np.uint8)),
            ("ocr", self.cascade.ocr_fn, np.zeros((48, 160, 3), np.uint8)),
        )
        for name, fn, dummy in steps:
            try:
                fn(dummy)
            except Exception:  # noqa: BLE001
                log.warning("warm-up of %s stage failed", name, exc_info=True)

    def run(self, max_frames: int | None = None) -> dict:
        """Process frames until the source ends, ``stop()`` or ``max_frames``.

        Returns:
            Final metrics snapshot.
        """
        q: LatestQueue = self.source.queue
        self.warmup()
        self.metrics.restart_clock()
        self.source.start()
        last_report = time.perf_counter()
        last_idx = 0
        try:
            while not self._stopped:
                frame = q.get(timeout=0.5)
                if frame is None:
                    if q.closed:
                        break
                    continue
                with self.metrics.timer("frame_ms"):
                    self.cascade.process(frame.image, frame.index, frame.ts)
                    self._emit(self.cascade.newly_settled())
                    self._emit(self.cascade.drain(frame.index, self.max_gap))
                self.metrics.frames_processed += 1
                last_idx = frame.index
                if max_frames and self.metrics.frames_processed >= max_frames:
                    break
                now = time.perf_counter()
                if self.report_every and now - last_report >= self.report_every:
                    log.info("metrics %s", self.metrics.snapshot())
                    last_report = now
        finally:
            self.source.stop()
            self._emit(self.cascade.drain(last_idx, flush=True))
            self.sink.close()
        return self.metrics.snapshot()
