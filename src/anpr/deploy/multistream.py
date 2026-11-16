"""Serve many camera streams from one GPU.

Each camera has its own grabber thread, drop-oldest queue, tracker and read
history, so one slow or broken stream cannot stall or corrupt another. A single
inference loop serves all cameras: every round it takes the newest frame of the
cameras that have waited longest, runs vehicle detection for them in one batched
model call, then runs the rest of the cascade per camera. When the GPU cannot
keep up, load is shed fairly: every camera drops its own stale frames, and the
per-camera counters show exactly how many.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from anpr.pipeline.cascade import Cascade
from anpr.stream.events import to_event
from anpr.stream.metrics import Metrics
from anpr.stream.runner import instrument
from anpr.stream.sinks import Sink
from anpr.stream.source import FrameGrabber, LatestQueue

log = logging.getLogger("anpr.multistream")

BatchFn = Callable[[list[np.ndarray]], list]


@dataclass
class Camera:
    """One camera: its source, queue, cascade and counters.

    Attributes:
        camera_id: Name attached to every event from this camera.
        source: Frame grabber feeding ``queue``.
        cascade: This camera's own cascade (tracker and read history).
        metrics: Per-camera frame counters (shared with ``source`` and queue).
        last_served: Time this camera was last given to the model.
    """

    camera_id: str
    source: FrameGrabber
    cascade: Cascade
    metrics: Metrics
    last_served: float = field(default=0.0)
    last_idx: int = 0

    @property
    def queue(self) -> LatestQueue:
        """The camera's frame queue."""
        return self.source.queue


class MultiStreamRunner:
    """Run several cameras through shared models and emit plate events.

    Args:
        cameras: The cameras to serve.
        sink: Event sink shared by all cameras (events carry ``camera_id``).
        vehicle_batch_fn: Frames to one ``Detections`` per frame. When None each
            camera's cascade runs its own vehicle detector, one frame at a time.
        max_batch: Most frames sent to the vehicle detector in one call.
        max_gap: Frames without a sighting before a vehicle is considered gone.
        report_every: Seconds between metric log lines (0 disables).
        on_frame: Optional ``(camera_id, image, results, frame_index)`` callback
            after every processed frame. Its errors are logged, never raised.
    """

    def __init__(
        self,
        cameras: list[Camera],
        sink: Sink,
        vehicle_batch_fn: BatchFn | None = None,
        max_batch: int = 8,
        max_gap: int = 45,
        report_every: float = 10.0,
        on_frame: Callable | None = None,
    ) -> None:
        if not cameras:
            raise ValueError("at least one camera is required")
        ids = [c.camera_id for c in cameras]
        if len(set(ids)) != len(ids):
            raise ValueError(f"camera ids must be unique, got {ids}")
        self.cameras = cameras
        self.sink = sink
        self.vehicle_batch_fn = vehicle_batch_fn
        self.max_batch = max_batch
        self.max_gap = max_gap
        self.report_every = report_every
        self.on_frame = on_frame
        self.metrics = Metrics()
        self.events = 0
        self._stopped = False
        for cam in cameras:
            instrument(cam.cascade, self.metrics)

    def stop(self) -> None:
        """Request a graceful stop (safe to call from a signal handler)."""
        self._stopped = True

    def warmup(self) -> None:
        """Load and warm every model on dummy input before streams open."""
        cas = self.cameras[0].cascade
        steps = (
            ("vehicle", self.vehicle_batch_fn or cas.vehicle_fn),
            ("plate", cas.plate_fn),
            ("ocr", cas.ocr_fn),
        )
        shapes = {
            "vehicle": (720, 1280, 3),
            "plate": (240, 320, 3),
            "ocr": (48, 160, 3),
        }
        for name, fn in steps:
            dummy = np.zeros(shapes[name], np.uint8)
            if name == "vehicle" and self.vehicle_batch_fn is not None:
                dummy = [dummy] * min(2, self.max_batch)
            try:
                fn(dummy)
            except Exception:  # noqa: BLE001 - optional model may be missing
                log.warning("warm-up of %s stage failed", name, exc_info=True)

    def _next_batch(self) -> list[tuple[Camera, object]]:
        """Next queued frame of each camera, longest-waiting camera first."""
        ready = []
        for cam in sorted(self.cameras, key=lambda c: c.last_served):
            if len(ready) >= self.max_batch:
                break
            frame = cam.queue.get(timeout=0.0)
            if frame is not None:
                ready.append((cam, frame))
        return ready

    def _notify(self, cam: Camera, frame, results) -> None:
        if self.on_frame is None:
            return
        try:
            self.on_frame(cam.camera_id, frame.image, results, frame.index)
        except Exception:  # noqa: BLE001 - a broken viewer must not stop streams
            log.warning("on_frame callback failed", exc_info=True)

    def _emit(self, cam: Camera, summaries) -> None:
        for s in summaries:
            self.sink.emit(to_event(s, cam.camera_id))
            self.events += 1

    def _serve(self, batch: list[tuple[Camera, object]]) -> None:
        dets: list = [None] * len(batch)
        if self.vehicle_batch_fn is not None:
            with self.metrics.timer("vehicle_batch_ms"):
                dets = self.vehicle_batch_fn([f.image for _, f in batch])
            self.metrics.record("batch_size", float(len(batch)))
        for (cam, frame), det in zip(batch, dets, strict=True):
            t0 = time.perf_counter()
            results = cam.cascade.process(frame.image, frame.index, frame.ts, det)
            self._emit(cam, cam.cascade.newly_settled())
            self._emit(cam, cam.cascade.drain(frame.index, self.max_gap))
            self.metrics.record("camera_ms", (time.perf_counter() - t0) * 1000.0)
            cam.metrics.frames_processed += 1
            self.metrics.frames_processed += 1
            cam.last_served = time.perf_counter()
            cam.last_idx = frame.index
            self._notify(cam, frame, results)

    def snapshot(self) -> dict:
        """Aggregate and per-camera counters plus per-stage latency."""
        snap = self.metrics.snapshot()
        cams = {}
        for cam in self.cameras:
            m = cam.metrics.snapshot()
            cams[cam.camera_id] = {
                k: m[k]
                for k in (
                    "frames_in",
                    "frames_processed",
                    "frames_dropped",
                    "fps_in",
                    "fps_processed",
                    "reconnects",
                )
            }
        snap["frames_in"] = sum(c["frames_in"] for c in cams.values())
        snap["frames_dropped"] = sum(c["frames_dropped"] for c in cams.values())
        snap["fps_in"] = round(sum(c["fps_in"] for c in cams.values()), 2)
        snap["cameras"] = cams
        return snap

    def run(self, max_frames: int | None = None) -> dict:
        """Serve all cameras until every stream ends, ``stop()`` or ``max_frames``.

        Returns:
            Final snapshot (see ``snapshot``).
        """
        self.warmup()
        for cam in self.cameras:
            cam.metrics.restart_clock()
        self.metrics.restart_clock()
        for cam in self.cameras:
            cam.source.start()
        last_report = time.perf_counter()
        try:
            while not self._stopped:
                batch = self._next_batch()
                if not batch:
                    if all(c.queue.closed for c in self.cameras):
                        break
                    time.sleep(0.005)
                    continue
                self._serve(batch)
                if max_frames and self.metrics.frames_processed >= max_frames:
                    break
                now = time.perf_counter()
                if self.report_every and now - last_report >= self.report_every:
                    log.info("metrics %s", self.snapshot())
                    last_report = now
        finally:
            for cam in self.cameras:
                cam.source.stop()
                self._emit(cam, cam.cascade.drain(cam.last_idx, flush=True))
            self.sink.close()
        return self.snapshot()
