"""Frame sources for live and recorded video.

A ``FrameGrabber`` reads frames on its own thread and hands them to the
inference loop through a bounded queue that drops the *oldest* frame when full.
For a live camera this keeps latency bounded: if inference is slower than the
camera, we skip stale frames instead of falling further behind real time.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from anpr.stream.metrics import Metrics


@dataclass(frozen=True)
class Frame:
    """A decoded frame with its sequence number and capture time."""

    image: np.ndarray
    index: int
    ts: float


class LatestQueue:
    """Thread-safe bounded queue that discards the oldest item when full.

    Args:
        maxsize: Maximum number of frames held.
        metrics: Optional metrics object that counts dropped frames.
    """

    def __init__(self, maxsize: int = 2, metrics: Metrics | None = None) -> None:
        self._q: deque[Frame] = deque()
        self._max = maxsize
        self._cv = threading.Condition()
        self._closed = False
        self.metrics = metrics

    def put(self, item: Frame) -> None:
        """Add a frame, dropping the oldest one if the queue is full."""
        with self._cv:
            if len(self._q) >= self._max:
                self._q.popleft()
                if self.metrics:
                    self.metrics.frames_dropped += 1
            self._q.append(item)
            self._cv.notify()

    def get(self, timeout: float = 1.0) -> Frame | None:
        """Return the oldest queued frame, or None on timeout or when closed."""
        with self._cv:
            if not self._q and not self._closed:
                self._cv.wait(timeout)
            return self._q.popleft() if self._q else None

    def close(self) -> None:
        """Wake any waiting consumer; no more frames will arrive."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        """True once the producer has finished and the queue is drained."""
        with self._cv:
            return self._closed and not self._q


def _open_cv(source: str | int):
    import cv2

    cap = cv2.VideoCapture(source)
    if isinstance(source, str) and source.startswith("rtsp"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class FrameGrabber:
    """Read frames from RTSP, a file or a webcam on a background thread.

    Args:
        source: RTSP/HTTP URL, file path or webcam index.
        queue: Destination queue.
        live: True for cameras (timestamps use the wall clock, reconnect on
            failure). False for files (stop at end of file).
        realtime: For files, sleep to match the source FPS so that a recording
            behaves like a live stream when benchmarking.
        max_retries: Consecutive reconnect attempts before giving up (live).
        backoff: Initial reconnect delay in seconds; doubles up to 10 s.
        opener: Factory returning a ``cv2.VideoCapture``-like object. Injected
            in tests.
        metrics: Optional metrics collector.
    """

    def __init__(
        self,
        source: str | int,
        queue: LatestQueue,
        live: bool = True,
        realtime: bool = False,
        max_retries: int = 20,
        backoff: float = 0.5,
        opener: Callable[[str | int], object] = _open_cv,
        metrics: Metrics | None = None,
    ) -> None:
        self.source = source
        self.queue = queue
        self.live = live
        self.realtime = realtime
        self.max_retries = max_retries
        self.backoff = backoff
        self.opener = opener
        self.metrics = metrics
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> FrameGrabber:
        """Start the reader thread."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Ask the reader thread to stop and wait for it."""
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        idx, retries, delay = 0, 0, self.backoff
        cap = self.opener(self.source)
        fps = (float(cap.get(5) or 0) if hasattr(cap, "get") else 0.0) or 25.0
        t_start = time.perf_counter()
        while not self._stop.is_set():
            ok, img = cap.read()
            if not ok or img is None:
                if not self.live:
                    break
                cap.release()
                retries += 1
                if retries > self.max_retries:
                    break
                if self.metrics:
                    self.metrics.reconnects += 1
                self._stop.wait(delay)
                delay = min(delay * 2, 10.0)
                cap = self.opener(self.source)
                continue
            retries, delay = 0, self.backoff
            ts = time.time() if self.live else idx / fps
            self.queue.put(Frame(img, idx, ts))
            if self.metrics:
                self.metrics.frames_in += 1
            idx += 1
            if self.realtime and not self.live:
                due = t_start + idx / fps
                self._stop.wait(max(0.0, due - time.perf_counter()))
        cap.release()
        self.queue.close()
