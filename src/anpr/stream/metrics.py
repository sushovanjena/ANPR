"""Latency and throughput metrics for the streaming pipeline."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np


class Metrics:
    """Rolling per-stage latency, frame rate and drop counters.

    Args:
        window: Number of recent samples kept per stage.
    """

    def __init__(self, window: int = 300) -> None:
        self._lat: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.frames_in = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.reconnects = 0
        self._t0 = time.perf_counter()

    @contextmanager
    def timer(self, stage: str):
        """Time a block of work and record it under ``stage`` (milliseconds)."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self._lat[stage].append((time.perf_counter() - t) * 1000.0)

    def record(self, stage: str, ms: float) -> None:
        """Record a latency sample directly."""
        self._lat[stage].append(ms)

    def snapshot(self) -> dict:
        """Return counters, FPS and p50/p95 latency per stage."""
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        stages = {}
        for name, vals in self._lat.items():
            a = np.asarray(vals)
            stages[name] = {
                "p50_ms": round(float(np.percentile(a, 50)), 2),
                "p95_ms": round(float(np.percentile(a, 95)), 2),
                "n": len(a),
            }
        return {
            "frames_in": self.frames_in,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "reconnects": self.reconnects,
            "fps_in": round(self.frames_in / elapsed, 2),
            "fps_processed": round(self.frames_processed / elapsed, 2),
            "stages": stages,
        }
