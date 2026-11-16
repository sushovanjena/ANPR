import json
import sqlite3
import threading
import time

import numpy as np

from anpr.pipeline.cascade import Cascade, QualityGate
from anpr.pipeline.vehicle import Detections
from anpr.stream.events import PlateEvent
from anpr.stream.metrics import Metrics
from anpr.stream.runner import StreamRunner
from anpr.stream.sinks import CallbackSink, JsonlSink, MultiSink, SqliteSink
from anpr.stream.source import Frame, FrameGrabber, LatestQueue


class FakeCap:
    """A scripted capture: yields n frames, optionally failing in between."""

    def __init__(self, n, fail_after=None):
        self.n, self.i, self.fail_after = n, 0, fail_after
        self.released = False

    def get(self, prop):
        return 100.0

    def read(self):
        if self.fail_after is not None and self.i == self.fail_after:
            self.fail_after = None
            return False, None
        if self.i >= self.n:
            return False, None
        self.i += 1
        rng = np.random.default_rng(self.i)
        return True, rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def event(**kw):
    base = dict(
        camera_id="c",
        track_id=1,
        event="final",
        vehicle_class="car",
        plate="MH12AB1234",
        confidence=0.9,
        valid=True,
        n_reads=5,
        first_seen=0.0,
        last_seen=1.0,
        state="MH",
        plate_box=(1.0, 2.0, 3.0, 4.0),
    )
    base.update(kw)
    return PlateEvent(**base)


def test_latest_queue_drops_oldest_and_counts():
    m = Metrics()
    q = LatestQueue(maxsize=2, metrics=m)
    for i in range(5):
        q.put(Frame(np.zeros((2, 2, 3), np.uint8), i, float(i)))
    assert m.frames_dropped == 3
    assert [q.get().index, q.get().index] == [3, 4]


def test_queue_get_times_out_empty():
    assert LatestQueue().get(timeout=0.01) is None


def test_grabber_file_mode_reads_all_frames_then_closes():
    q = LatestQueue(maxsize=100)
    g = FrameGrabber("x", q, live=False, opener=lambda s: FakeCap(10))
    g.start()
    g._thread.join(2)
    got = []
    while (f := q.get(0.05)) is not None:
        got.append(f.index)
    assert got == list(range(10)) and q.closed


def test_grabber_reconnects_after_read_failure():
    m = Metrics()
    q = LatestQueue(maxsize=100, metrics=m)
    caps = []

    def opener(src):
        cap = FakeCap(5, fail_after=2) if not caps else FakeCap(3)
        caps.append(cap)
        return cap

    g = FrameGrabber(
        "rtsp://x", q, live=True, backoff=0.01, max_retries=1, opener=opener, metrics=m
    )
    g.start()
    g._thread.join(3)
    assert m.reconnects >= 1 and m.frames_in >= 5 and len(caps) >= 2


def test_realtime_pacing_slows_file_to_source_fps():
    q = LatestQueue(maxsize=100)
    g = FrameGrabber("x", q, live=False, realtime=True, opener=lambda s: FakeCap(20))
    t = time.perf_counter()
    g.start()
    g._thread.join(3)
    assert time.perf_counter() - t >= 0.15  # 20 frames at 100 fps


def test_sinks_roundtrip(tmp_path):
    jl = JsonlSink(tmp_path / "e.jsonl")
    db = SqliteSink(tmp_path / "e.db")
    seen = []
    multi = MultiSink([jl, db, CallbackSink(seen.append)])
    multi.emit(event())
    multi.close()
    row = json.loads((tmp_path / "e.jsonl").read_text().splitlines()[0])
    assert row["plate"] == "MH12AB1234" and seen[0].plate == "MH12AB1234"
    con = sqlite3.connect(tmp_path / "e.db")
    assert con.execute("select plate, valid from plate_events").fetchall() == [
        ("MH12AB1234", 1)
    ]


def test_multisink_isolates_failures():
    class Boom:
        def emit(self, e):
            raise RuntimeError("x")

        def close(self):
            pass

    seen = []
    m = MultiSink([Boom(), CallbackSink(seen.append)])
    m.emit(event())
    assert len(seen) == 1 and m.errors == 1


def make_cascade():
    def vehicles(frame):
        h, w = frame.shape[:2]
        return Detections(
            np.array([[20.0, 60, 200, 200]]), np.array([0.9]), np.array([2])
        )

    def plates(crop):
        h, w = crop.shape[:2]
        return np.array([[w * 0.3, h * 0.6, w * 0.8, h * 0.8]]), np.array([0.9])

    return Cascade(
        vehicles,
        plates,
        lambda c: ("MH12AB1234", 0.9),
        gate=QualityGate(min_width=10, min_height=4, min_sharpness=1.0),
        settle_reads=3,
    )


def test_runner_emits_confirmed_then_no_duplicate_final():
    m = Metrics()
    q = LatestQueue(maxsize=100, metrics=m)
    src = FrameGrabber("x", q, live=False, opener=lambda s: FakeCap(30), metrics=m)
    out = []
    runner = StreamRunner(
        make_cascade(), src, CallbackSink(out.append), "camA", m, report_every=0
    )
    snap = runner.run()
    assert snap["frames_processed"] == 30
    assert [e.event for e in out] == ["confirmed"]
    assert out[0].plate == "MH12AB1234" and out[0].valid and out[0].state == "MH"
    assert "ocr_ms" in snap["stages"] and "vehicle_ms" in snap["stages"]


def test_runner_reports_vehicle_without_plate_at_end():
    m = Metrics()
    q = LatestQueue(maxsize=100, metrics=m)
    src = FrameGrabber("x", q, live=False, opener=lambda s: FakeCap(8), metrics=m)
    cascade = make_cascade()
    cascade.plate_fn = lambda c: (np.zeros((0, 4)), np.zeros(0))
    out = []
    StreamRunner(
        cascade, src, CallbackSink(out.append), "camA", m, report_every=0
    ).run()
    assert len(out) == 1 and out[0].plate is None and out[0].event == "final"


def test_runner_graceful_stop_flushes():
    m = Metrics()
    q = LatestQueue(maxsize=100, metrics=m)
    src = FrameGrabber("x", q, live=True, opener=lambda s: FakeCap(10**6), metrics=m)
    out = []
    runner = StreamRunner(
        make_cascade(), src, CallbackSink(out.append), "c", m, report_every=0
    )
    threading.Timer(0.3, runner.stop).start()
    runner.run()
    assert len(out) >= 1


def test_real_video_file_through_opencv(tmp_path):
    import cv2

    path = str(tmp_path / "t.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
    for i in range(12):
        vw.write(np.full((120, 160, 3), i * 10, dtype=np.uint8))
    vw.release()
    q = LatestQueue(maxsize=100)
    g = FrameGrabber(path, q, live=False).start()
    g._thread.join(3)
    n = 0
    while q.get(0.05) is not None:
        n += 1
    assert n == 12


def test_runner_warms_up_models_before_stream_opens():
    from anpr.stream.runner import StreamRunner

    calls = []

    class Cas:
        def vehicle_fn(self, f):
            calls.append("vehicle")

        def plate_fn(self, f):
            calls.append("plate")

        def ocr_fn(self, f):
            raise RuntimeError("missing weights")

    runner = StreamRunner.__new__(StreamRunner)
    runner.cascade = Cas()
    runner.warmup()  # a failing stage must be logged, not raised
    assert calls == ["vehicle", "plate"]


def test_restart_clock_excludes_setup_time_from_fps():
    import time

    from anpr.stream.metrics import Metrics

    m = Metrics()
    time.sleep(0.3)  # stands in for model warm-up
    m.restart_clock()
    m.frames_in = 10
    time.sleep(0.1)
    assert m.snapshot()["fps_in"] > 40  # would be about 25 without the restart


def test_runner_calls_on_frame_for_every_processed_frame_and_survives_errors():
    m = Metrics()
    q = LatestQueue(maxsize=100, metrics=m)
    src = FrameGrabber("x", q, live=False, opener=lambda s: FakeCap(10), metrics=m)
    seen = []

    def on_frame(image, results, index):
        seen.append((image.shape, len(results), index))
        if index == 3:
            raise RuntimeError("viewer broke")

    runner = StreamRunner(
        make_cascade(),
        src,
        CallbackSink(lambda e: None),
        "camA",
        m,
        report_every=0,
        on_frame=on_frame,
    )
    snap = runner.run()
    assert snap["frames_processed"] == len(seen) == 10
    assert seen[0][0] == (240, 320, 3)
