import numpy as np
import pytest

from anpr.deploy.multistream import Camera, MultiStreamRunner
from anpr.pipeline.cascade import Cascade, QualityGate
from anpr.pipeline.vehicle import Detections
from anpr.stream.metrics import Metrics
from anpr.stream.sinks import CallbackSink
from anpr.stream.source import FrameGrabber, LatestQueue
from anpr.viz import GridViewer


class FakeCap:
    def __init__(self, n):
        self.n, self.i = n, 0

    def get(self, prop):
        return 100.0

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.full((240, 320, 3), self.i % 250, dtype=np.uint8)

    def release(self):
        pass


def make_cascade():
    def vehicles(frame):
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


def make_camera(cam_id, n):
    m = Metrics()
    q = LatestQueue(maxsize=100, metrics=m)
    src = FrameGrabber(cam_id, q, live=False, opener=lambda s: FakeCap(n), metrics=m)
    return Camera(cam_id, src, make_cascade(), m)


def test_serves_every_camera_and_tags_events():
    out = []
    cams = [make_camera("a", 12), make_camera("b", 12), make_camera("c", 12)]
    batches = []

    def batch(frames):
        batches.append(len(frames))
        return [
            Detections(np.array([[20.0, 60, 200, 200]]), np.array([0.9]), np.array([2]))
            for _ in frames
        ]

    runner = MultiStreamRunner(
        cams, CallbackSink(out.append), batch, max_batch=4, report_every=0
    )
    snap = runner.run()
    assert snap["frames_processed"] == 36
    assert {c: v["frames_processed"] for c, v in snap["cameras"].items()} == {
        "a": 12,
        "b": 12,
        "c": 12,
    }
    assert {e.camera_id for e in out} == {"a", "b", "c"}
    assert max(batches) > 1 and "vehicle_batch_ms" in snap["stages"]


def test_cameras_keep_separate_tracks_and_reads():
    cams = [make_camera("a", 8), make_camera("b", 8)]
    MultiStreamRunner(cams, CallbackSink(lambda e: None), report_every=0).run()
    assert cams[0].cascade is not cams[1].cascade
    assert cams[0].cascade.tracker is not cams[1].cascade.tracker


def test_slow_model_drops_per_camera_not_globally():
    m1, m2 = Metrics(), Metrics()
    cams = []
    for cid, m in (("a", m1), ("b", m2)):
        q = LatestQueue(maxsize=2, metrics=m)
        src = FrameGrabber(cid, q, live=False, opener=lambda s: FakeCap(200), metrics=m)
        cams.append(Camera(cid, src, make_cascade(), m))
    import time

    for cam in cams:
        inner = cam.cascade.vehicle_fn

        def slow(frame, inner=inner):
            time.sleep(0.01)
            return inner(frame)

        cam.cascade.vehicle_fn = slow
    snap = MultiStreamRunner(cams, CallbackSink(lambda e: None), report_every=0).run()
    served = [c["frames_processed"] for c in snap["cameras"].values()]
    assert min(served) > 0 and abs(served[0] - served[1]) <= 3  # fair
    assert snap["frames_dropped"] > 0


def test_rejects_empty_and_duplicate_cameras():
    with pytest.raises(ValueError):
        MultiStreamRunner([], CallbackSink(lambda e: None))
    with pytest.raises(ValueError):
        MultiStreamRunner(
            [make_camera("a", 1), make_camera("a", 1)], CallbackSink(lambda e: None)
        )


def test_viewer_errors_do_not_stop_streams():
    cams = [make_camera("a", 5)]

    def boom(*a):
        raise RuntimeError("x")

    snap = MultiStreamRunner(
        cams, CallbackSink(lambda e: None), report_every=0, on_frame=boom
    ).run()
    assert snap["frames_processed"] == 5


def test_grid_viewer_tiles_all_cameras(tmp_path):
    cams = [make_camera("a", 1), make_camera("b", 1), make_camera("c", 1)]
    shown = []
    gv = GridViewer(
        lambda jpg, s: shown.append(s),
        {c.camera_id: c.cascade for c in cams},
        cols=2,
        tile_width=160,
        out_path=tmp_path / "g.mp4",
        min_interval=0.0,
    )
    img = np.zeros((120, 160, 3), np.uint8)
    for cid in ("a", "b", "c"):
        gv(cid, img, [], 0)
    gv.close()
    assert shown[-1].startswith("3 cameras") and (tmp_path / "g.mp4").exists()
    assert gv._grid().shape == (240, 320, 3)
