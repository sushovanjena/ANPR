import cv2
import numpy as np

from anpr.pipeline.cascade import Cascade, FrameResult
from anpr.pipeline.vehicle import Detections
from anpr.viz import draw_frame, render_video, running_consensus, track_color


def _result(tid=1, plate=True):
    return FrameResult(
        tid,
        np.array([20.0, 20.0, 120.0, 90.0]),
        2,
        np.array([50.0, 60.0, 90.0, 75.0]) if plate else None,
        None,
        None,
    )


def test_draw_frame_marks_pixels_and_keeps_input():
    frame = np.zeros((120, 160, 3), np.uint8)
    out = draw_frame(frame, [_result()], {1: "TS09EA1234"})
    assert frame.sum() == 0
    assert out.shape == frame.shape
    assert out.sum() > 0
    assert tuple(out[20, 60]) == track_color(1)


def test_draw_frame_without_results_is_identical():
    frame = np.full((50, 60, 3), 7, np.uint8)
    assert np.array_equal(draw_frame(frame, []), frame)


def test_track_color_is_stable_and_cycles():
    assert track_color(3) == track_color(3)
    assert track_color(0) == track_color(6)


def _write_video(path, n=6):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    for _ in range(n):
        w.write(np.full((120, 160, 3), 90, np.uint8))
    w.release()


def test_render_video_writes_annotated_file(tmp_path):
    src, dst = tmp_path / "in.avi", tmp_path / "out.mp4"
    _write_video(src)

    def vehicle(frame):
        return Detections(
            np.array([[20.0, 20.0, 120.0, 90.0]]), np.array([0.9]), np.array([2])
        )

    cascade = Cascade(
        vehicle,
        lambda crop: (np.zeros((0, 4)), np.zeros(0)),
        lambda crop: ("", 0.0),
    )
    stats = render_video(src, dst, cascade, max_seconds=None, out_width=80)
    assert stats["frames"] == 6
    assert stats["tracks"] >= 1
    assert dst.exists() and dst.stat().st_size > 0
    assert running_consensus(cascade) == {}


def test_render_video_missing_file_raises(tmp_path):
    cascade = Cascade(lambda f: None, lambda c: None, lambda c: None)
    try:
        render_video(tmp_path / "nope.avi", tmp_path / "o.mp4", cascade)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
