import numpy as np

from anpr.track.bytetrack import ByteTracker, iou_matrix


def box(x, y, w=50, h=50):
    return [x, y, x + w, y + h]


def test_iou_identity_and_disjoint():
    a = np.array([box(0, 0)])
    assert iou_matrix(a, a)[0, 0] == 1.0
    assert iou_matrix(a, np.array([box(200, 200)]))[0, 0] == 0.0


def test_id_stable_for_moving_vehicle():
    tr = ByteTracker(min_hits=2)
    ids = set()
    for i in range(10):
        out = tr.update(np.array([box(10 + 4 * i, 20)]), np.array([0.9]))
        ids |= {t.track_id for t in out}
    assert ids == {1}


def test_two_vehicles_keep_distinct_ids():
    tr = ByteTracker(min_hits=2)
    last = []
    for i in range(8):
        last = tr.update(
            np.array([box(10 + 3 * i, 20), box(300 - 3 * i, 200)]), np.array([0.9, 0.9])
        )
    assert sorted(t.track_id for t in last) == [1, 2]


def test_low_score_detection_keeps_track_alive():
    tr = ByteTracker(min_hits=1, high_thr=0.5, low_thr=0.1)
    tr.update(np.array([box(10, 10)]), np.array([0.9]))
    out = tr.update(np.array([box(12, 10)]), np.array([0.3]))
    assert [t.track_id for t in out] == [1]


def test_track_dropped_after_max_age():
    tr = ByteTracker(min_hits=1, max_age=2)
    tr.update(np.array([box(10, 10)]), np.array([0.9]))
    for _ in range(4):
        tr.update(np.zeros((0, 4)), np.zeros(0))
    assert tr.tracks == []
    out = tr.update(np.array([box(10, 10)]), np.array([0.9]))
    assert out[0].track_id == 2
