import numpy as np

from anpr.pipeline.cascade import Cascade, QualityGate, sharpness
from anpr.pipeline.vehicle import Detections


def frame_with_texture(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def moving_vehicles(i):
    # One car moving right, one motorcycle moving left; class ids are COCO.
    boxes = np.array(
        [[20 + 5 * i, 60, 140 + 5 * i, 160], [250 - 4 * i, 120, 300 - 4 * i, 200]]
    )
    return Detections(boxes.astype(float), np.array([0.9, 0.8]), np.array([2, 3]))


def plate_in_crop(crop):
    h, w = crop.shape[:2]
    return np.array([[w * 0.3, h * 0.6, w * 0.8, h * 0.8]]), np.array([0.9])


def test_sharpness_blur_vs_texture():
    sharp = frame_with_texture()
    assert sharpness(sharp) > 1000
    assert sharpness(np.full((50, 50), 120, dtype=np.uint8)) == 0.0


def test_quality_gate_rejects_small_and_flat():
    gate = QualityGate()
    assert not gate.passes(np.zeros((5, 10, 3), dtype=np.uint8))
    assert not gate.passes(np.full((30, 90, 3), 100, dtype=np.uint8))
    assert gate.passes(frame_with_texture()[:30, :90])


def test_cascade_end_to_end_recovers_plate_from_noisy_reads():
    reads = iter(
        [
            ("MH12AB1234", 0.8),
            ("MH12AB1284", 0.5),
            ("MH12A81234", 0.7),
            ("MH12AB1234", 0.9),
            ("MH12AB1234", 0.8),
        ]
        * 20
    )
    cascade = Cascade(
        vehicle_fn=lambda f: moving_vehicles(cascade_frame[0]),
        plate_fn=plate_in_crop,
        ocr_fn=lambda c: next(reads),
        gate=QualityGate(min_width=10, min_height=4, min_sharpness=1.0),
        settle_reads=100,
    )
    cascade_frame = [0]
    frame = frame_with_texture()
    last = []
    for i in range(12):
        cascade_frame[0] = i
        last = cascade.process(frame, i)
    assert len(last) == 2
    assert {r.vehicle_cls for r in last} == {2, 3}
    final = cascade.finalize()
    assert final and all(c.text == "MH12AB1234" and c.valid for c in final.values())


def test_settled_track_stops_running_ocr():
    calls = []

    def ocr(crop):
        calls.append(1)
        return "KA01AB1234", 0.9

    cascade = Cascade(
        vehicle_fn=lambda f: Detections(
            np.array([[20.0, 60, 160, 180]]), np.array([0.9]), np.array([2])
        ),
        plate_fn=plate_in_crop,
        ocr_fn=ocr,
        gate=QualityGate(min_width=10, min_height=4, min_sharpness=1.0),
        settle_reads=3,
    )
    frame = frame_with_texture()
    for i in range(20):
        cascade.process(frame, i)
    assert len(calls) < 8
    assert cascade.finalize()[1].text == "KA01AB1234"


def test_vehicle_without_plate_is_still_reported():
    cascade = Cascade(
        vehicle_fn=lambda f: Detections(
            np.array([[20.0, 60, 160, 180]]), np.array([0.9]), np.array([5])
        ),
        plate_fn=lambda c: (np.zeros((0, 4)), np.zeros(0)),
        ocr_fn=lambda c: ("X", 0.0),
    )
    frame = frame_with_texture()
    out = []
    for i in range(4):
        out = cascade.process(frame, i)
    assert len(out) == 1 and out[0].plate_box is None and cascade.finalize() == {}
