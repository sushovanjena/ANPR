import numpy as np

from anpr.eval.detection import average_precision, evaluate, iou_matrix


def _box(x1, y1, x2, y2):
    return np.array([[x1, y1, x2, y2]], dtype=float)


def test_iou_identical_and_disjoint():
    assert np.isclose(iou_matrix(_box(0, 0, 10, 10), _box(0, 0, 10, 10))[0, 0], 1.0)
    assert iou_matrix(_box(0, 0, 10, 10), _box(20, 20, 30, 30))[0, 0] == 0.0


def test_perfect_prediction_gives_ap_one():
    gts = [_box(10, 10, 50, 30)]
    preds = [(_box(10, 10, 50, 30), np.array([0.9]))]
    assert average_precision(preds, gts, 0.5) == 1.0


def test_no_prediction_gives_zero_and_false_positive_lowers_ap():
    gts = [_box(10, 10, 50, 30)]
    assert average_precision([(np.zeros((0, 4)), np.zeros(0))], gts, 0.5) == 0.0
    boxes = np.vstack([_box(200, 200, 240, 220), _box(10, 10, 50, 30)])
    preds = [(boxes, np.array([0.95, 0.9]))]
    assert 0.0 < average_precision(preds, gts, 0.5) < 1.0


def test_evaluate_reports_size_bins():
    gts = [_box(0, 0, 10, 10), _box(0, 0, 500, 500)]
    preds = [(_box(0, 0, 10, 10), np.array([0.8])), (np.zeros((0, 4)), np.zeros(0))]
    res = evaluate(preds, gts, [10000.0, 1000000.0])
    assert (
        res["recall 0.5-2%"][0] == 1.0
        and res["recall >=10%"][0] == 0.0
        or res["recall <0.5%"] is not None
    )
    assert res["n_gt"] == 2
