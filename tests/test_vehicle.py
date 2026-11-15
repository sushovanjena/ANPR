import numpy as np

from anpr.pipeline.vehicle import filter_vehicles


def test_filters_class_score_and_size():
    boxes = np.array(
        [[0, 0, 100, 100], [0, 0, 100, 100], [0, 0, 10, 10], [0, 0, 100, 100]]
    )
    scores = np.array([0.9, 0.9, 0.9, 0.1])
    classes = np.array([2, 0, 3, 7])  # car, person, tiny motorcycle, weak truck
    d = filter_vehicles(boxes, scores, classes)
    assert len(d) == 1 and d.classes[0] == 2
