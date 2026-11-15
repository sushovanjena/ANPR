import numpy as np

from anpr.pipeline.association import associate, containment


def test_containment_full_and_none():
    plate = np.array([[10, 10, 20, 20]])
    veh = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
    c = containment(plate, veh)
    assert c[0, 0] == 1.0 and c[0, 1] == 0.0


def test_plate_goes_to_smallest_containing_vehicle():
    plates = np.array([[40, 40, 50, 45]])
    vehicles = np.array([[0, 0, 200, 200], [30, 30, 70, 70]])
    assert associate(plates, np.array([0.9]), vehicles) == {1: 0}


def test_orphan_plate_dropped():
    plates = np.array([[500, 500, 520, 510]])
    vehicles = np.array([[0, 0, 100, 100]])
    assert associate(plates, np.array([0.9]), vehicles) == {}


def test_vehicle_keeps_best_plate():
    plates = np.array([[10, 10, 30, 20], [50, 50, 70, 60]])
    vehicles = np.array([[0, 0, 100, 100]])
    assert associate(plates, np.array([0.4, 0.8]), vehicles) == {0: 1}


def test_hungarian_one_plate_per_vehicle_in_adjacent_lanes():
    vehicles = np.array([[0, 0, 100, 100], [90, 0, 200, 100]])
    plates = np.array([[40, 70, 60, 80], [140, 70, 160, 80]])
    out = associate(plates, np.array([0.8, 0.9]), vehicles)
    assert out == {0: 0, 1: 1}


def test_position_prior_prefers_lower_centred_plate():
    vehicles = np.array([[0, 0, 100, 100]])
    top_left = [2, 2, 22, 12]
    rear_plate = [40, 70, 60, 80]
    out = associate(np.array([top_left, rear_plate]), np.array([0.8, 0.8]), vehicles)
    assert out == {0: 1}
