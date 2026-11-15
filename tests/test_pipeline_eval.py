from anpr.eval.pipeline_eval import edit_distance, evaluate_events


def ev(track, plate, valid=True, event="final"):
    return {
        "camera_id": "c",
        "track_id": track,
        "event": event,
        "plate": plate,
        "valid": valid,
    }


def test_edit_distance():
    assert edit_distance("MH12AB1234", "MH12AB1234") == 0
    assert edit_distance("MH12AB1234", "MH12AB1284") == 1
    assert edit_distance("", "ABC") == 3


def test_scores_recall_precision_cer_and_unread():
    events = [
        ev(1, "MH12AB1234"),
        ev(2, "KA01AB1284"),  # one character wrong
        ev(3, None, valid=False),  # vehicle without a plate
        ev(4, "XXXX", valid=False),  # failed grammar -> review
    ]
    r = evaluate_events(events, ["MH12AB1234", "KA01AB1234"])
    assert r["recall"] == 0.5
    assert r["precision"] == 0.5
    assert abs(r["cer"] - 0.05) < 1e-9
    assert r["unread_vehicles"] == 1 and abs(r["review_rate"] - 1 / 3) < 1e-9


def test_final_event_supersedes_confirmed():
    events = [
        ev(1, "MH12AB1284", event="confirmed"),
        ev(1, "MH12AB1234", event="final"),
    ]
    assert evaluate_events(events, ["MH12AB1234"])["recall"] == 1.0
