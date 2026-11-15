import json
from pathlib import Path

import numpy as np
import pytest

from anpr import factory, run
from anpr.config import load_config
from anpr.pipeline.cascade import Cascade, QualityGate
from anpr.pipeline.vehicle import Detections

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_loads_and_matches_dataclasses():
    cfg = load_config(ROOT / "configs/default.yaml")
    assert cfg.plate.kind == "rfdetr" and cfg.tracker.max_age == 30
    assert cfg.sinks.sqlite.endswith("events.db")


def test_unknown_key_fails_loudly(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("tracker:\n  max_agee: 3\n")
    with pytest.raises(KeyError, match="max_agee"):
        load_config(p)


def test_dotted_overrides_applied_and_none_ignored():
    cfg = load_config(None, {"plate.enabled": False, "source": "a.mp4", "live": None})
    assert cfg.plate.enabled is False and cfg.source == "a.mp4" and cfg.live is False


def test_missing_weights_give_clear_error():
    from anpr.models.plate_detector import PlateDetector

    with pytest.raises(FileNotFoundError, match="weights not found"):
        PlateDetector("/nope/best.pt")


def test_cli_end_to_end_on_a_real_video_file(tmp_path, monkeypatch, capsys):
    import cv2

    video = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 50, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(40):
        vw.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    vw.release()

    def fake_build(cfg):
        return Cascade(
            lambda f: Detections(
                np.array([[20.0, 60, 200, 200]]), np.array([0.9]), np.array([2])
            ),
            lambda c: (
                np.array(
                    [
                        [
                            c.shape[1] * 0.3,
                            c.shape[0] * 0.6,
                            c.shape[1] * 0.8,
                            c.shape[0] * 0.8,
                        ]
                    ]
                ),
                np.array([0.9]),
            ),
            lambda c: ("KA01AB1234", 0.9),
            gate=QualityGate(min_width=10, min_height=4, min_sharpness=1.0),
            settle_reads=3,
        )

    monkeypatch.setattr(factory, "build_cascade", fake_build)
    out = tmp_path / "ev.jsonl"
    rc = run.main(
        ["--source", video, "--jsonl", str(out), "--sqlite", str(tmp_path / "e.db")]
    )
    assert rc == 0
    snap = json.loads(capsys.readouterr().out)
    assert snap["frames_processed"] == 40 and snap["stages"]["ocr_ms"]["n"] >= 3
    ev = [json.loads(line) for line in out.read_text().splitlines()]
    assert ev and ev[0]["plate"] == "KA01AB1234" and ev[0]["state"] == "KA"
