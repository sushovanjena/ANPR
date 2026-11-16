"""The RF-DETR adapter uses the compiled predictor and falls back when it fails."""

import sys
import types

import numpy as np

from anpr.models.plate_detector import PlateDetector


class _Det:
    xyxy = np.array([[1.0, 2.0, 30.0, 12.0]])
    confidence = np.array([0.9])


class _Predictor:
    def predict(self, image, threshold=0.3):
        return _Det()


def _install_fake_rfdetr(monkeypatch, inference_ok: bool):
    class Model(_Predictor):
        def __init__(self, pretrain_weights=None):
            self.weights = pretrain_weights

        def inference(self, compile=True, batch_size=1):
            if not inference_ok:
                raise RuntimeError("no compiler")
            return _Predictor()

    fake = types.ModuleType("rfdetr")
    fake.RFDETRNano = Model
    monkeypatch.setitem(sys.modules, "rfdetr", fake)
    return Model


def _crop():
    return np.zeros((64, 128, 3), dtype=np.uint8)


def test_uses_compiled_predictor(monkeypatch, tmp_path):
    _install_fake_rfdetr(monkeypatch, inference_ok=True)
    w = tmp_path / "w.pth"
    w.write_bytes(b"x")
    det = PlateDetector(w, upscale_to=0)
    boxes, scores = det(_crop())
    assert isinstance(det._model, _Predictor) and boxes.shape == (1, 4)
    assert scores[0] == 0.9


def test_falls_back_when_inference_fails(monkeypatch, tmp_path):
    model_cls = _install_fake_rfdetr(monkeypatch, inference_ok=False)
    w = tmp_path / "w.pth"
    w.write_bytes(b"x")
    det = PlateDetector(w, upscale_to=0)
    det(_crop())
    assert isinstance(det._model, model_cls)
