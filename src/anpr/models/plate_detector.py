"""Plate detector adapter for YOLO26 and RF-DETR weights.

The trained weights are plugged in by path, so the rest of the cascade never
depends on which detector family produced them. The adapter returns boxes in
the coordinates of the crop it was given.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KINDS = ("yolo", "rfdetr")


class PlateDetector:
    """Detect plates in a (vehicle) crop.

    Args:
        weights: Path to ``best.pt`` (YOLO) or ``checkpoint_best_ema.pth``
            (RF-DETR Nano).
        kind: ``"yolo"`` or ``"rfdetr"``.
        conf: Detector confidence threshold.
        imgsz: Inference size for YOLO. RF-DETR uses its own resolution.
        min_crop: Crops smaller than this (pixels, either side) are skipped.
        upscale_to: Crops whose longer side is below this are upscaled before
            detection, because a plate in a small vehicle crop is only a few
            pixels wide and the detector was trained at larger plate sizes.
        device: Torch device string or None for automatic.
        optimize: For RF-DETR, build the compiled inference predictor (much
            faster than the training-mode model). Falls back to the plain model
            if compilation is not available.
    """

    def __init__(
        self,
        weights: str | Path,
        kind: str = "rfdetr",
        conf: float = 0.3,
        imgsz: int = 640,
        min_crop: int = 16,
        upscale_to: int = 320,
        device: str | None = None,
        optimize: bool = True,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Plate detector weights not found: {self.weights}. Train the "
                "detector (see docs) and pass the checkpoint path."
            )
        self.kind = kind
        self.conf = conf
        self.imgsz = imgsz
        self.min_crop = min_crop
        self.upscale_to = upscale_to
        self.device = device
        self.optimize = optimize
        self._model = None

    def _load(self):
        if self.kind == "yolo":
            from ultralytics import YOLO

            return YOLO(str(self.weights))
        from rfdetr import RFDETRNano

        model = RFDETRNano(pretrain_weights=str(self.weights))
        if self.optimize:
            try:
                return model.inference(compile=True, batch_size=1)
            except Exception as exc:  # noqa: BLE001 - any failure means "use plain"
                logger.warning("RF-DETR inference optimisation failed (%s)", exc)
        return model

    def __call__(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boxes, scores)`` in crop coordinates for a BGR crop."""
        empty = (np.zeros((0, 4)), np.zeros(0))
        h, w = crop.shape[:2]
        if min(h, w) < self.min_crop:
            return empty
        if self._model is None:
            self._model = self._load()
        scale = 1.0
        img = crop
        if max(h, w) < self.upscale_to:
            import cv2

            scale = self.upscale_to / max(h, w)
            img = cv2.resize(
                crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )
        if self.kind == "yolo":
            res = self._model.predict(
                img, imgsz=self.imgsz, conf=self.conf, verbose=False
            )[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
        else:
            from PIL import Image

            rgb = np.ascontiguousarray(img[:, :, ::-1])
            det = self._model.predict(Image.fromarray(rgb), threshold=self.conf)
            boxes = np.asarray(det.xyxy, dtype=float).reshape(-1, 4)
            scores = np.asarray(det.confidence, dtype=float)
        return boxes / scale, scores
