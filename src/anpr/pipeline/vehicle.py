"""Vehicle detection stage: COCO-pretrained detector filtered to road vehicles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO class ids for the vehicle types seen on Indian roads.
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class Detections:
    """Boxes (xyxy), confidences and class ids for one frame."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def __len__(self) -> int:
        return len(self.boxes)


def filter_vehicles(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    min_score: float = 0.25,
    min_area: float = 24.0 * 24.0,
    allowed: frozenset[int] | None = None,
) -> Detections:
    """Keep only vehicle classes above a score and size floor.

    Args:
        boxes: ``(N, 4)`` xyxy boxes in pixels.
        scores: ``(N,)`` confidences.
        classes: ``(N,)`` COCO class ids.
        min_score: Minimum confidence.
        min_area: Minimum box area in pixels; tiny far-away vehicles have no
            readable plate, so skipping them saves compute.
        allowed: Class ids to keep. Defaults to car, motorcycle, bus, truck.

    Returns:
        The filtered detections.
    """
    allowed = frozenset(COCO_VEHICLES) if allowed is None else allowed
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    classes = np.asarray(classes, dtype=int).reshape(-1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = np.isin(classes, list(allowed)) & (scores >= min_score) & (area >= min_area)
    return Detections(boxes[keep], scores[keep], classes[keep])


class VehicleDetector:
    """Ultralytics wrapper returning vehicle-only detections.

    The model is loaded lazily so that importing this module does not require
    ``ultralytics`` or a GPU.

    Args:
        weights: Path or name of a COCO-trained detector, for example
            ``"yolo26n.pt"``.
        imgsz: Inference size.
        conf: Confidence threshold passed to the model.
    """

    def __init__(
        self, weights: str = "yolo26n.pt", imgsz: int = 640, conf: float = 0.25
    ):
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self._model = None

    def _predict(self, frames):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model.predict(
            frames, imgsz=self.imgsz, conf=self.conf, verbose=False
        )

    def _filter(self, res) -> Detections:
        return filter_vehicles(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
            min_score=self.conf,
        )

    def __call__(self, frame: np.ndarray) -> Detections:
        """Detect vehicles in one BGR frame."""
        return self._filter(self._predict(frame)[0])

    def detect_batch(self, frames: list[np.ndarray]) -> list[Detections]:
        """Detect vehicles in several frames with one model call.

        Batching keeps the GPU busy when many cameras are served by one model.
        """
        if not frames:
            return []
        return [self._filter(r) for r in self._predict(list(frames))]
