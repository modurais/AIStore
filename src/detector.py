from dataclasses import dataclass
from pathlib import Path
from typing import List

from ultralytics import YOLO


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class YoloShelfDetector:
    def __init__(self, model_path: Path, confidence: float = 0.25, iou: float = 0.45):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.model = YOLO(str(self.model_path))

    def detect(self, image_path: Path) -> List[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        detections: List[Detection] = []
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            detections.append(
                Detection(
                    x1=max(0, int(x1)),
                    y1=max(0, int(y1)),
                    x2=max(0, int(x2)),
                    y2=max(0, int(y2)),
                    confidence=conf,
                )
            )

        return detections
