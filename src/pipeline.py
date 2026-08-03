from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image

from src.classifier import ClassificationResult, ClipProductClassifier
from src.detector import Detection, YoloShelfDetector


@dataclass
class ProductPrediction:
    detection: Detection
    classification: ClassificationResult


class ShelfAnalysisPipeline:
    def __init__(self, detector: YoloShelfDetector, classifier: ClipProductClassifier):
        self.detector = detector
        self.classifier = classifier

    def analyze_with_detections(
        self,
        image_path: Path,
        detections: List[Detection],
    ) -> List[ProductPrediction]:
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        predictions: List[ProductPrediction] = []
        for det in detections:
            crop = image.crop((det.x1, det.y1, det.x2, det.y2))
            classification = self.classifier.classify_or_none(crop)
            if classification is None:
                continue
            predictions.append(ProductPrediction(detection=det, classification=classification))

        return predictions

    def analyze_image(self, image_path: Path) -> List[ProductPrediction]:
        image_path = Path(image_path)
        detections = self.detector.detect(image_path)
        return self.analyze_with_detections(image_path=image_path, detections=detections)


def summarize_counts(predictions: List[ProductPrediction]) -> Counter:
    summary = Counter()
    for pred in predictions:
        key = f"{pred.classification.product.category} | {pred.classification.product.name}"
        summary[key] += 1
    return summary
