from pathlib import Path
from typing import List

import cv2

from src.pipeline import ProductPrediction


def draw_predictions(image_path: Path, predictions: List[ProductPrediction], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for pred in predictions:
        det = pred.detection
        label = pred.classification.product.name
        score = pred.classification.score

        cv2.rectangle(image, (det.x1, det.y1), (det.x2, det.y2), (40, 220, 120), 2)
        text = f"{label} ({score:.2f})"
        cv2.putText(
            image,
            text,
            (det.x1, max(10, det.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)
