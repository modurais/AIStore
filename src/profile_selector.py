from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Tuple

from PIL import Image

from src.classifier import ClipProductClassifier
from src.detector import Detection


def _score_profile(
    image: Image.Image,
    detections: Iterable[Detection],
    classifier: ClipProductClassifier,
) -> float:
    scores = []
    for det in detections:
        crop = image.crop((det.x1, det.y1, det.x2, det.y2))
        result = classifier.classify(crop)
        scores.append(result.score)
    if not scores:
        return 0.0
    return float(mean(scores))


def choose_best_profile(
    image_path: Path,
    detections: Iterable[Detection],
    classifiers_by_profile: Dict[str, ClipProductClassifier],
) -> Tuple[str, Dict[str, float]]:
    image = Image.open(image_path).convert("RGB")
    profile_scores: Dict[str, float] = {}

    for profile_name, classifier in classifiers_by_profile.items():
        profile_scores[profile_name] = _score_profile(
            image=image,
            detections=detections,
            classifier=classifier,
        )

    if not profile_scores:
        raise ValueError("No classifiers were provided for auto profile selection")

    best_profile = max(profile_scores, key=profile_scores.get)
    return best_profile, profile_scores
