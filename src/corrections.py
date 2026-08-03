import json
import os
from pathlib import Path
from typing import Dict, List

from src.classifier import ClassificationResult
from src.config import ProductLabel, normalize_category
from src.pipeline import ProductPrediction


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def load_corrections(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_corrections(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace prevents partial writes when Streamlit reruns quickly.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _profile_bucket(payload: Dict, profile: str) -> Dict:
    profiles = payload.setdefault("profiles", {})
    return profiles.setdefault(profile, {})


def set_profile_correction(
    payload: Dict,
    profile: str,
    wrong_name: str,
    corrected_label: ProductLabel,
) -> None:
    bucket = _profile_bucket(payload=payload, profile=profile)
    key = _normalize_name(wrong_name)
    bucket[key] = {
        "wrong_name": wrong_name,
        "category": normalize_category(corrected_label.category),
        "name": corrected_label.name,
        "prompt": corrected_label.prompt,
    }


def get_profile_corrections(payload: Dict, profile: str) -> Dict[str, ProductLabel]:
    profiles = payload.get("profiles", {})
    bucket = profiles.get(profile, {})
    result: Dict[str, ProductLabel] = {}

    for wrong_name, data in bucket.items():
        if not isinstance(data, dict):
            continue
        category = normalize_category(str(data.get("category", "Unknown")))
        name = str(data.get("name", wrong_name)).strip() or wrong_name
        prompt = str(data.get("prompt", name)).strip() or name
        result[str(wrong_name)] = ProductLabel(category=category, name=name, prompt=prompt)

    return result


def list_profile_corrections(payload: Dict, profile: str) -> List[Dict[str, str]]:
    profiles = payload.get("profiles", {})
    bucket = profiles.get(profile, {})
    rows: List[Dict[str, str]] = []

    for key, data in bucket.items():
        if not isinstance(data, dict):
            continue
        wrong_name = str(data.get("wrong_name", key)).strip() or str(key)
        category = normalize_category(str(data.get("category", "Unknown")))
        name = str(data.get("name", wrong_name)).strip() or wrong_name
        prompt = str(data.get("prompt", name)).strip() or name
        rows.append(
            {
                "wrong_name": wrong_name,
                "category": category,
                "name": name,
                "prompt": prompt,
            }
        )

    rows.sort(key=lambda r: r["wrong_name"].lower())
    return rows


def apply_corrections(
    predictions: List[ProductPrediction],
    profile: str,
    corrections_payload: Dict,
) -> List[ProductPrediction]:
    profile_map = get_profile_corrections(payload=corrections_payload, profile=profile)
    if not profile_map:
        return predictions

    corrected_predictions: List[ProductPrediction] = []
    for pred in predictions:
        product_name = pred.classification.product.name
        override = profile_map.get(product_name)
        if override is None:
            override = profile_map.get(_normalize_name(product_name))
        if override is None:
            corrected_predictions.append(pred)
            continue

        corrected_predictions.append(
            ProductPrediction(
                detection=pred.detection,
                classification=ClassificationResult(
                    product=override,
                    score=pred.classification.score,
                ),
            )
        )

    return corrected_predictions
