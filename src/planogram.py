import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class PlanogramItem:
    category: str
    product: str
    expected_facings: int
    full_stock_units: int
    reorder_level: int


def _item_key(category: str, product: str) -> str:
    return f"{category} | {product}"


def load_planogram(path: Path) -> Dict:
    if not path.exists():
        return {"items": {}}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"items": {}}

    if not isinstance(payload, dict):
        return {"items": {}}

    items = payload.get("items", {})
    if not isinstance(items, dict):
        payload["items"] = {}
    return payload


def save_planogram(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def set_planogram_item(
    payload: Dict,
    category: str,
    product: str,
    expected_facings: int,
    full_stock_units: int,
    reorder_level: int,
) -> None:
    items = payload.setdefault("items", {})
    key = _item_key(category=category, product=product)
    items[key] = {
        "category": category,
        "product": product,
        "expected_facings": int(expected_facings),
        "full_stock_units": int(full_stock_units),
        "reorder_level": int(reorder_level),
    }


def remove_planogram_item(payload: Dict, category: str, product: str) -> bool:
    items = payload.setdefault("items", {})
    key = _item_key(category=category, product=product)
    if key not in items:
        return False
    del items[key]
    return True


def list_planogram_items(payload: Dict) -> List[PlanogramItem]:
    rows: List[PlanogramItem] = []
    items = payload.get("items", {})
    if not isinstance(items, dict):
        return rows

    for _, raw in items.items():
        if not isinstance(raw, dict):
            continue
        rows.append(
            PlanogramItem(
                category=str(raw.get("category", "Unknown")),
                product=str(raw.get("product", "Unknown")),
                expected_facings=max(1, int(raw.get("expected_facings", 1))),
                full_stock_units=max(0, int(raw.get("full_stock_units", 0))),
                reorder_level=max(0, int(raw.get("reorder_level", 0))),
            )
        )

    rows.sort(key=lambda x: (x.category.lower(), x.product.lower()))
    return rows


def estimate_quantities(counts: Dict[str, int], planogram_payload: Dict) -> List[Dict]:
    rows: List[Dict] = []

    for item in list_planogram_items(planogram_payload):
        key = _item_key(item.category, item.product)
        detected_facings = max(0, int(counts.get(key, 0)))
        expected_facings = max(1, int(item.expected_facings))
        compliance_ratio = min(1.0, detected_facings / expected_facings)
        estimated_units = int(round(compliance_ratio * int(item.full_stock_units)))
        reorder_qty = max(0, int(item.reorder_level) - estimated_units)

        rows.append(
            {
                "Category": item.category,
                "Product": item.product,
                "Detected Facings": detected_facings,
                "Expected Facings": expected_facings,
                "Facing Compliance %": round((detected_facings / expected_facings) * 100, 1),
                "Estimated Units": estimated_units,
                "Reorder Level": int(item.reorder_level),
                "Reorder Qty": reorder_qty,
                "Needs Reorder": "Yes" if estimated_units < int(item.reorder_level) else "No",
            }
        )

    return rows
