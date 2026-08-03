import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class InventoryItem:
    category: str
    product: str
    stock: int
    reorder_level: int
    updated_at: str


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_inventory(path: Path) -> Dict:
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


def save_inventory(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _item_key(category: str, product: str) -> str:
    return f"{category} | {product}"


def set_item(payload: Dict, category: str, product: str, stock: int, reorder_level: int) -> None:
    items = payload.setdefault("items", {})
    key = _item_key(category=category, product=product)
    items[key] = {
        "category": category,
        "product": product,
        "stock": int(stock),
        "reorder_level": int(reorder_level),
        "updated_at": _timestamp(),
    }


def remove_item(payload: Dict, category: str, product: str) -> bool:
    items = payload.setdefault("items", {})
    key = _item_key(category=category, product=product)
    if key not in items:
        return False
    del items[key]
    return True


def adjust_stock(payload: Dict, counts: Dict[str, int], direction: str) -> None:
    items = payload.setdefault("items", {})
    sign = -1 if direction == "deduct" else 1

    for key, value in counts.items():
        amount = int(value)
        if amount <= 0:
            continue

        if key in items:
            current_stock = int(items[key].get("stock", 0))
            items[key]["stock"] = current_stock + (sign * amount)
            items[key]["updated_at"] = _timestamp()
            continue

        if "|" not in key:
            continue
        category, product = [part.strip() for part in key.split("|", maxsplit=1)]
        items[key] = {
            "category": category,
            "product": product,
            "stock": sign * amount,
            "reorder_level": 0,
            "updated_at": _timestamp(),
        }


def reconcile_stock(payload: Dict, counts: Dict[str, int]) -> None:
    items = payload.setdefault("items", {})

    for key, value in counts.items():
        amount = int(value)
        if amount < 0:
            continue

        if key in items:
            items[key]["stock"] = amount
            items[key]["updated_at"] = _timestamp()
            continue

        if "|" not in key:
            continue
        category, product = [part.strip() for part in key.split("|", maxsplit=1)]
        items[key] = {
            "category": category,
            "product": product,
            "stock": amount,
            "reorder_level": 0,
            "updated_at": _timestamp(),
        }


def list_items(payload: Dict) -> List[InventoryItem]:
    rows: List[InventoryItem] = []
    items = payload.get("items", {})
    if not isinstance(items, dict):
        return rows

    for _, raw in items.items():
        if not isinstance(raw, dict):
            continue
        rows.append(
            InventoryItem(
                category=str(raw.get("category", "Unknown")),
                product=str(raw.get("product", "Unknown")),
                stock=int(raw.get("stock", 0)),
                reorder_level=int(raw.get("reorder_level", 0)),
                updated_at=str(raw.get("updated_at", "")),
            )
        )

    rows.sort(key=lambda x: (x.category.lower(), x.product.lower()))
    return rows