from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ProductLabel:
    category: str
    name: str
    prompt: str


CATEGORY_ALIASES: Dict[str, str] = {
    "alcoholic": "Soft Drink",
    "alcoholic drink": "Soft Drink",
    "alcoholic drinks": "Soft Drink",
    "soft drinks": "Soft Drink",
    "soft drink": "Soft Drink",
    "beverages": "Beverages",
    "beverage": "Beverages",
    "personal care": "Personal Care",
    "energy drinks": "Energy Drinks",
    "sports drinks": "Sports Drinks",
}


def normalize_category(category: str) -> str:
    raw = str(category).strip()
    if not raw:
        return "Unknown"
    key = " ".join(raw.lower().split())
    return CATEGORY_ALIASES.get(key, raw)


# Brand-focused beverage labels adapted from Weights/yoloclip_july25.py.
BEVERAGE_CANDIDATE_PRODUCTS = [
    ProductLabel(
        category="Beverages",
        name="soda can",
        prompt="soda can",
    ),
    ProductLabel(
        category="Beverages",
        name="juice box",
        prompt="juice box",
    ),
    ProductLabel(
        category="Beverages",
        name="water bottle",
        prompt="water bottle",
    ),
    ProductLabel(
        category="Soft Drinks",
        name="Coca Cola can",
        prompt="Coca Cola can",
    ),
    ProductLabel(
        category="Soft Drinks",
        name="Sprite can",
        prompt="Sprite can",
    ),
    ProductLabel(
        category="Soft Drinks",
        name="Fanta can",
        prompt="Fanta can",
    ),
    ProductLabel(
        category="Soft Drinks",
        name="Pepsi can",
        prompt="Pepsi can",
    ),
    ProductLabel(
        category="Soft Drinks",
        name="Green Sands bottle",
        prompt="Green Sands bottle",
    ),
    ProductLabel(
        category="Soft Drinks",
        name="7UP can",
        prompt="7UP can",
    ),
    ProductLabel(
        category="Energy Drinks",
        name="energy drink can",
        prompt="energy drink can",
    ),
    ProductLabel(
        category="Soft Drink",
        name="wine bottle",
        prompt="wine bottle",
    ),
    ProductLabel(
        category="Soft Drink",
        name="beer bottle",
        prompt="beer bottle",
    ),
    ProductLabel(
        category="Dairy",
        name="milk carton",
        prompt="milk carton",
    ),
    ProductLabel(
        category="Beverages",
        name="coffee drink",
        prompt="coffee drink",
    ),
    ProductLabel(
        category="Beverages",
        name="tea drink",
        prompt="tea drink",
    ),
    ProductLabel(
        category="Beverages",
        name="fruit juice",
        prompt="fruit juice",
    ),
    ProductLabel(
        category="Sports Drinks",
        name="sports drink",
        prompt="sports drink",
    ),
]


# Shampoo/personal-care focused labels for shelves like shampoo.jpeg.
SHAMPOO_CANDIDATE_PRODUCTS = [
    ProductLabel(
        category="Personal Care",
        name="Head and Shoulders shampoo",
        prompt="Head and Shoulders shampoo",
    ),
    ProductLabel(
        category="Personal Care",
        name="Herbal Essences shampoo",
        prompt="Herbal Essences shampoo",
    ),
    ProductLabel(
        category="Personal Care",
        name="TRESemme shampoo",
        prompt="TRESemmé shampoo",
    ),
    ProductLabel(
        category="Personal Care",
        name="Dove shampoo",
        prompt="Dove shampoo",
    ),
    ProductLabel(
        category="Personal Care",
        name="Pantene shampoo",
        prompt="Pantene shampoo",
    ),
    ProductLabel(
        category="Personal Care",
        name="conditioner",
        prompt="conditioner",
    ),
    ProductLabel(
        category="Personal Care",
        name="hair spray",
        prompt="hair spray",
    ),
    ProductLabel(
        category="Personal Care",
        name="hair product",
        prompt="hair product",
    ),
    ProductLabel(
        category="Personal Care",
        name="lotion bottle",
        prompt="lotion bottle",
    ),
    ProductLabel(
        category="Personal Care",
        name="personal care item",
        prompt="personal care item",
    ),
    ProductLabel(
        category="Personal Care",
        name="cosmetic product",
        prompt="cosmetic product",
    ),
    ProductLabel(
        category="Personal Care",
        name="container",
        prompt="container",
    ),
]


DEFAULT_LABEL_PROFILE = "beverage"

LABEL_PROFILES = {
    "beverage": BEVERAGE_CANDIDATE_PRODUCTS,
    "shampoo": SHAMPOO_CANDIDATE_PRODUCTS,
}


def get_candidate_products(profile: str) -> List[ProductLabel]:
    try:
        return LABEL_PROFILES[profile]
    except KeyError as exc:
        available = ", ".join(sorted(LABEL_PROFILES.keys()))
        raise ValueError(f"Unknown label profile '{profile}'. Available: {available}") from exc


# Backward-compatible default alias used by existing imports.
CANDIDATE_PRODUCTS = get_candidate_products(DEFAULT_LABEL_PROFILE)

DEFAULT_DETECTOR_WEIGHTS = Path("Weights/best.pt")
DEFAULT_DETECTION_CONFIDENCE = 0.10
DEFAULT_DETECTION_IOU = 0.30
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CORRECTIONS_PATH = Path("data/label_corrections.json")
DEFAULT_INVENTORY_PATH = Path("data/inventory.json")
DEFAULT_PLANOGRAM_PATH = Path("data/planogram.json")

# Prompts used to reject detections that look like shelf signage/price tags/text blocks.
NON_PRODUCT_PROMPTS = [
    "price tag with printed text",
    "discount sticker or promotion label",
    "store shelf sign",
    "advertisement poster",
    "text-only product label",
    "barcode label sticker",
]

NON_PRODUCT_REJECT_MARGIN = 0.02
