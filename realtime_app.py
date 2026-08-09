import json
import tempfile
import base64
import mimetypes
from html import escape
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

try:
    from src.config import (
        DEFAULT_CLIP_MODEL,
        DEFAULT_CORRECTIONS_PATH,
        DEFAULT_DETECTION_CONFIDENCE,
        DEFAULT_DETECTION_IOU,
        DEFAULT_DETECTOR_WEIGHTS,
        DEFAULT_INVENTORY_PATH,
        DEFAULT_PLANOGRAM_PATH,
        NON_PRODUCT_PROMPTS,
        NON_PRODUCT_REJECT_MARGIN,
        ProductLabel,
        get_candidate_products,
        normalize_category,
    )
except ImportError:
    # Backward compatibility: older config.py may not define DEFAULT_PLANOGRAM_PATH.
    from src.config import (
        DEFAULT_CLIP_MODEL,
        DEFAULT_CORRECTIONS_PATH,
        DEFAULT_DETECTION_CONFIDENCE,
        DEFAULT_DETECTION_IOU,
        DEFAULT_DETECTOR_WEIGHTS,
        DEFAULT_INVENTORY_PATH,
        NON_PRODUCT_PROMPTS,
        NON_PRODUCT_REJECT_MARGIN,
        ProductLabel,
        get_candidate_products,
        normalize_category,
    )

    DEFAULT_PLANOGRAM_PATH = Path("data/planogram.json")
from src.inventory import (
    adjust_stock,
    list_items,
    load_inventory,
    remove_item,
    save_inventory,
    set_item,
)
from src.planogram import (
    estimate_quantities,
    list_planogram_items,
    load_planogram,
    remove_planogram_item,
    save_planogram,
    set_planogram_item,
)

try:
    from src.inventory import reconcile_stock
except ImportError:
    # Fallback for environments still loading an older src.inventory module.
    def reconcile_stock(payload: Dict, counts: Dict[str, int]) -> None:
        items = payload.setdefault("items", {})
        now = datetime.now().isoformat(timespec="seconds")
        for key, value in counts.items():
            amount = int(value)
            if amount < 0:
                continue
            if key in items:
                items[key]["stock"] = amount
                items[key]["updated_at"] = now
                continue
            if "|" not in key:
                continue
            category, product = [part.strip() for part in key.split("|", maxsplit=1)]
            items[key] = {
                "category": category,
                "product": product,
                "stock": amount,
                "reorder_level": 0,
                "updated_at": now,
            }

if TYPE_CHECKING:
    from src.classifier import ClipProductClassifier
    from src.detector import YoloShelfDetector
    from src.pipeline import ProductPrediction


st.set_page_config(page_title="Shelf Product Counter", page_icon="🛒", layout="wide")


DEFAULT_REORDER_LEVEL = 80

BRANDING = {
    "college_name": "Your College Name",
    "department_name": "Department of AI & DS",
    "project_title": "Shelf Intelligence for Retail",
    "team_label": "Group 1 Capstone",
    "guide_label": "Guided by: Faculty Name",
    "logo_path": "data/college_logo.png",
}


def _logo_data_uri(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return None
    return f"data:{mime};base64,{encoded}"


def inject_professional_theme() -> None:
    st.markdown(
    """
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

                :root {
                    --brand-ink: #0f2f44;
                    --brand-teal: #0f766e;
                    --brand-slate: #1f2937;
                    --brand-border: #dbe3ea;
                    --brand-surface: #ffffff;
                    --brand-bg: #f4f8fb;
                    --brand-accent: #0b5ea8;
                }

                html, body, [class*="css"], .stApp {
                    font-family: 'Manrope', sans-serif;
                }

                .stApp {
                    background:
                        radial-gradient(circle at 92% 6%, rgba(15, 118, 110, 0.07), transparent 26%),
                        radial-gradient(circle at 10% 90%, rgba(11, 94, 168, 0.09), transparent 32%),
                        var(--brand-bg);
                }

                [data-testid="stSidebar"] {
                    background: linear-gradient(180deg, #0f2f44 0%, #133a54 100%);
                    border-right: 1px solid rgba(255, 255, 255, 0.08);
                }

                [data-testid="stSidebar"] * {
                    color: #eef4f8 !important;
                }

                [data-testid="stSidebar"] .stTextInput input,
                [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
                [data-testid="stSidebar"] .stNumberInput input {
                    background: rgba(255, 255, 255, 0.1) !important;
                    border: 1px solid rgba(255, 255, 255, 0.24) !important;
                    color: #ffffff !important;
                }

                .hero-shell {
                    background: linear-gradient(120deg, #0f2f44 0%, #0b5ea8 65%, #0f766e 100%);
                    border-radius: 18px;
                    padding: 22px 24px;
                    box-shadow: 0 16px 34px rgba(12, 37, 58, 0.23);
                    margin-bottom: 14px;
                }

                .hero-grid {
                    display: grid;
                    grid-template-columns: 90px 1fr;
                    gap: 16px;
                    align-items: center;
                }

                .hero-logo {
                    width: 84px;
                    height: 84px;
                    border-radius: 16px;
                    background: rgba(255, 255, 255, 0.16);
                    border: 1px solid rgba(255, 255, 255, 0.32);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                    font-weight: 800;
                    color: #f8fdff;
                    font-size: 1.55rem;
                }

                .hero-logo img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }

                .hero-title {
                    color: #f5fbff;
                    font-weight: 800;
                    font-size: 1.6rem;
                    line-height: 1.2;
                    margin: 0;
                }

                .hero-subtitle {
                    color: #d2e6f5;
                    font-weight: 500;
                    font-size: 0.95rem;
                    margin-top: 7px;
                }

                .hero-meta {
                    margin-top: 8px;
                    color: #e9f6ff;
                    font-size: 0.83rem;
                    font-weight: 600;
                    letter-spacing: 0.04em;
                    text-transform: uppercase;
                }

                .hero-chip {
                    display: inline-block;
                    margin-top: 10px;
                    margin-right: 8px;
                    font-size: 0.74rem;
                    font-weight: 700;
                    letter-spacing: 0.04em;
                    color: #f4fbff;
                    background: rgba(255, 255, 255, 0.18);
                    border: 1px solid rgba(255, 255, 255, 0.28);
                    border-radius: 999px;
                    padding: 4px 10px;
                }

                .section-kicker {
                    margin-top: 4px;
                    margin-bottom: 10px;
                    color: #334155;
                    font-size: 0.82rem;
                    font-weight: 700;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                }

                .stButton button,
                .stDownloadButton button {
                    border-radius: 12px !important;
                    border: 1px solid var(--brand-border) !important;
                    font-weight: 700 !important;
                    transition: all 0.2s ease !important;
                }

                .stButton button:hover,
                .stDownloadButton button:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 8px 18px rgba(15, 47, 68, 0.12);
                }

                .stButton button[kind="primary"] {
                    background: linear-gradient(120deg, #0b5ea8 0%, #0f766e 100%) !important;
                    color: #ffffff !important;
                    border: none !important;
                }

                [data-testid="stMetric"] {
                    background: var(--brand-surface);
                    border: 1px solid var(--brand-border);
                    border-radius: 14px;
                    padding: 12px 14px;
                    box-shadow: 0 4px 14px rgba(15, 47, 68, 0.08);
                }

                [data-testid="stMetricLabel"] {
                    font-weight: 700;
                    color: #334155;
                }

                [data-testid="stMetricValue"] {
                    color: var(--brand-ink);
                    font-weight: 800;
                }

                [data-testid="stDataFrame"],
                .stTable {
                    border: 1px solid var(--brand-border);
                    border-radius: 12px;
                    overflow: hidden;
                    background: var(--brand-surface);
                }

                .stAlert {
                    border-radius: 12px;
                    border: 1px solid #d7e2ec;
                }

                .block-container {
                    padding-top: 1.2rem;
                    padding-bottom: 2.3rem;
                }
                </style>
                """,
        unsafe_allow_html=True,
    )


def render_page_hero(branding: Optional[Dict[str, str]] = None) -> None:
    effective_branding = BRANDING.copy()
    if branding:
        effective_branding.update(branding)

    logo_path = Path(str(effective_branding.get("logo_path", ""))).expanduser()
    logo_uri = _logo_data_uri(logo_path)
    college_name = str(effective_branding.get("college_name", "Your College Name"))
    department_name = str(effective_branding.get("department_name", "Department"))
    project_title = str(effective_branding.get("project_title", "Shelf Intelligence for Retail"))
    team_label = str(effective_branding.get("team_label", "Group 1 Capstone"))
    guide_label = str(effective_branding.get("guide_label", "Guided by: Faculty Name"))

    logo_html = (
        f'<div class="hero-logo"><img src="{logo_uri}" alt="College Logo"/></div>'
        if logo_uri
        else '<div class="hero-logo">AI</div>'
    )

    # Replace placeholders in one final render to keep template readable.
    rendered = (
        """
        <div class="hero-shell">
            <div class="hero-grid">
                __LOGO__
                <div>
                    <h1 class="hero-title">__PROJECT__</h1>
                    <div class="hero-subtitle">YOLO + CLIP powered detection, planogram analytics, and inventory actions in one workspace.</div>
                    <div class="hero-meta">__COLLEGE__ • __DEPARTMENT__</div>
                    <span class="hero-chip">__TEAM__</span>
                    <span class="hero-chip">__GUIDE__</span>
                </div>
            </div>
        </div>
        """
        .replace("__LOGO__", logo_html)
        .replace("__PROJECT__", escape(project_title))
        .replace("__COLLEGE__", escape(college_name))
        .replace("__DEPARTMENT__", escape(department_name))
        .replace("__TEAM__", escape(team_label))
        .replace("__GUIDE__", escape(guide_label))
    )
    st.markdown(rendered, unsafe_allow_html=True)


def build_count_table(counts: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for label, count in counts.items():
        category, product = [part.strip() for part in label.split("|", maxsplit=1)]
        rows.append({"Category": category, "Product": product, "Count": count})
    rows.sort(key=lambda r: (r["Category"], r["Product"]))
    return pd.DataFrame(rows)


def build_inventory_table(inventory_payload: Dict) -> pd.DataFrame:
    rows = []
    for item in list_items(inventory_payload):
        reorder_level = int(item.reorder_level)
        if reorder_level <= 0:
            reorder_level = DEFAULT_REORDER_LEVEL
        reorder_qty = max(0, reorder_level - int(item.stock))
        rows.append(
            {
                "Category": item.category,
                "Product": item.product,
                "Stock": item.stock,
                "Reorder Level": reorder_level,
                "Reorder Qty": reorder_qty,
                "Needs Reorder": "Yes" if item.stock <= reorder_level else "No",
                "Updated At": item.updated_at,
            }
        )
    return pd.DataFrame(rows)


def build_planogram_table(planogram_payload: Dict) -> pd.DataFrame:
    rows = []
    for item in list_planogram_items(planogram_payload):
        rows.append(
            {
                "Category": item.category,
                "Product": item.product,
                "Expected Facings": int(item.expected_facings),
                "Full Stock Units": int(item.full_stock_units),
                "Reorder Level": int(item.reorder_level),
            }
        )
    return pd.DataFrame(rows)


def build_inventory_report_payload(
    inventory_df: pd.DataFrame,
    latest_counts: Dict[str, int],
    latest_planogram_rows: List[Dict],
) -> Dict:
    low_stock_df = (
        inventory_df[inventory_df["Needs Reorder"] == "Yes"]
        if not inventory_df.empty and "Needs Reorder" in inventory_df.columns
        else pd.DataFrame()
    )
    by_category = (
        inventory_df.groupby("Category", as_index=False)["Stock"].sum().to_dict(orient="records")
        if not inventory_df.empty
        else []
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inventory_summary": {
            "tracked_skus": int(len(inventory_df)),
            "on_hand_units": int(inventory_df["Stock"].sum()) if not inventory_df.empty else 0,
            "low_stock_alerts": int(len(low_stock_df)),
            "latest_detection_products": int(sum(latest_counts.values())) if latest_counts else 0,
        },
        "stock_by_category": by_category,
        "low_stock_items": low_stock_df.to_dict(orient="records") if not low_stock_df.empty else [],
        "latest_planogram_estimates": latest_planogram_rows,
        "latest_detection_counts": latest_counts,
    }
    return payload


def render_inventory_dashboards(inventory_df: pd.DataFrame, latest_planogram_rows: List[Dict]) -> None:
    st.markdown("### Inventory Dashboards")
    if inventory_df.empty:
        st.info("Add inventory items to view dashboard analytics.")
        return

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Stock by Category**")
        by_category = inventory_df.groupby("Category", as_index=False)["Stock"].sum().sort_values("Stock", ascending=False)
        st.bar_chart(by_category.set_index("Category")["Stock"], use_container_width=True)

    with d2:
        st.markdown("**Low Stock Alerts by Category**")
        low_df = inventory_df[inventory_df["Needs Reorder"] == "Yes"]
        if low_df.empty:
            st.success("No low-stock categories right now.")
        else:
            low_by_category = low_df.groupby("Category", as_index=False).size().rename(columns={"size": "Low Stock Items"})
            st.bar_chart(low_by_category.set_index("Category")["Low Stock Items"], use_container_width=True)

    st.markdown("**Top Reorder Priority**")
    reorder_top = inventory_df.sort_values("Reorder Qty", ascending=False).head(8)
    st.dataframe(
        reorder_top[["Category", "Product", "Stock", "Reorder Level", "Reorder Qty", "Needs Reorder"]],
        use_container_width=True,
        hide_index=True,
    )

    if latest_planogram_rows:
        st.markdown("**Planogram Compliance Snapshot**")
        plan_df = pd.DataFrame(latest_planogram_rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Facing Compliance", f"{plan_df['Facing Compliance %'].mean():.1f}%")
        c2.metric("Estimated Units", int(plan_df["Estimated Units"].sum()))
        c3.metric("Planogram Reorder Alerts", int((plan_df["Needs Reorder"] == "Yes").sum()))


def build_planogram_diagram_html(
    planogram_payload: Dict,
    counts: Dict[str, int],
    max_slots: int = 20,
) -> str:
    items = list_planogram_items(planogram_payload)
    if not items:
        return ""

    lines = [
        "<style>",
        ".pog-wrap{display:grid;gap:10px;margin:6px 0 12px 0}",
        ".pog-row{border:1px solid #d8dce6;border-radius:10px;padding:10px;background:#fbfcfe}",
        ".pog-title{font-weight:600;font-size:0.95rem;color:#1f2937}",
        ".pog-meta{font-size:0.82rem;color:#475569;margin-top:2px;margin-bottom:8px}",
        ".pog-slots{display:grid;grid-template-columns:repeat(auto-fill,minmax(12px,1fr));gap:4px;max-width:380px}",
        ".pog-slot{height:14px;border-radius:3px;border:1px solid #cbd5e1;background:#ffffff}",
        ".pog-slot-filled{background:#16a34a;border-color:#15803d}",
        ".pog-over{font-size:0.78rem;color:#b45309;margin-left:8px}",
        "</style>",
        '<div class="pog-wrap">',
    ]

    for item in items:
        key = f"{item.category} | {item.product}"
        detected_facings = max(0, int(counts.get(key, 0)))
        expected_facings = max(1, int(item.expected_facings))
        full_stock_units = max(0, int(item.full_stock_units))

        compliance_ratio = min(1.0, detected_facings / expected_facings)
        estimated_units = int(round(compliance_ratio * full_stock_units))

        slot_count = max(1, min(max_slots, expected_facings))
        filled_slots = min(slot_count, int(round(compliance_ratio * slot_count)))
        over_facings = max(0, detected_facings - expected_facings)

        slot_blocks = []
        for idx in range(slot_count):
            cls = "pog-slot pog-slot-filled" if idx < filled_slots else "pog-slot"
            slot_blocks.append(f'<div class="{cls}"></div>')

        title = f"{escape(item.category)} / {escape(item.product)}"
        meta = (
            f"Detected {detected_facings} of {expected_facings} facings | "
            f"Estimated units: {estimated_units}"
        )
        over_text = f'<span class="pog-over">+{over_facings} extra facings</span>' if over_facings > 0 else ""

        lines.extend(
            [
                '<div class="pog-row">',
                f'<div class="pog-title">{title}</div>',
                f'<div class="pog-meta">{meta}{over_text}</div>',
                f'<div class="pog-slots">{"".join(slot_blocks)}</div>',
                "</div>",
            ]
        )

    lines.append("</div>")
    return "".join(lines)


def render_planogram_diagram(planogram_payload: Dict, counts: Dict[str, int]) -> None:
    html = build_planogram_diagram_html(planogram_payload=planogram_payload, counts=counts)
    if not html:
        st.info("No planogram items configured yet.")
        return
    st.markdown(html, unsafe_allow_html=True)


def sync_inventory_from_planogram_estimates(inventory_payload: Dict, estimate_rows: List[Dict]) -> None:
    for row in estimate_rows:
        category = str(row["Category"])
        product = str(row["Product"])
        estimated_units = int(row["Estimated Units"])
        reorder_level = int(row["Reorder Level"])
        set_item(
            payload=inventory_payload,
            category=category,
            product=product,
            stock=estimated_units,
            reorder_level=reorder_level,
        )


def apply_default_reorder_levels_for_detected_items(payload: Dict, counts: Dict[str, int]) -> None:
    items = payload.setdefault("items", {})
    for key in counts.keys():
        row = items.get(key)
        if not isinstance(row, dict):
            continue
        current_level = int(row.get("reorder_level", 0))
        if current_level <= 0:
            row["reorder_level"] = DEFAULT_REORDER_LEVEL


def _normalize_prompt_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _find_inventory_matches(prompt_norm: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    if inventory_df.empty:
        return inventory_df

    frame = inventory_df.copy()
    frame["_product_norm"] = frame["Product"].astype(str).map(_normalize_prompt_text)
    frame["_category_norm"] = frame["Category"].astype(str).map(_normalize_prompt_text)

    unique_products = sorted(
        [text for text in frame["_product_norm"].dropna().unique() if text],
        key=len,
        reverse=True,
    )
    for product_norm in unique_products:
        if product_norm in prompt_norm:
            return frame[frame["_product_norm"] == product_norm]

    tokens = [token for token in prompt_norm.split() if len(token) >= 3]
    if not tokens:
        return frame.iloc[0:0]

    mask = pd.Series(False, index=frame.index)
    for token in tokens:
        mask = mask | frame["_product_norm"].str.contains(token, na=False)
    return frame[mask]


def answer_inventory_prompt(prompt_text: str, inventory_df: pd.DataFrame) -> Dict:
    prompt_norm = _normalize_prompt_text(prompt_text)

    if not prompt_norm:
        return {
            "kind": "info",
            "message": "Try prompts like: 'stock of Pantene shampoo', 'reorder level of Himalaya', or 'which items need reorder'.",
        }

    if inventory_df.empty:
        return {
            "kind": "warning",
            "message": "Inventory is empty. Add items first, then ask stock or reorder questions.",
        }

    asks_summary = any(word in prompt_norm for word in ["summary", "overview", "health", "status"])
    asks_stock = "stock" in prompt_norm
    asks_reorder_level = "reorder level" in prompt_norm or "order level" in prompt_norm
    asks_reorder_qty = "reorder qty" in prompt_norm or "reorder quantity" in prompt_norm
    asks_reorder_state = any(
        phrase in prompt_norm
        for phrase in ["need reorder", "needs reorder", "reorder or not", "low stock", "below reorder"]
    )
    asks_list = any(word in prompt_norm for word in ["show", "list", "which", "what", "all"])

    if asks_summary and not (asks_stock or asks_reorder_level or asks_reorder_state):
        total_skus = len(inventory_df)
        total_stock = int(inventory_df["Stock"].sum())
        low_stock = int((inventory_df["Needs Reorder"] == "Yes").sum())
        return {
            "kind": "success",
            "message": f"Inventory summary: {total_skus} SKU(s), {total_stock} unit(s) on hand, {low_stock} SKU(s) need reorder.",
        }

    if asks_reorder_state and asks_list:
        low_df = inventory_df[inventory_df["Needs Reorder"] == "Yes"][
            ["Category", "Product", "Stock", "Reorder Level", "Reorder Qty", "Needs Reorder"]
        ]
        if low_df.empty:
            return {
                "kind": "success",
                "message": "No items currently need reorder.",
            }
        return {
            "kind": "table",
            "message": f"{len(low_df)} item(s) currently need reorder.",
            "table": low_df,
        }

    matches = _find_inventory_matches(prompt_norm, inventory_df)
    if matches.empty:
        sample_names = ", ".join(inventory_df["Product"].astype(str).head(5).tolist())
        return {
            "kind": "warning",
            "message": f"Could not find a matching product. Try one of these: {sample_names}",
        }

    if len(matches) > 1:
        options = ", ".join(matches["Product"].astype(str).drop_duplicates().head(6).tolist())
        return {
            "kind": "warning",
            "message": f"Multiple products matched. Please be more specific: {options}",
        }

    row = matches.iloc[0]
    category = str(row["Category"])
    product = str(row["Product"])
    stock = int(row["Stock"])
    reorder_level = int(row["Reorder Level"])
    reorder_qty = int(row["Reorder Qty"])
    needs_reorder = "Yes" if stock <= reorder_level else "No"

    if asks_reorder_level:
        return {
            "kind": "success",
            "message": f"{product} ({category}) reorder level is {reorder_level}.",
        }

    if asks_reorder_state:
        return {
            "kind": "success",
            "message": f"{product} ({category}) needs reorder: {needs_reorder}. Stock={stock}, Reorder level={reorder_level}, Reorder qty={reorder_qty}.",
        }

    if asks_reorder_qty:
        return {
            "kind": "success",
            "message": f"{product} ({category}) reorder quantity is {reorder_qty} unit(s).",
        }

    if asks_stock or "units" in prompt_norm:
        return {
            "kind": "success",
            "message": f"{product} ({category}) current stock is {stock} unit(s).",
        }

    return {
        "kind": "info",
        "message": "I can answer stock, reorder level, and needs-reorder questions. Example: 'does Pantene shampoo need reorder'.",
    }


@st.cache_resource(show_spinner=False)
def load_pipeline(
    detector_weights: str,
    clip_model: str,
    det_conf: float,
    det_iou: float,
) -> Tuple["YoloShelfDetector", Dict[str, "ClipProductClassifier"]]:
    from src.classifier import ClipProductClassifier
    from src.detector import YoloShelfDetector

    detector = YoloShelfDetector(
        model_path=Path(detector_weights),
        confidence=det_conf,
        iou=det_iou,
    )
    classifiers = {
        "beverage": ClipProductClassifier(
            model_name=clip_model,
            candidate_labels=get_candidate_products("beverage"),
            negative_prompts=NON_PRODUCT_PROMPTS,
            reject_margin=NON_PRODUCT_REJECT_MARGIN,
        ),
        "shampoo": ClipProductClassifier(
            model_name=clip_model,
            candidate_labels=get_candidate_products("shampoo"),
            negative_prompts=NON_PRODUCT_PROMPTS,
            reject_margin=NON_PRODUCT_REJECT_MARGIN,
        ),
    }
    return detector, classifiers


def save_uploaded_image(data: bytes, suffix: str) -> Path:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(data)
    tmp_file.flush()
    tmp_file.close()
    return Path(tmp_file.name)


def run_analysis(
    detector: "YoloShelfDetector",
    classifiers: Dict[str, "ClipProductClassifier"],
    image_path: Path,
    corrections_payload: Dict,
) -> Tuple[List["ProductPrediction"], Dict[str, int], Path, str, Dict[str, float]]:
    from src.corrections import apply_corrections
    from src.pipeline import ProductPrediction, ShelfAnalysisPipeline, summarize_counts
    from src.profile_selector import choose_best_profile
    from src.visualize import draw_predictions

    detections = detector.detect(image_path)
    selected_profile, profile_scores = choose_best_profile(
        image_path=image_path,
        detections=detections,
        classifiers_by_profile=classifiers,
    )
    pipeline = ShelfAnalysisPipeline(detector=detector, classifier=classifiers[selected_profile])
    predictions = pipeline.analyze_with_detections(image_path=image_path, detections=detections)
    predictions = apply_corrections(
        predictions=predictions,
        profile=selected_profile,
        corrections_payload=corrections_payload,
    )
    counts = {key: int(value) for key, value in summarize_counts(predictions).items()}

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"annotated_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    draw_predictions(image_path=image_path, predictions=predictions, output_path=output_path)
    return predictions, counts, output_path, selected_profile, profile_scores


def main() -> None:
    inject_professional_theme()
    branding_values = BRANDING.copy()

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None

    with st.sidebar:
        st.markdown("### Workspace")
        screen = st.radio(
            "Choose screen",
            ["Detection Studio", "Inventory Dashboard", "Planogram Dashboard", "AI Query"],
            index=0,
        )
        st.divider()
        st.markdown("### Model Settings")
        detector_weights = st.text_input(
            "YOLO weights path",
            value=str(DEFAULT_DETECTOR_WEIGHTS),
        )
        clip_model = st.text_input("CLIP model", value=DEFAULT_CLIP_MODEL)
        det_conf = st.slider("Detection confidence", 0.05, 0.90, DEFAULT_DETECTION_CONFIDENCE, 0.05)
        det_iou = st.slider("NMS IoU", 0.10, 0.90, DEFAULT_DETECTION_IOU, 0.05)
        st.caption("Label profile is selected automatically (beverage vs shampoo).")
        st.divider()
        auto_sync_inventory = st.checkbox(
            "Auto-sync inventory from detection",
            value=True,
            help="When enabled, detected counts directly become stock values for detected products.",
        )
        use_planogram_sync = st.checkbox(
            "Use planogram quantity estimation",
            value=False,
            help=(
                "Estimate stock using detected facings against expected facings "
                "and full-stock units from planogram data."
            ),
        )
        st.divider()
        st.markdown("### Branding")
        branding_values["project_title"] = st.text_input("Project title", value=str(BRANDING["project_title"]))
        branding_values["college_name"] = st.text_input("College name", value=str(BRANDING["college_name"]))
        branding_values["department_name"] = st.text_input("Department", value=str(BRANDING["department_name"]))
        branding_values["team_label"] = st.text_input("Team label", value=str(BRANDING["team_label"]))
        branding_values["guide_label"] = st.text_input("Guide label", value=str(BRANDING["guide_label"]))
        branding_values["logo_path"] = st.text_input(
            "Logo path (optional)",
            value=str(BRANDING["logo_path"]),
            help="Example: data/college_logo.png",
        )

    render_page_hero(branding=branding_values)
    st.caption("No login required. Use Detection Studio or Inventory Center directly.")

    latest_result = st.session_state.latest_result
    latest_counts = latest_result["counts"] if latest_result else {}

    if screen == "AI Query":
        st.markdown('<div class="section-kicker">Assistant</div>', unsafe_allow_html=True)
        st.subheader("AI Query")
        st.caption("Ask inventory questions in natural language.")

        inventory_payload = load_inventory(Path(DEFAULT_INVENTORY_PATH))
        inventory_df = build_inventory_table(inventory_payload)

        a1, a2, a3 = st.columns(3)
        a1.metric("Tracked SKUs", int(len(inventory_df)))
        a2.metric("On-Hand Units", int(inventory_df["Stock"].sum()) if not inventory_df.empty else 0)
        a3.metric(
            "Low Stock Alerts",
            int((inventory_df["Needs Reorder"] == "Yes").sum()) if not inventory_df.empty else 0,
        )

        if "ai_query_text" not in st.session_state:
            st.session_state.ai_query_text = ""

        default_product = "Pantene shampoo"
        if not inventory_df.empty and "Product" in inventory_df.columns:
            try:
                default_product = str(inventory_df.iloc[0]["Product"])
            except Exception:
                pass

        quick_queries = [
            "inventory summary",
            "which items need reorder",
            f"stock of {default_product}",
            f"reorder level of {default_product}",
            f"reorder quantity of {default_product}",
            "latest detection summary",
            "planogram summary",
        ]

        st.markdown("**Quick Queries**")
        q1, q2, q3, q4 = st.columns(4)
        if q1.button("Inventory Summary", use_container_width=True, key="qq_summary"):
            st.session_state.ai_query_text = quick_queries[0]
            st.session_state.ai_query_autorun = True
        if q2.button("Need Reorder", use_container_width=True, key="qq_reorder"):
            st.session_state.ai_query_text = quick_queries[1]
            st.session_state.ai_query_autorun = True
        if q3.button("Latest Detection", use_container_width=True, key="qq_detect"):
            st.session_state.ai_query_text = quick_queries[5]
            st.session_state.ai_query_autorun = True
        if q4.button("Planogram Summary", use_container_width=True, key="qq_planogram"):
            st.session_state.ai_query_text = quick_queries[6]
            st.session_state.ai_query_autorun = True

        st.caption(
            "More examples: "
            f"'{quick_queries[2]}', '{quick_queries[3]}', '{quick_queries[4]}'"
        )

        prompt_query = st.text_input(
            "Ask inventory assistant",
            placeholder="Example: Does Pantene shampoo need reorder?",
            key="ai_query_text",
        )
        run_clicked = st.button("Run Query", use_container_width=True, key="inventory_prompt_btn_main")
        should_auto_run = bool(st.session_state.pop("ai_query_autorun", False))

        if run_clicked or should_auto_run:
            query_norm = _normalize_prompt_text(prompt_query)

            if "latest detection" in query_norm:
                if latest_counts:
                    latest_df = build_count_table(latest_counts)
                    st.success(
                        f"Latest detection captured {int(sum(latest_counts.values()))} item(s) across {int(len(latest_counts))} product label(s)."
                    )
                    st.dataframe(latest_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No latest detection is available yet. Run Detection Studio first.")

            elif "planogram summary" in query_norm:
                planogram_payload = load_planogram(Path(DEFAULT_PLANOGRAM_PATH))
                planogram_rows = estimate_quantities(latest_counts, planogram_payload) if latest_counts else []
                if not planogram_rows:
                    st.warning("No planogram summary available yet. Add planogram items and run a detection.")
                else:
                    plan_df = pd.DataFrame(planogram_rows)
                    st.success(
                        f"Planogram summary: Estimated units={int(plan_df['Estimated Units'].sum())}, Alerts={int((plan_df['Needs Reorder'] == 'Yes').sum())}, Avg compliance={plan_df['Facing Compliance %'].mean():.1f}%."
                    )
                    st.dataframe(plan_df, use_container_width=True, hide_index=True)

            else:
                prompt_result = answer_inventory_prompt(prompt_query, inventory_df)
                kind = prompt_result.get("kind", "info")
                message = str(prompt_result.get("message", ""))
                if kind == "success":
                    st.success(message)
                elif kind == "warning":
                    st.warning(message)
                elif kind == "table":
                    st.info(message)
                    st.dataframe(prompt_result["table"], use_container_width=True, hide_index=True)
                else:
                    st.info(message)

        if not inventory_df.empty:
            st.markdown("**Current Inventory Snapshot**")
            st.dataframe(inventory_df, use_container_width=True, hide_index=True)
        else:
            st.info("Inventory is empty. Add items from Inventory Dashboard.")
        return

    if screen == "Planogram Dashboard":
        st.markdown('<div class="section-kicker">Shelf Layout Analytics</div>', unsafe_allow_html=True)
        st.subheader("Planogram Dashboard")
        st.caption("Configure planogram, review facing compliance, and apply estimated stock.")

        planogram_path = Path(DEFAULT_PLANOGRAM_PATH)
        planogram_payload = load_planogram(planogram_path)
        planogram_df = build_planogram_table(planogram_payload)
        latest_planogram_rows = estimate_quantities(latest_counts, planogram_payload) if latest_counts else []

        p1, p2, p3 = st.columns(3)
        p1.metric("Planogram Items", int(len(planogram_df)))
        p2.metric(
            "Estimated Units",
            int(pd.DataFrame(latest_planogram_rows)["Estimated Units"].sum())
            if latest_planogram_rows
            else 0,
        )
        p3.metric(
            "Reorder Alerts",
            int((pd.DataFrame(latest_planogram_rows)["Needs Reorder"] == "Yes").sum())
            if latest_planogram_rows
            else 0,
        )

        if latest_counts and latest_planogram_rows:
            if st.button("Apply Planogram Estimate to Inventory", use_container_width=True):
                inventory_path = Path(DEFAULT_INVENTORY_PATH)
                inventory_payload = load_inventory(inventory_path)
                sync_inventory_from_planogram_estimates(inventory_payload, latest_planogram_rows)
                save_inventory(inventory_path, inventory_payload)
                st.success("Inventory reconciled to planogram-estimated quantities.")

        pg_left, pg_right = st.columns([2, 1])
        with pg_left:
            if planogram_df.empty:
                st.info("No planogram items configured yet.")
            else:
                st.dataframe(planogram_df, use_container_width=True, hide_index=True)

            st.markdown("**Planogram Diagram (Latest Detection)**")
            if latest_counts:
                render_planogram_diagram(planogram_payload, latest_counts)
            else:
                st.caption("Run Detection Studio to overlay current detected facings on the planogram diagram.")
                render_planogram_diagram(planogram_payload, {})

            if latest_planogram_rows:
                st.markdown("**Planogram Quantity Estimate (Latest Detection)**")
                st.dataframe(pd.DataFrame(latest_planogram_rows), use_container_width=True, hide_index=True)

            st.download_button(
                "Download Planogram JSON",
                data=json.dumps(planogram_payload, indent=2).encode("utf-8"),
                file_name="planogram_config.json",
                mime="application/json",
                use_container_width=True,
            )

        with pg_right:
            st.markdown("**Manage Planogram Item**")
            planogram_options = [
                f"{row['Category']} | {row['Product']}"
                for _, row in planogram_df.iterrows()
            ] if not planogram_df.empty else []
            selected_planogram_option = st.selectbox(
                "Planogram item",
                options=["Custom"] + planogram_options,
                key="planogram_base_item",
            )

            plan_category = ""
            plan_product = ""
            plan_expected_facings = 1
            plan_full_stock_units = 100
            plan_reorder_level = DEFAULT_REORDER_LEVEL

            if selected_planogram_option != "Custom" and "|" in selected_planogram_option:
                plan_category, plan_product = [part.strip() for part in selected_planogram_option.split("|", maxsplit=1)]
                match = planogram_df[
                    (planogram_df["Category"] == plan_category)
                    & (planogram_df["Product"] == plan_product)
                ] if not planogram_df.empty else pd.DataFrame()
                if not match.empty:
                    plan_expected_facings = int(match.iloc[0]["Expected Facings"])
                    plan_full_stock_units = int(match.iloc[0]["Full Stock Units"])
                    plan_reorder_level = int(match.iloc[0]["Reorder Level"])

            plan_category_input = st.text_input("Planogram category", value=plan_category, key="planogram_category")
            plan_product_input = st.text_input("Planogram product", value=plan_product, key="planogram_product")
            plan_expected_input = st.number_input(
                "Expected facings",
                min_value=1,
                max_value=1000,
                value=int(plan_expected_facings),
                step=1,
                key="planogram_expected_facings",
            )
            plan_full_input = st.number_input(
                "Full stock units",
                min_value=0,
                max_value=100000,
                value=int(plan_full_stock_units),
                step=1,
                key="planogram_full_stock_units",
            )
            plan_reorder_input = st.number_input(
                "Planogram reorder level",
                min_value=0,
                max_value=100000,
                value=int(plan_reorder_level),
                step=1,
                key="planogram_reorder_level",
            )

            pp1, pp2 = st.columns(2)
            if pp1.button("Save Planogram", use_container_width=True):
                category_value = plan_category_input.strip()
                product_value = plan_product_input.strip()
                if not category_value or not product_value:
                    st.error("Planogram category and product are required.")
                else:
                    set_planogram_item(
                        payload=planogram_payload,
                        category=category_value,
                        product=product_value,
                        expected_facings=int(plan_expected_input),
                        full_stock_units=int(plan_full_input),
                        reorder_level=int(plan_reorder_input),
                    )
                    save_planogram(planogram_path, planogram_payload)
                    st.success("Planogram item saved.")
                    st.rerun()

            if pp2.button("Delete Planogram", use_container_width=True):
                category_value = plan_category_input.strip()
                product_value = plan_product_input.strip()
                if not category_value or not product_value:
                    st.error("Planogram category and product are required.")
                else:
                    removed = remove_planogram_item(
                        payload=planogram_payload,
                        category=category_value,
                        product=product_value,
                    )
                    if removed:
                        save_planogram(planogram_path, planogram_payload)
                        st.success("Planogram item deleted.")
                        st.rerun()
                    else:
                        st.warning("Planogram item not found.")
        return

    if screen == "Inventory Dashboard":
        st.markdown('<div class="section-kicker">Operations Panel</div>', unsafe_allow_html=True)
        st.subheader("Inventory Dashboard")
        st.caption("Operational stock management view for shelf items.")

        inventory_path = Path(DEFAULT_INVENTORY_PATH)
        inventory_payload = load_inventory(inventory_path)
        inventory_df = build_inventory_table(inventory_payload)

        total_skus = len(inventory_df)
        total_stock = int(inventory_df["Stock"].sum()) if not inventory_df.empty else 0
        reorder_count = int((inventory_df["Needs Reorder"] == "Yes").sum()) if not inventory_df.empty else 0

        k1, k2, k3 = st.columns(3)
        k1.metric("Tracked SKUs", total_skus)
        k2.metric("On-Hand Units", total_stock)
        k3.metric("Low Stock Alerts", reorder_count)

        inventory_report_payload = build_inventory_report_payload(
            inventory_df=inventory_df,
            latest_counts=latest_counts,
            latest_planogram_rows=[],
        )
        report_json_bytes = json.dumps(inventory_report_payload, indent=2).encode("utf-8")
        report_csv_bytes = (
            inventory_df.to_csv(index=False).encode("utf-8")
            if not inventory_df.empty
            else b"Category,Product,Stock,Reorder Level,Reorder Qty,Needs Reorder,Updated At\n"
        )

        r1, r2 = st.columns(2)
        r1.download_button(
            "Download Inventory Report (JSON)",
            data=report_json_bytes,
            file_name="inventory_report.json",
            mime="application/json",
            use_container_width=True,
        )
        r2.download_button(
            "Download Inventory Snapshot (CSV)",
            data=report_csv_bytes,
            file_name="inventory_snapshot.csv",
            mime="text/csv",
            use_container_width=True,
        )

        render_inventory_dashboards(inventory_df=inventory_df, latest_planogram_rows=[])

        if latest_counts:
            ia1, ia2, ia3 = st.columns(3)
            if ia1.button("Deduct Latest Detection", use_container_width=True):
                adjust_stock(inventory_payload, latest_counts, direction="deduct")
                save_inventory(inventory_path, inventory_payload)
                st.success("Latest detection deducted from stock.")
                st.rerun()
            if ia2.button("Add Latest Detection", use_container_width=True):
                adjust_stock(inventory_payload, latest_counts, direction="add")
                save_inventory(inventory_path, inventory_payload)
                st.success("Latest detection added to stock.")
                st.rerun()
            if ia3.button("Set Stock = Latest Detection", use_container_width=True):
                reconcile_stock(inventory_payload, latest_counts)
                apply_default_reorder_levels_for_detected_items(inventory_payload, latest_counts)
                save_inventory(inventory_path, inventory_payload)
                st.success("Stock reconciled to latest detected counts.")
                st.rerun()
        else:
            st.info("Run Detection Studio once to enable one-click inventory adjustments.")

        inv_left, inv_right = st.columns([2, 1])
        with inv_left:
            if inventory_df.empty:
                st.info("No inventory items yet. Add products from the panel on the right.")
            else:
                st.dataframe(inventory_df, use_container_width=True, hide_index=True)
                reorder_df = inventory_df[inventory_df["Needs Reorder"] == "Yes"]
                if not reorder_df.empty:
                    st.warning(f"{len(reorder_df)} item(s) are below reorder threshold.")

                inventory_csv = inventory_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Inventory CSV",
                    data=inventory_csv,
                    file_name="inventory_status.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with inv_right:
            st.markdown("**Manage Product Stock**")
            st.caption("Create, update, or delete inventory rows.")
            inventory_options = [
                f"{row['Category']} | {row['Product']}"
                for _, row in inventory_df.iterrows()
            ] if not inventory_df.empty else []
            detected_keys = sorted(latest_counts.keys())
            merged_options = sorted(set(inventory_options + detected_keys))
            detected_options = ["Custom"] + merged_options
            selected_option = st.selectbox("Base item", options=detected_options, key="inventory_base_item")

            detected_category = ""
            detected_product = ""
            selected_stock = 0
            selected_reorder = 0
            if selected_option != "Custom" and "|" in selected_option:
                detected_category, detected_product = [part.strip() for part in selected_option.split("|", maxsplit=1)]
                matching = inventory_df[
                    (inventory_df["Category"] == detected_category)
                    & (inventory_df["Product"] == detected_product)
                ] if not inventory_df.empty else pd.DataFrame()
                if not matching.empty:
                    selected_stock = int(matching.iloc[0]["Stock"])
                    selected_reorder = int(matching.iloc[0]["Reorder Level"])

            inv_category = st.text_input("Category", value=detected_category, key="inventory_category")
            inv_product = st.text_input("Product", value=detected_product, key="inventory_product")
            inv_stock = st.number_input(
                "Current stock",
                min_value=-10000,
                max_value=100000,
                value=selected_stock,
                step=1,
                key="inventory_stock",
            )
            inv_reorder = st.number_input(
                "Reorder level",
                min_value=0,
                max_value=100000,
                value=selected_reorder,
                step=1,
                key="inventory_reorder",
            )

            i1, i2 = st.columns(2)
            if i1.button("Save Item", use_container_width=True):
                category_value = inv_category.strip()
                product_value = inv_product.strip()
                if not category_value or not product_value:
                    st.error("Category and Product are required.")
                else:
                    set_item(
                        payload=inventory_payload,
                        category=category_value,
                        product=product_value,
                        stock=int(inv_stock),
                        reorder_level=int(inv_reorder),
                    )
                    save_inventory(inventory_path, inventory_payload)
                    st.success("Inventory item saved.")
                    st.rerun()

            if i2.button("Delete Item", use_container_width=True):
                category_value = inv_category.strip()
                product_value = inv_product.strip()
                if not category_value or not product_value:
                    st.error("Category and Product are required.")
                else:
                    removed = remove_item(
                        payload=inventory_payload,
                        category=category_value,
                        product=product_value,
                    )
                    if removed:
                        save_inventory(inventory_path, inventory_payload)
                        st.success("Inventory item deleted.")
                        st.rerun()
                    else:
                        st.warning("Inventory item not found.")

        return

    st.markdown('<div class="section-kicker">Computer Vision Workflow</div>', unsafe_allow_html=True)
    st.subheader("Detection Studio")
    st.caption("Upload or capture shelf images to classify products and review counts.")

    input_mode = st.radio("Image source", ["Upload file", "Camera"], horizontal=True)

    uploaded_file = None
    camera_image = None
    if input_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload shelf image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )
    else:
        camera_image = st.camera_input("Capture shelf image")

    has_input = bool(uploaded_file or camera_image)
    if has_input:
        image_name = uploaded_file.name if uploaded_file else "camera_capture.jpg"
        image_bytes = uploaded_file.read() if uploaded_file else camera_image.getvalue()
        image_suffix = Path(image_name).suffix if Path(image_name).suffix else ".jpg"
    else:
        st.info("Select an image to start detection.")

    if has_input and st.button("Run Detection", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading models and analyzing image..."):
                from src.corrections import load_corrections

                corrections_path = Path(DEFAULT_CORRECTIONS_PATH)
                corrections_payload = load_corrections(corrections_path)
                detector, classifiers = load_pipeline(
                    detector_weights=detector_weights,
                    clip_model=clip_model,
                    det_conf=det_conf,
                    det_iou=det_iou,
                )
                input_path = save_uploaded_image(data=image_bytes, suffix=image_suffix)
                predictions, counts, annotated_path, selected_profile, profile_scores = run_analysis(
                    detector=detector,
                    classifiers=classifiers,
                    image_path=input_path,
                    corrections_payload=corrections_payload,
                )
        except Exception as exc:
            st.error(
                "Detection dependencies are unavailable in this environment. "
                f"Details: {exc}"
            )
            return

        st.session_state.latest_result = {
            "image_name": image_name,
            "image_bytes": image_bytes,
            "input_path": str(input_path),
            "predictions": predictions,
            "counts": counts,
            "annotated_path": str(annotated_path),
            "selected_profile": selected_profile,
            "profile_scores": profile_scores,
        }

        if auto_sync_inventory and counts:
            inventory_path = Path(DEFAULT_INVENTORY_PATH)
            inventory_payload = load_inventory(inventory_path)
            if use_planogram_sync:
                planogram_payload = load_planogram(Path(DEFAULT_PLANOGRAM_PATH))
                estimated_rows = estimate_quantities(counts, planogram_payload)
                if estimated_rows:
                    sync_inventory_from_planogram_estimates(inventory_payload, estimated_rows)
                else:
                    reconcile_stock(inventory_payload, counts)
                    apply_default_reorder_levels_for_detected_items(inventory_payload, counts)
            else:
                reconcile_stock(inventory_payload, counts)
                apply_default_reorder_levels_for_detected_items(inventory_payload, counts)
            save_inventory(inventory_path, inventory_payload)
            st.success("Inventory auto-synced from latest detection.")

    latest_result = st.session_state.latest_result
    if not latest_result:
        return

    st.info(
        "Auto-selected profile: "
        f"{latest_result['selected_profile']} "
        f"(beverage={latest_result['profile_scores']['beverage']:.3f}, "
        f"shampoo={latest_result['profile_scores']['shampoo']:.3f})"
    )

    from src.corrections import (
        apply_corrections,
        list_profile_corrections,
        load_corrections,
        save_corrections,
        set_profile_correction,
    )
    from src.config import ProductLabel
    from src.pipeline import summarize_counts
    from src.visualize import draw_predictions

    planogram_payload = load_planogram(Path(DEFAULT_PLANOGRAM_PATH))
    planogram_rows = estimate_quantities(latest_result["counts"], planogram_payload)
    counts_df = build_count_table(latest_result["counts"]) if latest_result["counts"] else pd.DataFrame()
    detection_report_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image": latest_result["image_name"],
        "selected_profile": latest_result["selected_profile"],
        "profile_scores": latest_result["profile_scores"],
        "counts": latest_result["counts"],
        "planogram_estimate": planogram_rows,
        "annotated_image": latest_result["annotated_path"],
    }

    tab_overview, tab_analysis, tab_corrections, tab_exports = st.tabs(
        ["Overview", "Analysis", "Corrections", "Exports"]
    )

    with tab_overview:
        left, right = st.columns(2)
        with left:
            st.subheader("Input Image")
            st.image(latest_result["image_bytes"], use_container_width=True)
        with right:
            st.subheader("Detected Output")
            st.image(str(latest_result["annotated_path"]), use_container_width=True)

        total_items = int(sum(latest_result["counts"].values()))
        unique_products = int(len(latest_result["counts"]))
        total_boxes = int(len(latest_result["predictions"]))
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Products", total_items)
        m2.metric("Unique Labels", unique_products)
        m3.metric("Detected Regions", total_boxes)

    with tab_analysis:
        st.subheader("Counts and Planogram Insights")
        if not latest_result["counts"]:
            st.warning("No products detected in this image.")
        else:
            st.dataframe(counts_df, use_container_width=True, hide_index=True)
            a1, a2 = st.columns(2)
            with a1:
                st.markdown("**Counts by Product**")
                chart_df = counts_df.copy()
                chart_df["Label"] = chart_df["Category"] + " | " + chart_df["Product"]
                st.bar_chart(chart_df.set_index("Label")["Count"], use_container_width=True)
            with a2:
                st.markdown("**Counts by Category**")
                cat_df = counts_df.groupby("Category", as_index=False)["Count"].sum().sort_values("Count", ascending=False)
                st.bar_chart(cat_df.set_index("Category")["Count"], use_container_width=True)

            if planogram_rows:
                st.subheader("Planogram Quantity Estimate")
                st.caption(
                    "Estimated units are calculated from detected facings vs expected facings and full-stock units."
                )
                st.markdown("**Planogram Diagram**")
                render_planogram_diagram(planogram_payload, latest_result["counts"])
                st.dataframe(pd.DataFrame(planogram_rows), use_container_width=True, hide_index=True)

    with tab_corrections:
        st.subheader("Correction Studio")
        st.caption("Fix wrong labels once; saved mappings are applied automatically on future runs.")

        corrections_path = Path(DEFAULT_CORRECTIONS_PATH)
        current_corrections = load_corrections(corrections_path)
        saved_rows = list_profile_corrections(
            payload=current_corrections,
            profile=latest_result["selected_profile"],
        )
        with st.expander("Saved Corrections (Current Profile)", expanded=False):
            if saved_rows:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Wrong Label": row["wrong_name"],
                                "Correct Category": row["category"],
                                "Correct Product": row["name"],
                            }
                            for row in saved_rows
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No saved corrections for this profile yet.")

        predictions = latest_result["predictions"]
        wrong_options = sorted(
            [
                {
                    "name": pred.classification.product.name,
                    "category": pred.classification.product.category,
                }
                for pred in predictions
            ],
            key=lambda x: x["name"].lower(),
        )

        if wrong_options:
            unique_options = {
                (item["name"], item["category"]): item for item in wrong_options
            }
            option_values = list(unique_options.values())
            chosen_option = st.selectbox(
                "Wrong label",
                options=option_values,
                format_func=lambda x: f"{x['name']} ({x['category']})",
                key="wrong_label_select",
            )
            chosen_name = chosen_option["name"]
            default_category = chosen_option["category"]

            sample_prediction = next(
                (
                    pred
                    for pred in predictions
                    if pred.classification.product.name == chosen_name
                ),
                None,
            )
            if sample_prediction is not None:
                source_image = Image.open(Path(latest_result["input_path"])).convert("RGB")
                det = sample_prediction.detection
                sample_crop = source_image.crop((det.x1, det.y1, det.x2, det.y2))
                st.image(sample_crop, caption=f"Sample object for '{chosen_name}'", width=220)

            corrected_name = st.text_input("Correct name", value=chosen_name, key="correct_name")
            corrected_category = st.text_input(
                "Correct category",
                value=default_category,
                key="correct_category",
            )

            if st.button("Save correction for future", use_container_width=True, key="save_correction"):
                corrections_payload = load_corrections(corrections_path)
                resolved_name = corrected_name.strip() or chosen_name
                resolved_category = normalize_category(
                    corrected_category.strip() or default_category
                )
                set_profile_correction(
                    payload=corrections_payload,
                    profile=latest_result["selected_profile"],
                    wrong_name=chosen_name,
                    corrected_label=ProductLabel(
                        category=resolved_category,
                        name=resolved_name,
                        prompt=resolved_name,
                    ),
                )
                save_corrections(corrections_path, corrections_payload)

                persisted_payload = load_corrections(corrections_path)
                refreshed_predictions = apply_corrections(
                    predictions=latest_result["predictions"],
                    profile=latest_result["selected_profile"],
                    corrections_payload=persisted_payload,
                )
                refreshed_counts = {
                    key: int(value)
                    for key, value in summarize_counts(refreshed_predictions).items()
                }
                refreshed_annotated = Path("outputs") / f"annotated_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                draw_predictions(
                    image_path=Path(latest_result["input_path"]),
                    predictions=refreshed_predictions,
                    output_path=refreshed_annotated,
                )
                st.session_state.latest_result = {
                    "image_name": latest_result["image_name"],
                    "image_bytes": latest_result["image_bytes"],
                    "input_path": latest_result["input_path"],
                    "predictions": refreshed_predictions,
                    "counts": refreshed_counts,
                    "annotated_path": str(refreshed_annotated),
                    "selected_profile": latest_result["selected_profile"],
                    "profile_scores": latest_result["profile_scores"],
                }
                st.success(
                    f"Saved correction and reapplied now: {chosen_name} -> {resolved_name}"
                )
                st.rerun()

    with tab_exports:
        st.subheader("Exports and Report Files")
        if latest_result["counts"]:
            csv_bytes = counts_df.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(
                {"image": latest_result["image_name"], "counts": latest_result["counts"]},
                indent=2,
            ).encode("utf-8")
            report_bytes = json.dumps(detection_report_payload, indent=2).encode("utf-8")

            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "Download Counts CSV",
                data=csv_bytes,
                file_name="counts_result.csv",
                mime="text/csv",
                use_container_width=True,
            )
            c2.download_button(
                "Download Counts JSON",
                data=json_bytes,
                file_name="counts_result.json",
                mime="application/json",
                use_container_width=True,
            )
            c3.download_button(
                "Download Detection Report",
                data=report_bytes,
                file_name="detection_report.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Run detection to enable export downloads.")

        st.success(f"Annotated image saved: {latest_result['annotated_path']}")


if __name__ == "__main__":
    main()
