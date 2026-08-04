import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

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
from src.inventory import (
    adjust_stock,
    list_items,
    load_inventory,
    remove_item,
    save_inventory,
    set_item,
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
    st.title("Shelf Product Console")
    st.caption("No login required. Use Detection Studio or Inventory Center directly.")

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None

    with st.sidebar:
        st.header("Workspace")
        screen = st.radio(
            "Choose screen",
            ["Detection Studio", "Inventory Center"],
            index=0,
        )
        st.divider()
        st.header("Model Settings")
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

    latest_result = st.session_state.latest_result
    latest_counts = latest_result["counts"] if latest_result else {}

    if screen == "Inventory Center":
        st.subheader("Inventory Center")
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
            st.caption("Quick Q&A: ask stock, reorder level, or reorder status.")
            prompt_query = st.text_input(
                "Inventory prompt",
                placeholder="Example: Does Pantene shampoo need reorder?",
                key="inventory_prompt_query",
            )
            if st.button("Analyze Prompt", use_container_width=True, key="inventory_prompt_btn"):
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

            st.divider()
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

    left, right = st.columns(2)
    with left:
        st.subheader("Input Image")
        st.image(latest_result["image_bytes"], use_container_width=True)

    with right:
        st.subheader("Detected Output")
        st.image(str(latest_result["annotated_path"]), use_container_width=True)

    total_items = sum(latest_result["counts"].values())
    unique_products = len(latest_result["counts"])

    m1, m2 = st.columns(2)
    m1.metric("Total Products", total_items)
    m2.metric("Unique Product Labels", unique_products)

    st.subheader("Counts")
    if not latest_result["counts"]:
        st.warning("No products detected in this image.")
    else:
        df = build_count_table(latest_result["counts"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(
            {"image": latest_result["image_name"], "counts": latest_result["counts"]},
            indent=2,
        ).encode("utf-8")

        c1, c2 = st.columns(2)
        c1.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="counts_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
        c2.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="counts_result.json",
            mime="application/json",
            use_container_width=True,
        )

        st.success(f"Annotated image saved: {latest_result['annotated_path']}")

        st.subheader("Teach Correction")
        st.caption("If a label is wrong, save a correction once. It is applied immediately.")

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
            # Unique options avoid ambiguous parsing and keep state stable across reruns.
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
                corrections_path = Path(DEFAULT_CORRECTIONS_PATH)
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

                # Re-load from disk to verify persistence, then refresh UI from saved mapping.
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


if __name__ == "__main__":
    main()
