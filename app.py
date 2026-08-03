import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.classifier import ClipProductClassifier
from src.config import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_CORRECTIONS_PATH,
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_DETECTION_IOU,
    DEFAULT_DETECTOR_WEIGHTS,
    NON_PRODUCT_PROMPTS,
    NON_PRODUCT_REJECT_MARGIN,
    get_candidate_products,
)
from src.corrections import apply_corrections, load_corrections
from src.detector import YoloShelfDetector
from src.pipeline import ShelfAnalysisPipeline, summarize_counts
from src.profile_selector import choose_best_profile
from src.visualize import draw_predictions


def find_images(image_dir: Path) -> List[Path]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    images: List[Path] = []
    for pattern in patterns:
        images.extend(image_dir.glob(pattern))
    return sorted(images)


def counts_to_serializable(counts) -> Dict[str, int]:
    return {key: int(value) for key, value in counts.items()}


def write_counts_csv(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "category", "product", "count"])
        writer.writeheader()
        writer.writerows(rows)


def write_counts_json(payload: Dict, output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shelf product detection + CLIP classification + counting"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to input shelf image")
    input_group.add_argument("--image-dir", help="Path to folder of shelf images")
    parser.add_argument(
        "--detector-weights",
        default=str(DEFAULT_DETECTOR_WEIGHTS),
        help="Path to YOLO weights (default uses Weights/best.pt)",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help="CLIP model name from Hugging Face",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=DEFAULT_DETECTION_CONFIDENCE,
        help="YOLO detection confidence threshold",
    )
    parser.add_argument(
        "--det-iou",
        type=float,
        default=DEFAULT_DETECTION_IOU,
        help="YOLO NMS IoU threshold",
    )
    parser.add_argument(
        "--output-image",
        default="outputs/annotated_result.jpg",
        help="Path to save annotated image for single-image mode",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder to save annotated images for batch mode",
    )
    parser.add_argument(
        "--counts-csv",
        default="outputs/counts_summary.csv",
        help="Path to save counts summary as CSV",
    )
    parser.add_argument(
        "--counts-json",
        default="outputs/counts_summary.json",
        help="Path to save counts summary as JSON",
    )
    parser.add_argument(
        "--label-profile",
        choices=["auto", "beverage", "shampoo"],
        default="auto",
        help="CLIP label profile to use (auto selects per image)",
    )
    parser.add_argument(
        "--corrections-file",
        default=str(DEFAULT_CORRECTIONS_PATH),
        help="Path to saved wrong->right label corrections",
    )
    parser.add_argument(
        "--disable-corrections",
        action="store_true",
        help="Disable applying saved label corrections",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    detector_weights = Path(args.detector_weights)
    corrections_payload = {}
    if not args.disable_corrections:
        corrections_payload = load_corrections(Path(args.corrections_file))

    print("Step 1/6: Loading YOLO detector...")
    detector = YoloShelfDetector(
        model_path=detector_weights,
        confidence=args.det_conf,
        iou=args.det_iou,
    )

    print("Step 2/6: Loading CLIP classifier(s)...")
    if args.label_profile == "auto":
        classifiers = {
            "beverage": ClipProductClassifier(
                model_name=args.clip_model,
                candidate_labels=get_candidate_products("beverage"),
                negative_prompts=NON_PRODUCT_PROMPTS,
                reject_margin=NON_PRODUCT_REJECT_MARGIN,
            ),
            "shampoo": ClipProductClassifier(
                model_name=args.clip_model,
                candidate_labels=get_candidate_products("shampoo"),
                negative_prompts=NON_PRODUCT_PROMPTS,
                reject_margin=NON_PRODUCT_REJECT_MARGIN,
            ),
        }
        print("Using label profile: auto (beverage + shampoo)")
    else:
        selected_labels = get_candidate_products(args.label_profile)
        classifiers = {
            args.label_profile: ClipProductClassifier(
                model_name=args.clip_model,
                candidate_labels=selected_labels,
                negative_prompts=NON_PRODUCT_PROMPTS,
                reject_margin=NON_PRODUCT_REJECT_MARGIN,
            )
        }
        print(f"Using label profile: {args.label_profile} ({len(selected_labels)} labels)")
    csv_rows: List[Dict[str, str]] = []
    json_per_image: List[Dict] = []
    aggregate_counts: Dict[str, int] = {}

    if args.image:
        image_paths = [Path(args.image)]
        output_paths = [Path(args.output_image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = find_images(image_dir)
        if not image_paths:
            raise FileNotFoundError(f"No images found in folder: {image_dir}")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [output_dir / f"{img.stem}_annotated.jpg" for img in image_paths]

    print("Step 3/6: Running product detection and crop classification...")
    for image_path, output_path in zip(image_paths, output_paths):
        detections = detector.detect(image_path)

        selected_profile = next(iter(classifiers))
        if args.label_profile == "auto":
            selected_profile, profile_scores = choose_best_profile(
                image_path=image_path,
                detections=detections,
                classifiers_by_profile=classifiers,
            )
            print(
                "Auto profile selection "
                f"for {image_path.name}: {selected_profile} "
                f"(beverage={profile_scores['beverage']:.3f}, shampoo={profile_scores['shampoo']:.3f})"
            )

        pipeline = ShelfAnalysisPipeline(detector=detector, classifier=classifiers[selected_profile])
        predictions = pipeline.analyze_with_detections(
            image_path=image_path,
            detections=detections,
        )
        if not args.disable_corrections:
            predictions = apply_corrections(
                predictions=predictions,
                profile=selected_profile,
                corrections_payload=corrections_payload,
            )

        print(f"Processing: {image_path.name}")
        counts = summarize_counts(predictions)
        serial_counts = counts_to_serializable(counts)

        for label, count in serial_counts.items():
            aggregate_counts[label] = aggregate_counts.get(label, 0) + count

            category, product = [part.strip() for part in label.split("|", maxsplit=1)]
            csv_rows.append(
                {
                    "image": image_path.name,
                    "category": category,
                    "product": product,
                    "count": str(count),
                }
            )

        json_per_image.append(
            {
                "image": image_path.name,
                "counts": serial_counts,
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        draw_predictions(
            image_path=image_path,
            predictions=predictions,
            output_path=output_path,
        )

    print("Step 4/6: Writing CSV and JSON count summaries...")
    csv_path = Path(args.counts_csv)
    json_path = Path(args.counts_json)
    write_counts_csv(rows=csv_rows, output_csv=csv_path)
    write_counts_json(
        payload={
            "images": json_per_image,
            "totals": aggregate_counts,
        },
        output_json=json_path,
    )

    print("Step 5/6: Printing aggregated counts...")
    if not aggregate_counts:
        print("No products detected.")
    else:
        for key, count in aggregate_counts.items():
            print(f"- {key}: {count}")

    print("Step 6/6: Done.")
    if args.image:
        print(f"Annotated output saved at: {output_paths[0]}")
    else:
        print(f"Annotated outputs saved in: {Path(args.output_dir)}")
    print(f"CSV summary saved at: {csv_path}")
    print(f"JSON summary saved at: {json_path}")


if __name__ == "__main__":
    main()
