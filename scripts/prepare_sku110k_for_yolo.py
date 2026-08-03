import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


ANNOTATION_COLS = ["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]


def to_yolo_line(x1: float, y1: float, x2: float, y2: float, width: float, height: float) -> str:
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def convert_annotations(csv_path: Path, labels_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    missing = [col for col in ANNOTATION_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_name, group in tqdm(df.groupby("image_name"), desc=f"Converting {csv_path.name}"):
        lines = []
        width = float(group.iloc[0]["image_width"])
        height = float(group.iloc[0]["image_height"])

        for _, row in group.iterrows():
            lines.append(
                to_yolo_line(
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                    width=width,
                    height=height,
                )
            )

        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")


def create_data_yaml(output_root: Path) -> None:
    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "product"},
        "nc": 1,
    }
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SKU110K annotations into YOLO format")
    parser.add_argument("--dataset-root", required=True, help="Path to SKU110K root folder")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    output_root = root / "sku110k_yolo"

    for split in ["train", "val", "test"]:
        csv_file = root / f"annotations_{split}.csv"
        labels_dir = output_root / "labels" / split
        if not csv_file.exists():
            print(f"Skipping {split}: {csv_file} does not exist")
            continue
        convert_annotations(csv_file, labels_dir)

    create_data_yaml(output_root)
    print(f"YOLO-ready dataset generated at: {output_root}")


if __name__ == "__main__":
    main()
