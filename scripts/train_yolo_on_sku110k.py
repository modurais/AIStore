import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector on SKU110K in YOLO format")
    parser.add_argument("--data-yaml", required=True, help="Path to data.yaml")
    parser.add_argument("--base-weights", default="yolov8n.pt", help="Starting weights")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="sku110k_yolo26")
    args = parser.parse_args()

    model = YOLO(args.base_weights)
    model.train(
        data=str(Path(args.data_yaml)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    print("Training finished.")
    print("Best model is typically saved at: runs/train/<name>/weights/best.pt")


if __name__ == "__main__":
    main()
