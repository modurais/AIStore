---
title: Retail AI Shelf Intelligence
emoji: "🛒"
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
---

# Shelf Product Detection and Counting (YOLO + CLIP + SKU110K)

This project detects products on a supermarket shelf, classifies each product into a category/name such as:
- `Soft Drinks | Coca-Cola`
- `Soft Drinks | Sprite`

Then it counts how many of each product is visible.

## What this solution does

1. **Train/Use YOLO detector** to find product bounding boxes on shelf images.
2. **Use CLIP classifier** on each detected crop to recognize product label/brand.
3. **Aggregate counts** by `Category | ProductName`.
4. **Save annotated output image** with bounding boxes and class names.

## Important note about "YOLO26"

There is no official widely-used public release named exactly `YOLO26` at the time of writing.
This code supports any YOLO weight file path. The default is now `Weights/best.pt` (from your better run). You can still override it with any custom path:

```bash
python app.py --image path/to/image.jpg --detector-weights Weights/best.pt
```

If you do not have custom weights yet, start from standard Ultralytics weights (`yolov8n.pt`) for initial testing/training.

## Project structure

- `app.py` : end-to-end inference pipeline on one image.
- `src/detector.py` : YOLO product detection.
- `src/classifier.py` : CLIP crop classification.
- `src/pipeline.py` : orchestration + counting.
- `src/visualize.py` : annotated output image.
- `src/config.py` : candidate product list and defaults.
- `scripts/prepare_sku110k_for_yolo.py` : convert SKU110K CSV annotations into YOLO labels.
- `scripts/train_yolo_on_sku110k.py` : train YOLO detector.

## Step-by-step setup and run

### Step 1: Create and activate environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Prepare SKU110K dataset

Expected raw dataset structure (example):

```text
data/SKU110K/
  images/
    train/
    val/
    test/
  annotations_train.csv
  annotations_val.csv
  annotations_test.csv
```

Convert CSV annotations to YOLO labels:

```bash
python scripts/prepare_sku110k_for_yolo.py --dataset-root data/SKU110K
```

This generates:
- `data/SKU110K/sku110k_yolo/labels/<split>/*.txt`
- `data/SKU110K/sku110k_yolo/data.yaml`

### Step 3: Train YOLO detector

```bash
python scripts/train_yolo_on_sku110k.py --data-yaml data/SKU110K/sku110k_yolo/data.yaml --epochs 50 --imgsz 960 --batch 8
```

After training, copy the best weights to:
- `Weights/best.pt`, or use direct path from `runs/train/.../best.pt`.

### Step 4: Configure CLIP labels

`src/config.py` now includes classification labels adapted from `Weights/yoloclip_july25.py` (for example: `Coca Cola can`, `Sprite can`, `soft drink`, `fruit juice`, `water bottle`, etc.).
The CLIP model chooses among these prompts.

### Step 5: Run inference on your shelf image

```bash
python app.py --image data/sample_shelf.jpg --detector-weights Weights/best.pt --output-image outputs/annotated_result.jpg
```

You will get:
- Console product counts.
- Annotated image in `outputs/annotated_result.jpg`.
- CSV summary in `outputs/counts_summary.csv`.
- JSON summary in `outputs/counts_summary.json`.

### Step 6: Run batch inference for a folder of images

```bash
python app.py --image-dir data/shelf_images --detector-weights Weights/best.pt --output-dir outputs/batch --counts-csv outputs/batch_counts.csv --counts-json outputs/batch_counts.json
```

You will get:
- One annotated image per input image in `outputs/batch`.
- `batch_counts.csv` with rows by image/category/product.
- `batch_counts.json` with per-image counts and aggregated totals.

## How each code step works

1. `app.py` loads YOLO and CLIP models.
2. It reads either one image (`--image`) or many images (`--image-dir`).
3. YOLO predicts product boxes from each shelf image.
4. Each box is cropped and passed to CLIP.
5. CLIP assigns the best matching product prompt from `CANDIDATE_PRODUCTS`.
6. Predictions are grouped and counted per image, then aggregated across all images.
7. Annotated image(s) are saved with product name + confidence.
8. Counts are exported to CSV and JSON files.

## Practical improvements for production

- Add an "Unknown" class threshold for low CLIP confidence.
- Add planogram constraints (shelf-row consistency) to reduce false labels.
- Fine-tune CLIP or a brand classifier on your own product photos.
- Track IDs across video frames for robust counting in live camera streams.
- Add evaluation scripts with mAP (detector) and Top-1/Top-5 accuracy (classifier).

## Real-time style UI app (upload image)

Use Streamlit app for an interactive workflow:

```bash
streamlit run realtime_app.py
```

What this app provides:
- Upload image or capture from camera.
- Run YOLO + CLIP detection and classification.
- Show input and annotated output side by side.
- Display total count and per-product count table.
- Download result as CSV and JSON.
