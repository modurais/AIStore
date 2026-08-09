# GROUP 1 CAPSTONE PROJECT REPORT

## Shelf Product Detection, Classification, Counting, and Planogram-Aware Inventory Estimation using YOLO and CLIP

Submitted in partial fulfillment of the requirements for the award of the degree of

BACHELOR OF ENGINEERING / TECHNOLOGY

in

[Department Name]

by

- [Member 1 Name] ([Reg No])
- [Member 2 Name] ([Reg No])
- [Member 3 Name] ([Reg No])
- [Member 4 Name] ([Reg No])

Under the guidance of

[Guide Name and Designation]

[Institution Name]

[Month Year]

[[PAGE_BREAK]]

## CERTIFICATE

This is to certify that the project report titled "Shelf Product Detection, Classification, Counting, and Planogram-Aware Inventory Estimation using YOLO and CLIP" is a bonafide work carried out by Group 1 in partial fulfillment of the requirements for the award of the degree of Bachelor of Engineering / Technology in [Department Name], [Institution Name].

The work embodied in this report has not been submitted elsewhere for the award of any other degree or diploma.

Place: [City]

Date: [DD-MM-YYYY]

Guide Signature: ____________________

Head of Department Signature: ____________________

[[PAGE_BREAK]]

## DECLARATION

We declare that this capstone report titled "Shelf Product Detection, Classification, Counting, and Planogram-Aware Inventory Estimation using YOLO and CLIP" is our original work and has been completed under the guidance of [Guide Name].

We also declare that this work has not been submitted in full or in part for the award of any degree, diploma, or similar title at any institution.

Team Members:

- [Member 1 Name and Signature]
- [Member 2 Name and Signature]
- [Member 3 Name and Signature]
- [Member 4 Name and Signature]

Date: [DD-MM-YYYY]

[[PAGE_BREAK]]

## ACKNOWLEDGEMENT

We express our sincere gratitude to our guide, [Guide Name], for valuable guidance, encouragement, and technical support throughout this capstone project.

We thank the Head of the Department, faculty members, and our institution for providing the infrastructure and learning environment required to complete this work.

We also thank our peers and family members for their continuous support and motivation.

[[PAGE_BREAK]]

## ABSTRACT

Retail shelf auditing is commonly manual, time-consuming, and error-prone. This project presents an AI-enabled system that automatically detects products on retail shelves, classifies each detected region into product labels, counts visible facings, and estimates stock using planogram expectations.

The implemented solution combines a YOLO detector with a CLIP-based classifier. It supports command-line and Streamlit web interfaces, automatic label-profile selection (beverage vs shampoo), visual output with annotations, and CSV/JSON export for integration into inventory workflows. A planogram module maps detected facings to estimated units and reorder signals.

The project demonstrates a practical and extensible foundation for intelligent shelf analytics and inventory decision support in retail operations.

Keywords: Computer Vision, YOLO, CLIP, Shelf Analytics, Planogram, Inventory Estimation, Streamlit.

[[PAGE_BREAK]]

## TABLE OF CONTENTS

1. Introduction
2. Problem Statement
3. Objectives
4. System Architecture
5. Dataset and Models
6. Methodology
7. Implementation Details
8. Results and Discussion
9. Strengths and Limitations
10. Future Scope
11. Conclusion
12. References
13. Appendix

[[PAGE_BREAK]]

## 1. INTRODUCTION

Retail stores require frequent shelf monitoring to prevent stockouts and maintain planogram compliance. Manual auditing is labor-intensive and often inconsistent across operators and store shifts.

This project builds an automated computer vision pipeline for shelf product analysis. It identifies product regions, classifies products, computes count summaries, and supports inventory updates through a simple web interface.

## 2. PROBLEM STATEMENT

Given shelf images, automatically estimate product-level shelf presence and quantities with minimal human effort, and provide outputs that are directly usable for retail inventory actions.

Expected outputs:

- Product-wise counts.
- Annotated visual proof image(s).
- CSV and JSON summaries.
- Planogram-based estimated stock and reorder indicators.

## 3. OBJECTIVES

1. Detect shelf products robustly from images.
2. Classify each detected crop into meaningful product labels.
3. Aggregate counts by category and product.
4. Build an operator-friendly web application.
5. Support planogram-driven inventory estimation and reorder decisions.

## 4. SYSTEM ARCHITECTURE

High-level flow:

1. User uploads or captures shelf image.
2. YOLO model detects product bounding boxes.
3. Cropped regions are classified with CLIP prompts.
4. Predictions are grouped as Category | Product counts.
5. Counts and annotated output are generated.
6. Planogram module estimates stock and reorder quantities.
7. Inventory records can be auto-synced or manually reconciled.

Core project modules:

- Command-line inference pipeline.
- Streamlit interface with Detection Studio and Inventory Center.
- Detection engine (YOLO wrapper).
- Classification engine (CLIP wrapper).
- Profile selector for beverage/shampoo domains.
- Planogram and inventory utilities.

## 5. DATASET AND MODELS

### 5.1 Dataset Strategy

The training workflow is designed around SKU110K-style shelf annotations converted to YOLO format using a preparation script.

### 5.2 Detection Model

- Model family: Ultralytics YOLO.
- Default project weights: best.pt in Weights folder.
- Typical project thresholds:
  - Detection confidence: 0.10
  - NMS IoU: 0.30

### 5.3 Classification Model

- Model: CLIP (openai/clip-vit-base-patch32).
- Prompt engineering with domain profiles:
  - Beverage profile
  - Shampoo profile
- Non-product rejection prompts for tags/signage reduce false positives.

## 6. METHODOLOGY

### 6.1 Detection to Classification

For each image:

1. Run detector and collect product boxes.
2. Crop each detection.
3. Run CLIP text-image scoring for candidate prompts.
4. Select best label (or reject as non-product where applicable).
5. Aggregate final product counts.

### 6.2 Auto Profile Selection

The system compares average confidence scores from beverage and shampoo classifiers and chooses the profile with higher mean score for that image.

### 6.3 Planogram-Based Quantity Estimation

Each planogram item stores:

- expected facings
- full stock units
- reorder level

Per product computation:

- detected facings = d
- expected facings = e
- compliance ratio = min(1, d / e)
- estimated units = round(compliance ratio x full stock units)
- reorder quantity = max(0, reorder level - estimated units)
- needs reorder = Yes if estimated units < reorder level, else No

### 6.4 Export and Reporting

The system exports:

- Annotated image output.
- CSV count summaries.
- JSON structured summaries.
- Inventory status tables through the web app.

## 7. IMPLEMENTATION DETAILS

### 7.1 Interfaces

- Command-line interface supports single image and batch folder inference.
- Streamlit app supports upload/camera input, output visualization, count display, and inventory actions.

### 7.2 Inventory Center Features

- Add, deduct, or reconcile stock from latest detections.
- Manage product-wise reorder levels.
- Configure and edit planogram records.
- Apply stock update from planogram estimates.

### 7.3 Technology Stack

- Python 3.11
- Streamlit
- Pandas
- Pillow
- NumPy
- Ultralytics YOLO
- Transformers and Torch (for CLIP)

## 8. RESULTS AND DISCUSSION

Observed sample output from current project run indicates successful end-to-end operation.

Sample totals from cooldrinks image:

- Soft Drinks | Coca-Cola: 17
- Soft Drinks | Fanta: 6
- Soft Drinks | Sprite: 5
- Soft Drinks | 7UP: 2

Discussion:

- The pipeline effectively localizes and classifies multiple shelf products.
- Export outputs are suitable for further analytics.
- Planogram logic provides practical stock approximation from visual facings.

## 9. STRENGTHS AND LIMITATIONS

### 9.1 Strengths

1. Modular and extensible architecture.
2. Practical UI and CLI support.
3. Profile-aware classification strategy.
4. Inventory and planogram integration for operations.

### 9.2 Limitations

1. Accuracy may degrade with heavy occlusion or blur.
2. Similar packaging can cause misclassification.
3. Proportional planogram estimate does not model backroom stock.
4. Full benchmark metrics are pending comprehensive evaluation.

## 10. FUTURE SCOPE

1. Add object tracking for video-based real-time counting.
2. Fine-tune domain-specific brand classifiers.
3. Add unknown-class handling with confidence calibration.
4. Improve planogram logic with row and slot constraints.
5. Integrate dashboard analytics and automated alerts.

## 11. CONCLUSION

The project demonstrates a complete AI-driven shelf intelligence workflow from image input to inventory decision support. By combining YOLO detection, CLIP classification, and planogram-aware stock estimation, the system reduces manual audit effort and supports better retail shelf management.

## 12. REFERENCES

1. Ultralytics YOLO documentation.
2. CLIP research paper and model resources.
3. SKU110K dataset resources.
4. Streamlit documentation.
5. Hugging Face Transformers documentation.

## 13. APPENDIX

### 13.1 Example Commands

Single image:

python app.py --image cooldrinks.jpg --detector-weights Weights/best.pt --output-image outputs/annotated_result.jpg

Web app:

streamlit run realtime_app.py

### 13.2 Submission Checklist

- Report PDF generated.
- Code repository prepared.
- Output evidence (images, CSV, JSON) included.
- Team details and signatures updated in report.
