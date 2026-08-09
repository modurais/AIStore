# Deploy To Hugging Face Spaces (Docker)

## 1) Create a new Space
- Go to Hugging Face -> New Space.
- Choose:
  - SDK: Docker
  - Visibility: Public or Private
  - Hardware: CPU Basic (start here)

## 2) Add Space metadata README
Hugging Face Spaces uses README frontmatter for metadata. Replace the Space README with:

---
title: Retail AI Shelf Intelligence
emoji: "🛒"
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
---

Shelf analytics app using YOLO + CLIP + Streamlit.

## 3) Push this project to the Space repo
Use git remote for the Space and push:

1. git init
2. git add .
3. git commit -m "Initial Docker deployment"
4. git remote add origin https://huggingface.co/spaces/<username>/<space-name>
5. git push -u origin main

## 4) Model/data considerations
- Ensure required model weights exist in repo paths used by config:
  - Weights/best.pt
- Keep large files minimal; large artifacts increase build time.
- For persistent inventory/planogram data, consider external storage (the container filesystem can reset).

## 5) Runtime behavior
- Container starts Streamlit on PORT (defaults to 7860).
- Main app entrypoint: realtime_app.py

## 6) Troubleshooting
- If build fails on dependencies, check requirements.docker.txt.
- If model loading fails, verify weight file paths and presence in repository.
- If memory is insufficient, reduce model size or upgrade Space hardware.
