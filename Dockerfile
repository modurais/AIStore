FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# Runtime libraries commonly required by OpenCV/Ultralytics/Streamlit image processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements.docker.txt requirements.docker.txt
RUN pip install --upgrade pip && pip install -r requirements.docker.txt

COPY . .

EXPOSE 7860

# Hugging Face Spaces sets PORT=7860 for Docker apps.
CMD ["sh", "-c", "streamlit run realtime_app.py --server.address=0.0.0.0 --server.port=${PORT:-7860}"]
