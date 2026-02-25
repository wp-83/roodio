# ── Roodio ML API — Dockerfile ──────────────────────────────
# Python 3.11 slim (compatible with all dependencies)
FROM python:3.11-slim

WORKDIR /app

# System dependencies for librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (models/ excluded via .dockerignore)
COPY app.py utils.py ./

# Temp folder for uploaded files
RUN mkdir -p temp_uploads

EXPOSE 7860

CMD ["python", "app.py"]
