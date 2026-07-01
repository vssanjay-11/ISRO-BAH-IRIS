# ────────────────────────────────────────────────────────────────────────────
# IRIS-AI Dockerfile
# Intelligent Remote-sensing Infrared Interpretation Suite
# ISRO Bharatiya Antariksh Hackathon 2026 – PS-10
# ────────────────────────────────────────────────────────────────────────────

# Base image: CUDA 11.8 + cuDNN + Python 3.10
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="IRIS-AI Team"
LABEL description="ISRO BAH 2026 PS-10 – Infrared Image Colorization & Analysis"
LABEL version="1.0.0"

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy only requirements first (layer cache) ───────────────────────────────
COPY requirements.txt .

# ── Install Python dependencies ──────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy full project ────────────────────────────────────────────────────────
COPY . .

# ── Create runtime directories ───────────────────────────────────────────────
RUN mkdir -p /app/weights \
             /app/outputs \
             /app/reports \
             /app/assets \
             /app/configs

# ── Expose Streamlit port ────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# ── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
