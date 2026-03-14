FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serving_api source
COPY app/ ./app/

# Copy ml_pipeline source (needed for inference.py + preprocess constants)
COPY ../ml_pipeline/src/ ./ml_pipeline/src/

# Copy model artifacts and processed data
# In production these should be mounted as volumes, not baked into the image
COPY ../models/ ./models/
COPY ../data/processed/ ./data/processed/

# Environment — configurable via docker run -e or docker-compose
ENV MODELS_DIR=/app/models
ENV PROCESSED_DIR=/app/data/processed
ENV PYTHONPATH=/app/app:/app/ml_pipeline/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]