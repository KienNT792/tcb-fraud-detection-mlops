FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY serving_api/requirements.txt /tmp/serving_requirements.txt
RUN pip install --no-cache-dir -r /tmp/serving_requirements.txt

COPY serving_api/app ./app
COPY ml_pipeline/src ./ml_pipeline/src
COPY scripts ./scripts
COPY models ./models
COPY docs ./docs

ENV MODELS_DIR=/app/models
ENV PROCESSED_DIR=/app/data/processed
ENV FEATURE_READY_DIR=/app/data/feature-ready
ENV PYTHONPATH=/app:/app/app:/app/ml_pipeline/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["bash", "scripts/start_api.sh"]
