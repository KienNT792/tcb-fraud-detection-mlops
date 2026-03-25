FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY serving_api/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r /tmp/requirements.txt


FROM python:3.10-slim AS runtime

ENV MODELS_DIR=/app/models \
    PROCESSED_DIR=/app/data/processed \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /wheels /wheels
# Install from local wheels (no internet needed).
# Then uninstall GPU-only CUDA packages (XGBoost depends on them but doesn't need
# them on a CPU-only VPS). Remove /wheels dir to keep image size minimal.
RUN pip install --no-cache-dir /wheels/* \
    && pip uninstall -y nvidia-nccl-cu12 2>/dev/null || true \
    && rm -rf /wheels

COPY ml_pipeline/src ./ml_pipeline/src
COPY monitoring/evidently_ai ./monitoring/evidently_ai
COPY serving_api/app ./serving_api/app
RUN mkdir -p /app/models /app/data/processed && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serving_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]