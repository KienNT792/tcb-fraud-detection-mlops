FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ml_pipeline/requirements.txt /tmp/ml_pipeline_requirements.txt
RUN pip install --no-cache-dir -r /tmp/ml_pipeline_requirements.txt

COPY ml_pipeline ./ml_pipeline
COPY scripts ./scripts
COPY models ./models
COPY docs ./docs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "ml_pipeline.src.train"]
