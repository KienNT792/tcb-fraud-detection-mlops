FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY ml_pipeline/requirements.txt /tmp/ml_pipeline_requirements.txt
COPY serving_api/requirements.txt /tmp/serving_api_requirements.txt
RUN pip install --no-cache-dir \
    -r /tmp/ml_pipeline_requirements.txt \
    -r /tmp/serving_api_requirements.txt

WORKDIR /opt/airflow
