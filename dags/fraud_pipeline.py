from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
ML_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline")

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

shared_env = {
    "PYTHONPATH": PROJECT_ROOT,
    "MLFLOW_TRACKING_URI": os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow:5000",
    ),
    "MLFLOW_S3_ENDPOINT_URL": os.getenv(
        "MLFLOW_S3_ENDPOINT_URL",
        "http://minio:9000",
    ),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
    "AWS_SECRET_ACCESS_KEY": os.getenv(
        "AWS_SECRET_ACCESS_KEY",
        "minioadmin123",
    ),
}

with DAG(
    dag_id="fraud_detection_training_pipeline",
    description="Nightly fraud detection retraining pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["fraud-detection", "mlops"],
) as dag:
    preprocess = BashOperator(
        task_id="preprocess",
        cwd=ML_PIPELINE_DIR,
        env=shared_env,
        bash_command="python src/preprocess.py",
    )

    train = BashOperator(
        task_id="train",
        cwd=ML_PIPELINE_DIR,
        env=shared_env,
        bash_command="python src/train.py",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        cwd=ML_PIPELINE_DIR,
        env=shared_env,
        bash_command="python src/evaluate.py",
    )

    preprocess >> train >> evaluate
