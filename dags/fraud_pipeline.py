from __future__ import annotations

import os
import json
import urllib.request
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator


PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
ML_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline")
AIRFLOW_VENV_PYTHON = os.getenv("AIRFLOW_VENV_PYTHON", "/opt/airflow/venv/bin/python")
AIRFLOW_USER_SITE = os.getenv(
    "AIRFLOW_USER_SITE",
    "/home/airflow/.local/lib/python3.10/site-packages",
)
DEFAULT_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "fraud-detection-training-pipeline",
)
DEFAULT_RETRAIN_METRIC = os.getenv("MODEL_QUALITY_METRIC", "eval_f1")


def _parse_threshold(raw_value: str) -> float:
    threshold = float(raw_value)
    return threshold / 100.0 if threshold > 1 else threshold


def _mlflow_post(tracking_uri: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{tracking_uri.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_metric_value(run: dict, metric_name: str) -> float | None:
    metrics = run.get("data", {}).get("metrics", [])
    for metric in metrics:
        if metric.get("key") == metric_name:
            value = metric.get("value")
            return float(value) if value is not None else None
    return None


def _find_experiment_by_name(tracking_uri: str, experiment_name: str) -> dict | None:
    experiments_response = _mlflow_post(
        tracking_uri,
        "/api/2.0/mlflow/experiments/search",
        {"max_results": 100},
    )
    experiments = experiments_response.get("experiments", [])
    for experiment in experiments:
        if experiment.get("name") == experiment_name:
            return experiment
    return None


def should_trigger_retraining() -> str:
    tracking_uri = shared_env["MLFLOW_TRACKING_URI"]
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    metric_name = os.getenv("MODEL_QUALITY_METRIC", DEFAULT_RETRAIN_METRIC)
    threshold = _parse_threshold(os.getenv("MODEL_QUALITY_THRESHOLD", "0.80"))
    experiment = _find_experiment_by_name(tracking_uri, experiment_name)

    if not experiment:
        print(
            f"MLflow experiment '{experiment_name}' not found. "
            "Triggering retraining by default."
        )
        return "preprocess"

    runs_response = _mlflow_post(
        tracking_uri,
        "/api/2.0/mlflow/runs/search",
        {
            "experiment_ids": [experiment["experiment_id"]],
            "filter": "attributes.status = 'FINISHED'",
            "order_by": ["attributes.start_time DESC"],
            "max_results": 25,
        },
    )
    runs = runs_response.get("runs", [])

    for run in runs:
        metric_value = _extract_metric_value(run, metric_name)
        if metric_value is None:
            continue

        print(
            f"Latest MLflow metric {metric_name}={metric_value:.4f}; "
            f"threshold={threshold:.4f}; run_id={run['info']['run_id']}"
        )
        return "preprocess" if metric_value < threshold else "skip_retraining"

    print(
        f"No finished MLflow run with metric '{metric_name}' found. "
        "Triggering retraining by default."
    )
    return "preprocess"


def build_task_env(task_id: str) -> dict[str, str]:
    return {
        **shared_env,
        "PIPELINE_SOURCE": "airflow",
        "PIPELINE_STAGE": task_id,
        "PIPELINE_RUN_ID": "{{ dag_run.run_id if dag_run else '' }}",
        "AIRFLOW_PIPELINE_DAG_ID": "{{ dag.dag_id }}",
        "AIRFLOW_PIPELINE_TASK_ID": task_id,
        "AIRFLOW_PIPELINE_RUN_ID": "{{ dag_run.run_id if dag_run else '' }}",
        "AIRFLOW_PIPELINE_LOGICAL_DATE": "{{ ts }}",
        "MLFLOW_RUN_NAME": f"{task_id}-" + "{{ ts_nodash }}",
    }

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

shared_env = {
    "PYTHONPATH": f"{AIRFLOW_USER_SITE}:{PROJECT_ROOT}",
    "MLFLOW_TRACKING_URI": os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow:5000",
    ),
    "MLFLOW_EXPERIMENT_NAME": os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        DEFAULT_EXPERIMENT_NAME,
    ),
    "MLFLOW_S3_ENDPOINT_URL": os.getenv(
        "MLFLOW_S3_ENDPOINT_URL",
        "http://minio:9000",
    ),
    "MODEL_QUALITY_METRIC": os.getenv(
        "MODEL_QUALITY_METRIC",
        DEFAULT_RETRAIN_METRIC,
    ),
    "MODEL_QUALITY_THRESHOLD": os.getenv(
        "MODEL_QUALITY_THRESHOLD",
        "0.80",
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
    check_model_quality = BranchPythonOperator(
        task_id="check_model_quality",
        python_callable=should_trigger_retraining,
    )

    skip_retraining = EmptyOperator(task_id="skip_retraining")

    preprocess = BashOperator(
        task_id="preprocess",
        cwd=ML_PIPELINE_DIR,
        env=build_task_env("preprocess"),
        bash_command=f"{AIRFLOW_VENV_PYTHON} src/preprocess.py",
    )

    train = BashOperator(
        task_id="train",
        cwd=ML_PIPELINE_DIR,
        env=build_task_env("train"),
        bash_command=f"{AIRFLOW_VENV_PYTHON} src/train.py",
    )

    evaluate = BashOperator(
        task_id="evaluate",
        cwd=ML_PIPELINE_DIR,
        env=build_task_env("evaluate"),
        bash_command=f"{AIRFLOW_VENV_PYTHON} src/evaluate.py",
    )

    check_model_quality >> [preprocess, skip_retraining]
    preprocess >> train >> evaluate
