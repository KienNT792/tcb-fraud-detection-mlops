from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from mlflow import MlflowClient
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from ml_pipeline.src.registry_metadata import read_registry_metadata
from scripts.runtime_bundle_registry import (
    download_registered_version_bundle,
    publish_run,
    resolve_model_version,
)


PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT", "/opt/airflow/project",
)
ML_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline")
AIRFLOW_VENV_PYTHON = os.getenv(
    "AIRFLOW_VENV_PYTHON",
    "/opt/airflow/venv/bin/python",
)
AIRFLOW_USER_SITE = os.getenv(
    "AIRFLOW_USER_SITE",
    "/home/airflow/.local/lib/python3.10/site-packages",
)
DEFAULT_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "fraud-detection-training-pipeline",
)
DEFAULT_RETRAIN_METRIC = os.getenv("MODEL_QUALITY_METRIC", "eval_f1")

# Canary deployment paths (shared volume with FastAPI containers)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CANDIDATE_DIR = os.path.join(MODELS_DIR, "deployments", "candidate")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CANDIDATE_PROCESSED_DIR = os.path.join(CANDIDATE_DIR, "processed")
EVALUATION_DIR = os.path.join(MODELS_DIR, "evaluation")

# FastAPI candidate container URL (accessible via Docker network)
CANDIDATE_URL = os.getenv(
    "CANDIDATE_SERVICE_URL",
    "http://fastapi-candidate:8000",
)

# Model artifact filenames
_MODEL_FILENAME = "xgb_fraud_model.joblib"
_EVALUATION_DIRNAME = "evaluation"


def _parse_threshold(raw_value: str) -> float:
    threshold = float(raw_value)
    return threshold / 100.0 if threshold > 1 else threshold


def _mlflow_post(
    tracking_uri: str, path: str, payload: dict,
) -> dict:
    request = urllib.request.Request(
        f"{tracking_uri.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=15,
    ) as response:
        return json.loads(
            response.read().decode("utf-8"),
        )


def _extract_metric_value(run: dict, metric_name: str) -> float | None:
    metrics = run.get("data", {}).get("metrics", [])
    for metric in metrics:
        if metric.get("key") == metric_name:
            value = metric.get("value")
            return float(value) if value is not None else None
    return None


def _find_experiment_by_name(
    tracking_uri: str, experiment_name: str,
) -> dict | None:
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
    """Decide whether to retrain based on latest MLflow eval_f1 metric."""
    tracking_uri = shared_env["MLFLOW_TRACKING_URI"]
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        DEFAULT_EXPERIMENT_NAME,
    )
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
        metric_value = _extract_metric_value(
            run, metric_name,
        )
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


def stage_candidate_after_eval(**context: Any) -> dict[str, Any]:
    """Publish the evaluated runtime bundle and sync it into candidate slot.

    Raises:
        RuntimeError: If evaluation status is FAIL.
        FileNotFoundError:
            If required artifacts or registry metadata are missing.
    """
    models_path = Path(MODELS_DIR)
    candidate_path = Path(CANDIDATE_DIR)
    eval_path = models_path / _EVALUATION_DIRNAME / "evaluation.json"
    status = "UNKNOWN"

    # Step 1 — Check evaluation status
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as fh:
            evaluation = json.load(fh)
        status = evaluation.get("overall_status", "UNKNOWN")
        print(f"Evaluation status: {status}")
        if status == "FAIL":
            raise RuntimeError(
                "Model evaluation FAILED — blocking candidate staging. "
                "Metrics did not meet baseline requirements."
            )
    else:
        print(
            f"WARNING: {eval_path} not found — proceeding with "
            "staging (evaluation may not have been saved)."
        )

    # Step 2 — Verify source model exists
    source_model = models_path / _MODEL_FILENAME
    if not source_model.exists():
        raise FileNotFoundError(
            f"Trained model not found at {source_model}. "
            "train.py must run before stage_candidate."
        )

    registry_metadata = read_registry_metadata(models_path)
    if not registry_metadata:
        raise FileNotFoundError(
            "Missing registry metadata in models/. "
            "train.py must register the model before candidate staging."
        )

    dag_run = context.get("dag_run")
    run_id = dag_run.run_id if dag_run else "manual"
    registry_model_name = str(registry_metadata["model_name"])
    registry_version = int(registry_metadata["version"])
    runtime_bundle_path = str(
        registry_metadata.get("runtime_bundle_artifact_path", "runtime_bundle")
    )

    publish_run(
        run_id=str(registry_metadata["run_id"]),
        models_dir=str(models_path),
        processed_dir=PROCESSED_DIR,
        extra_metadata={
            "published_by": "airflow.stage_candidate",
            "evaluation_status": status,
            "airflow_run_id": run_id,
        },
    )

    client = MlflowClient()
    version = resolve_model_version(
        client=client,
        model_name=registry_model_name,
        version=registry_version,
    )
    result = download_registered_version_bundle(
        model_name=registry_model_name,
        version=version,
        artifact_path=runtime_bundle_path,
        output_root=PROJECT_ROOT,
        models_output_dir=str(candidate_path),
        processed_output_dir=CANDIDATE_PROCESSED_DIR,
        manifest_slot="candidate",
        registry_stage=str(registry_metadata.get("stage", "Staging")),
    )

    manifest_path = candidate_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["staged_by"] = "airflow"
    manifest["airflow_run_id"] = run_id
    manifest["source_model_dir"] = str(models_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    model_id = str(result["model_id"])
    print(
        "Candidate model staged from MLflow Registry: "
        f"model_id={model_id} | version={registry_version}"
    )
    return {
        "model_id": model_id,
        "registry_model_name": registry_model_name,
        "registry_version": registry_version,
        "run_id": str(registry_metadata["run_id"]),
    }


def verify_candidate_health(**context: Any) -> None:
    """Poll candidate FastAPI /health endpoint to confirm model loaded.

    If the candidate service profile is not running, the staged artifacts are
    left on disk and verification is skipped. When the profile is started
    later, the candidate container will load the staged manifest.

    Otherwise waits up to 60 seconds for the candidate container to report the
    expected model_id from the stage_candidate task.

    Raises:
        RuntimeError: If the candidate does not load the model in time.
    """
    # Pull model_id from XCom (set by stage_candidate)
    ti = context["ti"]
    stage_result = ti.xcom_pull(task_ids="stage_candidate") or {}
    expected_model_id = stage_result.get("model_id", "")

    candidate_url = CANDIDATE_URL.rstrip("/")
    health_url = f"{candidate_url}/health"

    service_probe_attempts = 3
    service_probe_wait_seconds = 2
    for probe_attempt in range(1, service_probe_attempts + 1):
        try:
            request = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(request, timeout=5):
                break
        except Exception as exc:
            print(
                "Candidate reachability probe "
                f"{probe_attempt}/{service_probe_attempts}"
                f" failed: {exc}"
            )
            if probe_attempt < service_probe_attempts:
                time.sleep(service_probe_wait_seconds)
    else:
        print(
            "Candidate service is not running. Candidate artifacts were "
            "staged successfully and will load when the `candidate` Docker "
            "profile is started."
        )
        return

    max_attempts = 12
    wait_seconds = 5
    last_payload: dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        try:
            request = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                last_payload = payload

                loaded_version = payload.get("model_version")
                model_loaded = payload.get("model_loaded", False)

                print(
                    f"Attempt {attempt}/{max_attempts} | "
                    f"model_loaded={model_loaded} | "
                    f"model_version={loaded_version} | "
                    f"expected={expected_model_id}"
                )

                if model_loaded and loaded_version == expected_model_id:
                    print(
                        f"✓ Candidate verified — model {expected_model_id} "
                        f"is live at {candidate_url}"
                    )
                    return
        except Exception as exc:
            last_payload = {"error": str(exc)}
            print(f"Attempt {attempt}/{max_attempts} | error: {exc}")

        if attempt < max_attempts:
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Candidate at {candidate_url} did not load model "
        f"{expected_model_id} within {max_attempts * wait_seconds}s. "
        f"Last response: {last_payload}"
    )


def build_task_env(task_id: str) -> dict[str, str]:
    """Build environment variables for BashOperator tasks."""
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
}

# Credentials — must be set via environment variables (no hardcoded defaults)
_aws_key = os.getenv("AWS_ACCESS_KEY_ID")
_aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
if not _aws_key or not _aws_secret:
    raise EnvironmentError(
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set. "
        "Configure them in .env or docker-compose environment."
    )
shared_env["AWS_ACCESS_KEY_ID"] = str(_aws_key)
shared_env["AWS_SECRET_ACCESS_KEY"] = str(_aws_secret)

# ═══════════════════════════════════════════════════════════════════════
# DAG Definition
# ═══════════════════════════════════════════════════════════════════════
with DAG(
    dag_id="fraud_detection_training_pipeline",
    description=(
        "Nightly fraud detection retraining pipeline with "
        "automatic canary candidate staging."
    ),
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["fraud-detection", "mlops", "canary"],
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

    stage_candidate = PythonOperator(
        task_id="stage_candidate",
        python_callable=stage_candidate_after_eval,
    )

    verify_candidate = PythonOperator(
        task_id="verify_candidate",
        python_callable=verify_candidate_health,
    )

    # Task dependencies:
    #
    #   check_model_quality ──→ preprocess → train → evaluate
    #                      ↘                           ↓
    #                   skip_retraining         stage_candidate
    #                                                  ↓
    #                                          verify_candidate
    check_model_quality >> [preprocess, skip_retraining]
    preprocess >> train >> evaluate >> stage_candidate >> verify_candidate
