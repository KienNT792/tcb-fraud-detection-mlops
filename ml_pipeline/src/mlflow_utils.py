"""
TCB Fraud Detection — Shared MLflow Utilities.

Centralises MLflow configuration, experiment setup, and tag-building logic
used by both the training and evaluation pipelines.  Extracted to satisfy
DRY and keep ``train.py`` / ``evaluate.py`` focused on their own concerns.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone


import mlflow

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME: str = "fraud-detection-training-pipeline"


def configure_mlflow(stage: str) -> str:
    """Set up the MLflow tracking URI, experiment, and return a run name.

    Resolution order for the run name:
    1. ``MLFLOW_RUN_NAME`` environment variable (if set).
    2. Auto-generated ``"{stage}-{utc_timestamp}"`` string.

    Args:
        stage: Pipeline stage identifier (e.g. ``"train"``, ``"evaluation"``).

    Returns:
        Human-readable MLflow run name.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        DEFAULT_EXPERIMENT_NAME,
    )
    mlflow.set_experiment(experiment_name)

    run_name = os.getenv("MLFLOW_RUN_NAME")
    if run_name:
        return run_name

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}-{timestamp}"


def build_mlflow_tags(
    stage: str,
    **artifact_dirs: str,
) -> dict[str, str]:
    """Build a tag dictionary for an MLflow run.

    Common Airflow / pipeline metadata tags are always included.
    Any additional keyword arguments are recorded as
    ``artifact.<key>`` tags — this allows each pipeline stage to
    declare its own set of artifact directories without duplicating
    the shared tag-building logic.

    Args:
        stage: Pipeline stage identifier.
        **artifact_dirs: Arbitrary ``name=path`` pairs that will be
            stored as ``artifact.<name>`` tags.  Examples::

                build_mlflow_tags(
                    "train",
                    processed_dir="/data/processed",
                    models_dir="/models",
                )

    Returns:
        Tag dictionary with empty values filtered out.
    """
    tags: dict[str, str] = {
        "pipeline.stage": stage,
        "pipeline.source": os.getenv("PIPELINE_SOURCE", "manual"),
        "pipeline.run_id": os.getenv("PIPELINE_RUN_ID", ""),
        "airflow.dag_id": os.getenv("AIRFLOW_PIPELINE_DAG_ID", ""),
        "airflow.task_id": os.getenv("AIRFLOW_PIPELINE_TASK_ID", ""),
        "airflow.run_id": os.getenv("AIRFLOW_PIPELINE_RUN_ID", ""),
        "airflow.logical_date": os.getenv("AIRFLOW_PIPELINE_LOGICAL_DATE", ""),
    }

    for name, path in artifact_dirs.items():
        tags[f"artifact.{name}"] = path

    return {key: value for key, value in tags.items() if value}
