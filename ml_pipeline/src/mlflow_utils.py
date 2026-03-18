from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

from infrastructure.pipeline import PIPELINE_CONFIG


DEFAULT_EXPERIMENT = "tcb-fraud-detection"
DEFAULT_MODEL_NAME = "tcb-fraud-detector"


def configure_mlflow(experiment_name: str | None = None) -> str:
    tracking_uri = PIPELINE_CONFIG.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    experiment = experiment_name or PIPELINE_CONFIG.mlflow.experiment_name
    mlflow.set_experiment(experiment)
    return tracking_uri


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_run_tags(stage: str) -> dict[str, str]:
    return {
        "pipeline_stage": stage,
        "dataset_version": PIPELINE_CONFIG.mlflow.dataset_version,
        "config_version": PIPELINE_CONFIG.mlflow.config_version,
        "pipeline_trigger": PIPELINE_CONFIG.mlflow.pipeline_trigger,
        "pipeline_actor": PIPELINE_CONFIG.mlflow.pipeline_actor,
        "git_commit": get_git_commit(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def default_model_name() -> str:
    return PIPELINE_CONFIG.mlflow.model_name


def build_lineage_payload(run_id: str, stage: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        "model_name": default_model_name(),
        "tracking_uri": mlflow.get_tracking_uri(),
        "dataset_version": PIPELINE_CONFIG.mlflow.dataset_version,
        "config_version": PIPELINE_CONFIG.mlflow.config_version,
        "pipeline_trigger": PIPELINE_CONFIG.mlflow.pipeline_trigger,
        "pipeline_actor": PIPELINE_CONFIG.mlflow.pipeline_actor,
        "git_commit": get_git_commit(),
        "recorded_at": datetime.now(tz=timezone.utc).isoformat(),
    }
