from __future__ import annotations

import logging
from typing import Any

from mlflow import MlflowClient

logger = logging.getLogger(__name__)


def register_model_from_run(
    *,
    run_id: str,
    artifact_path: str,
    model_name: str,
) -> int:
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )
    logger.info(
        "Registered model version in MLflow Registry"
        " | name=%s | version=%s | run_id=%s",
        model_name,
        registered.version,
        run_id,
    )
    return int(registered.version)


def transition_model_version_stage(
    *,
    model_name: str,
    version: int,
    stage: str,
    archive_existing_versions: bool = True,
) -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )
    logger.info(
        "Transitioned model stage | name=%s | version=%s | stage=%s",
        model_name,
        version,
        stage,
    )


def find_latest_version_by_run(
    *,
    model_name: str,
    run_id: str,
) -> int | None:
    client = MlflowClient()
    versions: list[Any] = client.search_model_versions(
        f"name='{model_name}'",
    )
    matched = [int(v.version) for v in versions if getattr(v, "run_id", "") == run_id]
    if not matched:
        return None
    return max(matched)
