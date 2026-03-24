from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

RUNTIME_BUNDLE_ARTIFACT_PATH = "runtime_bundle"

MODEL_RUNTIME_FILES: tuple[str, ...] = (
    "xgb_fraud_model.joblib",
    "metrics.json",
    "feature_importance.csv",
)

OPTIONAL_MODEL_RUNTIME_FILES: tuple[tuple[str, str], ...] = (
    ("evaluation", "evaluation.json"),
)

PROCESSED_RUNTIME_FILES: tuple[str, ...] = (
    "features.json",
    "customer_stats.parquet",
    "segment_label_map.json",
    "amount_median_train.json",
    "categorical_maps.json",
)


def build_runtime_bundle_metadata(
    models_dir: str,
    processed_dir: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "models_dir": str(Path(models_dir).resolve()),
        "processed_dir": str(Path(processed_dir).resolve()),
        "model_files": list(MODEL_RUNTIME_FILES),
        "processed_files": list(PROCESSED_RUNTIME_FILES),
        "optional_model_files": [
            str(Path(parent) / filename)
            for parent, filename in OPTIONAL_MODEL_RUNTIME_FILES
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def log_runtime_bundle(
    *,
    models_dir: str,
    processed_dir: str,
    artifact_path: str = RUNTIME_BUNDLE_ARTIFACT_PATH,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    models_path = Path(models_dir)
    processed_path = Path(processed_dir)

    for filename in MODEL_RUNTIME_FILES:
        mlflow.log_artifact(
            str(models_path / filename),
            artifact_path=f"{artifact_path}/models",
        )

    for parent, filename in OPTIONAL_MODEL_RUNTIME_FILES:
        candidate = models_path / parent / filename
        if candidate.exists():
            mlflow.log_artifact(
                str(candidate),
                artifact_path=f"{artifact_path}/models/{parent}",
            )

    for filename in PROCESSED_RUNTIME_FILES:
        mlflow.log_artifact(
            str(processed_path / filename),
            artifact_path=f"{artifact_path}/processed",
        )

    metadata = build_runtime_bundle_metadata(
        models_dir,
        processed_dir,
        extra=extra_metadata,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(
            str(metadata_path),
            artifact_path=artifact_path,
        )
