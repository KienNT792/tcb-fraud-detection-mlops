from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from infrastructure.pipeline import PIPELINE_CONFIG

from .mlflow_utils import (
    build_lineage_payload,
    build_run_tags,
    configure_mlflow,
    default_model_name,
    write_json,
)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _assert_acceptance(
    evaluation: dict[str, Any],
    min_recall: float,
    min_pr_auc: float | None,
    min_f1: float | None,
) -> None:
    metrics = evaluation["threshold_metrics"]
    if metrics["recall"] < min_recall:
        raise ValueError(f"Recall {metrics['recall']} is below {min_recall}")
    if min_pr_auc is not None and metrics["pr_auc"] < min_pr_auc:
        raise ValueError(f"PR-AUC {metrics['pr_auc']} is below {min_pr_auc}")
    if min_f1 is not None and metrics["f1"] < min_f1:
        raise ValueError(f"F1 {metrics['f1']} is below {min_f1}")
    if evaluation.get("overall_status") != "PASS":
        raise ValueError("Evaluation overall_status is not PASS")


def register_model(
    models_dir: str = str(PIPELINE_CONFIG.paths.models_dir),
    promote_to: str = "Staging",
    min_recall: float = PIPELINE_CONFIG.mlflow.min_recall,
    min_pr_auc: float | None = None,
    min_f1: float | None = None,
) -> dict[str, Any]:
    configure_mlflow()
    client = MlflowClient()

    base_dir = Path(models_dir)
    training_run = _load_json(base_dir / "training_run.json")
    evaluation = _load_json(base_dir / "evaluation" / "evaluation.json")
    _assert_acceptance(evaluation, min_recall, min_pr_auc, min_f1)

    model_name = training_run.get("model_name", default_model_name())
    run_id = training_run["run_id"]
    model_uri = f"runs:/{run_id}/model"

    mlflow.set_tags(build_run_tags("register"))
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = registered.version
    lineage = build_lineage_payload(run_id, "register")
    lineage["model_version"] = version
    lineage["source_model_uri"] = model_uri
    lineage["threshold"] = evaluation["threshold_metrics"]["threshold"]
    lineage["evaluation_status"] = evaluation["overall_status"]

    for key, value in lineage.items():
        client.set_model_version_tag(model_name, version, key, str(value))

    if promote_to:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=promote_to,
            archive_existing_versions=(promote_to == "Production"),
        )
        lineage["stage"] = promote_to

    write_json(base_dir / "registry_info.json", lineage)
    return lineage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the trained fraud model in MLflow Model Registry.")
    parser.add_argument("--models-dir", default=str(PIPELINE_CONFIG.paths.models_dir))
    parser.add_argument("--promote-to", default="Staging")
    parser.add_argument("--min-recall", type=float, default=PIPELINE_CONFIG.mlflow.min_recall)
    parser.add_argument("--min-pr-auc", type=float, default=None)
    parser.add_argument("--min-f1", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_model(
        models_dir=args.models_dir,
        promote_to=args.promote_to,
        min_recall=args.min_recall,
        min_pr_auc=args.min_pr_auc,
        min_f1=args.min_f1,
    )


if __name__ == "__main__":
    main()
