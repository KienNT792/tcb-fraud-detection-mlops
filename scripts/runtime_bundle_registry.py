#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline.src.model_registry import (  # noqa: E402
    register_model_from_run,
    transition_model_version_stage,
)
from ml_pipeline.src.runtime_bundle import (  # noqa: E402
    MODEL_RUNTIME_FILES,
    OPTIONAL_MODEL_RUNTIME_FILES,
    PROCESSED_RUNTIME_FILES,
    RUNTIME_BUNDLE_ARTIFACT_PATH,
    log_runtime_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish or download MLflow runtime bundles.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", ""),
        help="MLflow tracking URI.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    publish_parser = subparsers.add_parser(
        "publish-run",
        help="Attach runtime bundle artifacts to an existing MLflow run.",
    )
    publish_parser.add_argument("--run-id", required=True)
    publish_parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
    )
    publish_parser.add_argument(
        "--processed-dir",
        default=str(REPO_ROOT / "data" / "processed"),
    )

    bootstrap_parser = subparsers.add_parser(
        "bootstrap-run",
        help="Register a run into MLflow Registry, transition stage, and attach a runtime bundle.",
    )
    bootstrap_parser.add_argument("--run-id", required=True)
    bootstrap_parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "tcb-fraud-xgboost"),
    )
    bootstrap_parser.add_argument(
        "--stage",
        default=os.getenv("MLFLOW_DEPLOY_STAGE", "Production"),
    )
    bootstrap_parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
    )
    bootstrap_parser.add_argument(
        "--processed-dir",
        default=str(REPO_ROOT / "data" / "processed"),
    )

    bootstrap_artifacts_parser = subparsers.add_parser(
        "bootstrap-artifacts",
        help="Create the first MLflow run/model version from repository bootstrap artifacts.",
    )
    bootstrap_artifacts_parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "tcb-fraud-xgboost"),
    )
    bootstrap_artifacts_parser.add_argument(
        "--stage",
        default=os.getenv("MLFLOW_DEPLOY_STAGE", "Production"),
    )
    bootstrap_artifacts_parser.add_argument(
        "--model-artifact-dir",
        default=str(REPO_ROOT / "bootstrap_runtime_bundle" / "mlflow_model"),
    )
    bootstrap_artifacts_parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
    )
    bootstrap_artifacts_parser.add_argument(
        "--processed-dir",
        default=str(REPO_ROOT / "bootstrap_runtime_bundle" / "processed"),
    )
    bootstrap_artifacts_parser.add_argument(
        "--experiment-name",
        default=os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            "fraud-detection-bootstrap",
        ),
    )

    publish_stage_parser = subparsers.add_parser(
        "publish-stage",
        help="Attach runtime bundle artifacts to the latest model version in a registry stage.",
    )
    publish_stage_parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "tcb-fraud-xgboost"),
    )
    publish_stage_parser.add_argument(
        "--stage",
        default=os.getenv("MLFLOW_DEPLOY_STAGE", "Production"),
    )
    publish_stage_parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
    )
    publish_stage_parser.add_argument(
        "--processed-dir",
        default=str(REPO_ROOT / "data" / "processed"),
    )

    download_parser = subparsers.add_parser(
        "download-stage",
        help="Download runtime bundle from the latest model version in a registry stage.",
    )
    download_parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "tcb-fraud-xgboost"),
    )
    download_parser.add_argument(
        "--stage",
        default=os.getenv("MLFLOW_DEPLOY_STAGE", "Production"),
    )
    download_parser.add_argument(
        "--artifact-path",
        default=os.getenv(
            "MLFLOW_RUNTIME_BUNDLE_PATH",
            RUNTIME_BUNDLE_ARTIFACT_PATH,
        ),
    )
    download_parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT),
    )
    return parser


def configure_tracking_uri(tracking_uri: str) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def log_metrics_from_file(models_dir: str) -> None:
    metrics_path = Path(models_dir) / "metrics.json"
    if not metrics_path.exists():
        return

    with open(metrics_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    numeric_metrics = {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)


def publish_run(
    *,
    run_id: str,
    models_dir: str,
    processed_dir: str,
) -> None:
    with mlflow.start_run(run_id=run_id):
        log_runtime_bundle(
            models_dir=models_dir,
            processed_dir=processed_dir,
            extra_metadata={
                "published_at": datetime.now(tz=timezone.utc).isoformat(),
                "published_by": "runtime_bundle_registry.py",
            },
        )
        mlflow.set_tag("runtime_bundle.ready", "true")
        mlflow.set_tag(
            "runtime_bundle.artifact_path",
            RUNTIME_BUNDLE_ARTIFACT_PATH,
        )
    print(f"Runtime bundle published to run {run_id}")


def ensure_registered_model(
    *,
    client: MlflowClient,
    model_name: str,
) -> None:
    try:
        client.get_registered_model(model_name)
        return
    except RestException as exc:
        if "RESOURCE_DOES_NOT_EXIST" not in str(exc):
            raise

    client.create_registered_model(model_name)


def bootstrap_artifacts(
    *,
    model_name: str,
    stage: str,
    model_artifact_dir: str,
    models_dir: str,
    processed_dir: str,
    experiment_name: str,
) -> None:
    client = MlflowClient()
    ensure_registered_model(client=client, model_name=model_name)

    model_artifact_path = Path(model_artifact_dir)
    if not (model_artifact_path / "MLmodel").exists():
        raise FileNotFoundError(
            f"Missing MLflow model definition at {model_artifact_path / 'MLmodel'}"
        )

    models_path = Path(models_dir)
    processed_path = Path(processed_dir)
    for filename in MODEL_RUNTIME_FILES:
        if not (models_path / filename).exists():
            raise FileNotFoundError(
                f"Missing required model artifact: {models_path / filename}"
            )
    for filename in PROCESSED_RUNTIME_FILES:
        if not (processed_path / filename).exists():
            raise FileNotFoundError(
                f"Missing required processed artifact: {processed_path / filename}"
            )

    mlflow.set_experiment(experiment_name)
    run_name = f"bootstrap-{model_name}-{stage.lower()}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "pipeline.source": "repo-bootstrap",
                "pipeline.stage": "bootstrap",
                "bootstrap.model_name": model_name,
                "bootstrap.target_stage": stage,
                "bootstrap.model_artifact_dir": str(model_artifact_path.resolve()),
                "artifact.models_dir": str(models_path.resolve()),
                "artifact.processed_dir": str(processed_path.resolve()),
            }
        )
        mlflow.log_artifacts(str(model_artifact_path), artifact_path="model")
        log_metrics_from_file(models_dir)
        log_runtime_bundle(
            models_dir=models_dir,
            processed_dir=processed_dir,
            extra_metadata={
                "published_at": datetime.now(tz=timezone.utc).isoformat(),
                "published_by": "runtime_bundle_registry.py",
                "bootstrap_model_name": model_name,
                "bootstrap_stage": stage,
            },
        )
        mlflow.set_tag("runtime_bundle.ready", "true")
        mlflow.set_tag(
            "runtime_bundle.artifact_path",
            RUNTIME_BUNDLE_ARTIFACT_PATH,
        )
        run_id = mlflow.active_run().info.run_id

    version = register_model_from_run(
        run_id=run_id,
        artifact_path="model",
        model_name=model_name,
    )
    transition_model_version_stage(
        model_name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    print(
        json.dumps(
            {
                "model_name": model_name,
                "stage": stage,
                "version": version,
                "run_id": run_id,
                "model_artifact_dir": str(model_artifact_path.resolve()),
                "models_dir": str(models_path.resolve()),
                "processed_dir": str(processed_path.resolve()),
            },
            indent=2,
        )
    )


def bootstrap_run(
    *,
    run_id: str,
    model_name: str,
    stage: str,
    models_dir: str,
    processed_dir: str,
) -> None:
    client = MlflowClient()
    ensure_registered_model(client=client, model_name=model_name)
    version = register_model_from_run(
        run_id=run_id,
        artifact_path="model",
        model_name=model_name,
    )
    transition_model_version_stage(
        model_name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    publish_run(
        run_id=run_id,
        models_dir=models_dir,
        processed_dir=processed_dir,
    )
    print(
        json.dumps(
            {
                "model_name": model_name,
                "stage": stage,
                "version": version,
                "run_id": run_id,
            },
            indent=2,
        )
    )


def resolve_latest_model_version(
    *,
    client: MlflowClient,
    model_name: str,
    stage: str,
):
    versions = client.get_latest_versions(model_name, [stage])
    if not versions:
        raise RuntimeError(
            f"No model version found for name={model_name} stage={stage}"
        )
    return versions[0]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def download_stage_bundle(
    *,
    model_name: str,
    stage: str,
    artifact_path: str,
    output_root: str,
) -> None:
    client = MlflowClient()
    version = resolve_latest_model_version(
        client=client,
        model_name=model_name,
        stage=stage,
    )
    run_id = version.run_id
    download_parent = Path(output_root) / ".runtime_bundle"
    download_parent.mkdir(parents=True, exist_ok=True)

    bundle_dir = Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=str(download_parent),
        )
    )
    models_src = bundle_dir / "models"
    processed_src = bundle_dir / "processed"
    models_dst = Path(output_root) / "models"
    processed_dst = Path(output_root) / "data" / "processed"

    for filename in MODEL_RUNTIME_FILES:
        copy_file(models_src / filename, models_dst / filename)

    for parent, filename in OPTIONAL_MODEL_RUNTIME_FILES:
        candidate = models_src / parent / filename
        if candidate.exists():
            copy_file(candidate, models_dst / parent / filename)

    for filename in PROCESSED_RUNTIME_FILES:
        copy_file(processed_src / filename, processed_dst / filename)

    manifest = {
        "slot": "stable",
        "model_id": f"{model_name}-v{version.version}",
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        "registry_model_name": model_name,
        "registry_stage": stage,
        "registry_version": str(version.version),
        "run_id": run_id,
        "runtime_bundle_path": artifact_path,
    }
    manifest_path = models_dst / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "model_name": model_name,
                "stage": stage,
                "version": str(version.version),
                "run_id": run_id,
                "bundle_dir": str(bundle_dir),
                "output_root": str(Path(output_root).resolve()),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_tracking_uri(args.tracking_uri)

    if args.command == "publish-run":
        publish_run(
            run_id=args.run_id,
            models_dir=args.models_dir,
            processed_dir=args.processed_dir,
        )
        return

    if args.command == "bootstrap-run":
        bootstrap_run(
            run_id=args.run_id,
            model_name=args.model_name,
            stage=args.stage,
            models_dir=args.models_dir,
            processed_dir=args.processed_dir,
        )
        return

    if args.command == "bootstrap-artifacts":
        bootstrap_artifacts(
            model_name=args.model_name,
            stage=args.stage,
            model_artifact_dir=args.model_artifact_dir,
            models_dir=args.models_dir,
            processed_dir=args.processed_dir,
            experiment_name=args.experiment_name,
        )
        return

    if args.command == "publish-stage":
        client = MlflowClient()
        version = resolve_latest_model_version(
            client=client,
            model_name=args.model_name,
            stage=args.stage,
        )
        publish_run(
            run_id=version.run_id,
            models_dir=args.models_dir,
            processed_dir=args.processed_dir,
        )
        return

    download_stage_bundle(
        model_name=args.model_name,
        stage=args.stage,
        artifact_path=args.artifact_path,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
