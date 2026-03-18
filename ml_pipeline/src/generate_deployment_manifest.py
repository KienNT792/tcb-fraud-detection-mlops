from __future__ import annotations

import argparse
import json
from pathlib import Path

from infrastructure.pipeline import PIPELINE_CONFIG

from .mlflow_utils import default_model_name, write_json


def generate_manifest(
    models_dir: str = str(PIPELINE_CONFIG.paths.models_dir),
    output_path: str = str(PIPELINE_CONFIG.paths.deployment_manifest_path),
) -> dict:
    models_path = Path(models_dir)
    registry_info_path = models_path / "registry_info.json"
    if not registry_info_path.exists():
        raise FileNotFoundError(
            f"Missing {registry_info_path}. Run register_model.py before deploy manifest generation."
        )

    with open(registry_info_path, encoding="utf-8") as handle:
        registry_info = json.load(handle)

    manifest = {
        "app": "tcb-fraud-detection-api",
        "model_name": registry_info.get("model_name", default_model_name()),
        "model_version": registry_info.get("model_version", "unknown"),
        "stage": registry_info.get("stage", "Staging"),
        "tracking_uri": registry_info.get("tracking_uri"),
        "run_id": registry_info.get("run_id"),
        "dataset_version": registry_info.get("dataset_version"),
        "git_commit": registry_info.get("git_commit"),
        "threshold": registry_info.get("threshold"),
    }
    write_json(Path(output_path), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deployment manifest from the latest registry info.")
    parser.add_argument("--models-dir", default=str(PIPELINE_CONFIG.paths.models_dir))
    parser.add_argument("--output-path", default=str(PIPELINE_CONFIG.paths.deployment_manifest_path))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_manifest(args.models_dir, args.output_path)


if __name__ == "__main__":
    main()
