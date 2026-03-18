from __future__ import annotations

import json
from pathlib import Path

from ml_pipeline.src.generate_deployment_manifest import generate_manifest
from ml_pipeline.src.materialize_feature_ready import materialize_feature_ready


def test_materialize_feature_ready_copies_expected_files(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    feature_ready_dir = tmp_path / "feature-ready"
    processed_dir.mkdir()

    for filename in (
        "features.json",
        "customer_stats.parquet",
        "segment_label_map.json",
        "amount_median_train.json",
        "categorical_maps.json",
    ):
        (processed_dir / filename).write_text("demo", encoding="utf-8")

    result = materialize_feature_ready(str(processed_dir), str(feature_ready_dir))
    assert len(result["copied_files"]) == 5
    assert (feature_ready_dir / "manifest.json").exists()


def test_generate_manifest_reads_registry_info(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry_info = {
        "model_name": "tcb-fraud-detector",
        "model_version": "8",
        "stage": "Staging",
        "tracking_uri": "http://mlflow:5000",
        "run_id": "abc123",
        "dataset_version": "demo-v1",
        "git_commit": "deadbeef",
        "threshold": 0.6788,
    }
    with open(models_dir / "registry_info.json", "w", encoding="utf-8") as handle:
        json.dump(registry_info, handle)

    output_path = tmp_path / "artifacts" / "release_manifest.json"
    manifest = generate_manifest(str(models_dir), str(output_path))
    assert manifest["model_version"] == "8"
    assert output_path.exists()
