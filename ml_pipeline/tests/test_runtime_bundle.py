from __future__ import annotations

import pandas as pd

from ml_pipeline.src.runtime_bundle import (
    DRIFT_REFERENCE_FILENAME,
    ensure_optional_processed_runtime_artifacts,
)


def test_drift_reference_artifact_is_created_from_train_parquet(tmp_path):
    train_df = pd.DataFrame(
        {
            "amount": list(range(250)),
            "hour": [value % 24 for value in range(250)],
            "is_fraud": [value % 2 for value in range(250)],
        }
    )
    train_df.to_parquet(tmp_path / "train.parquet", index=False)

    artifacts = ensure_optional_processed_runtime_artifacts(
        tmp_path,
        drift_reference_sample_size=50,
    )

    drift_reference_path = tmp_path / DRIFT_REFERENCE_FILENAME
    assert artifacts == {DRIFT_REFERENCE_FILENAME: drift_reference_path}
    assert drift_reference_path.exists()

    reference_df = pd.read_parquet(drift_reference_path)
    assert len(reference_df) == 50
    assert set(reference_df.columns) == {"amount", "hour", "is_fraud"}
