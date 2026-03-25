from __future__ import annotations

import pandas as pd

from monitoring.evidently_ai.drift_monitor import DriftMonitor


def test_waiting_for_current_window_exposes_zero_feature_scores(tmp_path):
    monitor = DriftMonitor(processed_dir=str(tmp_path))
    monitor._reference_df = pd.DataFrame(
        {
            "amount": [100_000.0, 200_000.0, 150_000.0],
            "hour": [9.0, 10.0, 11.0],
        }
    )

    snapshot = monitor._build_snapshot(reason="waiting_for_current_window")

    assert snapshot.ready is False
    assert snapshot.features_total == 2
    assert snapshot.feature_scores == {
        "amount": 0.0,
        "hour": 0.0,
    }


def test_bootstrap_uses_drift_reference_artifact_when_available(tmp_path):
    reference_df = pd.DataFrame(
        {
            "amount": [100_000.0, 200_000.0, 150_000.0],
            "hour": [9.0, 10.0, 11.0],
            "is_fraud": [0, 1, 0],
        }
    )
    reference_df.to_parquet(tmp_path / "drift_reference.parquet", index=False)

    class DummyDetector:
        _feature_cols = ["amount", "hour"]

    monitor = DriftMonitor(processed_dir=str(tmp_path))
    snapshot = monitor.bootstrap(DummyDetector())

    assert snapshot.ready is False
    assert snapshot.reference_mode == "train_parquet"
    assert snapshot.reference_samples == 3
    assert snapshot.features_total == 2
    assert snapshot.feature_scores == {
        "amount": 0.0,
        "hour": 0.0,
    }
