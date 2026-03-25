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
