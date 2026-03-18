from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from serving_api.app.rollout import CanaryRolloutManager


class FakeDetector:
    def __init__(self, models_dir: str, processed_dir: str) -> None:
        self.models_dir = models_dir
        self.processed_dir = processed_dir
        self._threshold = 0.91 if "candidate" in models_dir else 0.42
        self._model_version = "8" if "candidate" in models_dir else "7"

    def health_check(self) -> dict[str, object]:
        return {
            "status": "OK",
            "model_type": "XGBClassifier",
            "feature_count": 31,
            "threshold": self._threshold,
            "best_iteration": 100,
            "loaded_at": "2026-03-14T10:23:01+00:00",
            "model_version": self._model_version,
        }

    def predict_single(self, transaction: dict[str, object]) -> dict[str, object]:
        return {
            "transaction_id": str(transaction["transaction_id"]),
            "fraud_score": 0.95 if self._model_version == "8" else 0.05,
            "is_fraud_pred": self._model_version == "8",
            "threshold": self._threshold,
            "risk_level": "HIGH" if self._model_version == "8" else "LOW",
            "model_version": self._model_version,
            "prediction_timestamp": "2026-03-14T10:23:01+00:00",
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["fraud_score"] = 0.95 if self._model_version == "8" else 0.05
        result["is_fraud_pred"] = 1 if self._model_version == "8" else 0
        result["risk_level"] = "HIGH" if self._model_version == "8" else "LOW"
        result["model_version"] = self._model_version
        result["prediction_timestamp"] = "2026-03-14T10:23:01+00:00"
        return result


def _prepare_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    stable_models = tmp_path / "models"
    stable_processed = tmp_path / "processed"
    candidate_models = tmp_path / "candidate_models"
    candidate_processed = tmp_path / "candidate_processed"

    stable_models.mkdir()
    stable_processed.mkdir()
    candidate_models.mkdir()
    candidate_processed.mkdir()
    (candidate_models / "xgb_fraud_model.joblib").write_text("placeholder", encoding="utf-8")
    return stable_models, stable_processed, candidate_models


def test_update_rollout_loads_candidate_and_sets_first_step(tmp_path: Path):
    stable_models, stable_processed, candidate_models = _prepare_dirs(tmp_path)
    config_path = tmp_path / "rollout.json"

    with patch("serving_api.app.rollout.FraudDetector", side_effect=FakeDetector):
        manager = CanaryRolloutManager(
            stable_models_dir=str(stable_models),
            stable_processed_dir=str(stable_processed),
            config_path=str(config_path),
        )

        status = manager.update_rollout(
            {
                "enabled": True,
                "candidate_models_dir": str(candidate_models),
                "candidate_processed_dir": str(stable_processed),
                "rollout_steps": [10, 50, 100],
            }
        )

    assert status["enabled"] is True
    assert status["traffic_percent_candidate"] == 10
    assert status["candidate"]["loaded"] is True
    assert status["candidate"]["model_version"] == "8"


def test_pause_rollout_keeps_current_traffic_split(tmp_path: Path):
    stable_models, stable_processed, candidate_models = _prepare_dirs(tmp_path)
    config_path = tmp_path / "rollout.json"

    with patch("serving_api.app.rollout.FraudDetector", side_effect=FakeDetector):
        manager = CanaryRolloutManager(
            stable_models_dir=str(stable_models),
            stable_processed_dir=str(stable_processed),
            config_path=str(config_path),
        )
        manager.update_rollout(
            {
                "enabled": True,
                "candidate_models_dir": str(candidate_models),
                "candidate_processed_dir": str(stable_processed),
                "traffic_percent_candidate": 25,
                "rollout_steps": [10, 25, 50, 100],
                "auto_advance": True,
            }
        )
        status = manager.pause_rollout()

    assert status["enabled"] is True
    assert status["status"] == "paused"
    assert status["traffic_percent_candidate"] == 25
    assert status["auto_advance"] is False
    assert status["next_step_at"] is None


def test_route_batch_adds_threshold_and_lane_metadata(tmp_path: Path):
    stable_models, stable_processed, candidate_models = _prepare_dirs(tmp_path)
    config_path = tmp_path / "rollout.json"
    batch = pd.DataFrame(
        [
            {
                "transaction_id": "TX_001",
                "timestamp": "2026-03-14 10:23:00",
                "customer_id": "CUST_001",
                "amount": 1000,
                "customer_tier": "PRIORITY",
            }
        ]
    )

    with patch("serving_api.app.rollout.FraudDetector", side_effect=FakeDetector):
        manager = CanaryRolloutManager(
            stable_models_dir=str(stable_models),
            stable_processed_dir=str(stable_processed),
            config_path=str(config_path),
        )
        manager.update_rollout(
            {
                "enabled": True,
                "candidate_models_dir": str(candidate_models),
                "candidate_processed_dir": str(stable_processed),
                "traffic_percent_candidate": 100,
                "rollout_steps": [10, 50, 100],
            }
        )
        routing, result = manager.route_batch(batch)

    assert routing == {"stable": 0, "candidate": 1}
    assert float(result["threshold"].iloc[0]) == 0.91
    assert str(result["served_by"].iloc[0]) == "candidate"
    assert str(result["model_version"].iloc[0]) == "8"


def test_promote_candidate_replaces_stable_model(tmp_path: Path):
    stable_models, stable_processed, candidate_models = _prepare_dirs(tmp_path)
    config_path = tmp_path / "rollout.json"

    with patch("serving_api.app.rollout.FraudDetector", side_effect=FakeDetector):
        manager = CanaryRolloutManager(
            stable_models_dir=str(stable_models),
            stable_processed_dir=str(stable_processed),
            config_path=str(config_path),
        )
        manager.update_rollout(
            {
                "enabled": True,
                "candidate_models_dir": str(candidate_models),
                "candidate_processed_dir": str(stable_processed),
                "traffic_percent_candidate": 100,
                "rollout_steps": [10, 50, 100],
            }
        )
        status = manager.promote_candidate()

    assert status["status"] == "promoted"
    assert status["enabled"] is False
    assert status["stable"]["model_version"] == "8"
