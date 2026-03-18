"""
Tests for FastAPI endpoint behaviour.

Uses FastAPI TestClient with a mocked FraudDetector so tests run without
requiring actual model artifacts on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from serving_api.app.main import app, get_detector, get_rollout_manager


VALID_TX = {
    "transaction_id": "TX_TEST_001",
    "timestamp": "2026-03-14 10:23:00",
    "customer_id": "CUST_001",
    "amount": 350000,
    "customer_tier": "PRIORITY",
    "card_type": "VISA",
    "card_tier": "GOLD",
    "currency": "VND",
    "merchant_name": "Grab",
    "mcc_code": 4121,
    "merchant_category": "Transport",
    "merchant_city": "Ha Noi",
    "merchant_country": "VN",
    "device_type": "Mobile",
    "os": "iOS",
    "ip_country": "VN",
    "distance_from_home_km": 2.5,
    "cvv_match": "Y",
    "is_3d_secure": "Y",
    "transaction_status": "APPROVED",
    "tx_count_last_1h": 1,
    "tx_count_last_24h": 3,
    "time_since_last_tx_min": 120.0,
    "avg_amount_last_30d": 400000,
    "amount_ratio_vs_avg": 0.875,
    "is_new_device": 0,
    "is_new_merchant": 0,
    "card_bin": 411111,
    "account_age_days": 730,
    "is_weekend": 0,
    "hour_of_day": 10,
}

MOCK_SINGLE_RESULT = {
    "transaction_id": "TX_TEST_001",
    "fraud_score": 0.0231,
    "is_fraud_pred": False,
    "threshold": 0.6788,
    "risk_level": "LOW",
    "model_version": "7",
    "prediction_timestamp": "2026-03-14T10:23:01+00:00",
}


def make_mock_detector(
    single_result: dict | None = None,
    batch_df: pd.DataFrame | None = None,
) -> MagicMock:
    detector = MagicMock()
    detector.health_check.return_value = {
        "status": "OK",
        "model_type": "XGBClassifier",
        "feature_count": 31,
        "threshold": 0.6788,
        "best_iteration": 373,
        "loaded_at": "2026-03-14T12:00:00+00:00",
        "model_version": "7",
    }
    detector.predict_single.return_value = single_result or MOCK_SINGLE_RESULT
    detector._threshold = 0.6788

    if batch_df is None:
        batch_df = pd.DataFrame(
            [
                {
                    "transaction_id": "TX_TEST_001",
                    "fraud_score": 0.0231,
                    "is_fraud_pred": 0,
                    "risk_level": "LOW",
                    "model_version": "7",
                    "prediction_timestamp": "2026-03-14T10:23:01+00:00",
                }
            ]
        )
    detector.predict_batch.return_value = batch_df
    return detector


class MockRolloutManager:
    def __init__(
        self,
        detector: MagicMock,
        single_lane: str = "stable",
        batch_lane: str = "stable",
        traffic_percent_candidate: int = 0,
    ) -> None:
        self._detector = detector
        self._single_lane = single_lane
        self._batch_lane = batch_lane
        self._traffic_percent_candidate = traffic_percent_candidate
        self._status = {
            "config_path": "artifacts/rollout/rollout_config.json",
            "enabled": traffic_percent_candidate > 0,
            "status": "running" if traffic_percent_candidate > 0 else "idle",
            "traffic_percent_candidate": traffic_percent_candidate,
            "rollout_steps": [10, 25, 50, 100],
            "current_step_index": 0 if traffic_percent_candidate > 0 else -1,
            "step_interval_minutes": 30,
            "auto_advance": False,
            "auto_promote_when_complete": False,
            "last_transition_at": "2026-03-14T10:00:00+00:00",
            "next_step_at": None,
            "stable": {
                "models_dir": "models",
                "processed_dir": "data/processed",
                "model_version": "7",
                "loaded": True,
            },
            "candidate": {
                "models_dir": "models/candidate",
                "processed_dir": "data/processed",
                "model_version": "8",
                "loaded": traffic_percent_candidate > 0,
            } if traffic_percent_candidate > 0 else None,
        }

    def get_stable_detector(self) -> MagicMock:
        return self._detector

    def get_rollout_status(self) -> dict:
        return self._status

    def route_single(self, transaction: dict) -> tuple[str, dict]:
        result = dict(self._detector.predict_single(transaction))
        if self._single_lane == "candidate":
            result["model_version"] = "8"
        return self._single_lane, result

    def route_batch(self, df: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
        result = self._detector.predict_batch(df)
        result = result.copy()
        result["threshold"] = 0.6788
        result["served_by"] = self._batch_lane
        if self._batch_lane == "candidate":
            result["model_version"] = "8"
        routing = {
            "stable": 0 if self._batch_lane == "candidate" else len(result),
            "candidate": len(result) if self._batch_lane == "candidate" else 0,
        }
        return routing, result

    def update_rollout(self, payload: dict) -> dict:
        self._status.update(payload)
        return self._status

    def advance_rollout(self) -> dict:
        self._status["traffic_percent_candidate"] = 25
        return self._status

    def pause_rollout(self) -> dict:
        self._status["status"] = "paused"
        return self._status

    def promote_candidate(self) -> dict:
        self._status["status"] = "promoted"
        self._status["enabled"] = False
        self._status["traffic_percent_candidate"] = 0
        return self._status

    def rollback_rollout(self) -> dict:
        self._status["status"] = "rolled_back"
        self._status["enabled"] = False
        self._status["traffic_percent_candidate"] = 0
        return self._status


@pytest.fixture()
def client():
    mock_detector = make_mock_detector()
    mock_rollout_manager = MockRolloutManager(mock_detector)
    app.dependency_overrides[get_detector] = lambda: mock_detector
    app.dependency_overrides[get_rollout_manager] = lambda: mock_rollout_manager

    with patch("serving_api.app.main.load_model"), patch("serving_api.app.main.unload_model"):
        with TestClient(app, raise_server_exceptions=True) as client_instance:
            yield client_instance, mock_detector, mock_rollout_manager

    app.dependency_overrides.clear()


class TestRoot:
    def test_returns_200(self, client):
        c, _, _ = client
        assert c.get("/").status_code == 200

    def test_contains_version(self, client):
        c, _, _ = client
        data = c.get("/").json()
        assert "version" in data

    def test_contains_endpoints(self, client):
        c, _, _ = client
        data = c.get("/").json()
        for field in ("predict", "health", "metrics", "ready", "live"):
            assert field in data


class TestHealthAndProbes:
    def test_health_returns_200(self, client):
        c, _, _ = client
        assert c.get("/health").status_code == 200

    def test_ready_returns_200(self, client):
        c, _, _ = client
        assert c.get("/ready").status_code == 200

    def test_live_returns_200(self, client):
        c, _, _ = client
        assert c.get("/live").status_code == 200

    def test_metrics_returns_prometheus_payload(self, client):
        c, _, _ = client
        resp = c.get("/metrics")
        assert resp.status_code == 200
        assert "fraud_api_requests_total" in resp.text


class TestPredict:
    def test_returns_200_for_valid_input(self, client):
        c, _, _ = client
        assert c.post("/predict", json=VALID_TX).status_code == 200

    def test_response_schema(self, client):
        c, _, _ = client
        data = c.post("/predict", json=VALID_TX).json()
        for field in (
            "transaction_id",
            "fraud_score",
            "is_fraud_pred",
            "threshold",
            "risk_level",
            "model_version",
            "prediction_timestamp",
            "served_by",
            "rollout_candidate_percent",
        ):
            assert field in data, f"Missing field: {field}"

    def test_fraud_score_in_range(self, client):
        c, _, _ = client
        data = c.post("/predict", json=VALID_TX).json()
        assert 0.0 <= data["fraud_score"] <= 1.0

    def test_calls_detector_once(self, client):
        c, mock, _ = client
        c.post("/predict", json=VALID_TX)
        mock.predict_single.assert_called_once()

    def test_invalid_customer_tier_returns_422(self, client):
        c, _, _ = client
        bad_tx = {**VALID_TX, "customer_tier": "UNKNOWN_TIER"}
        assert c.post("/predict", json=bad_tx).status_code == 422

    def test_process_time_header_present(self, client):
        c, _, _ = client
        resp = c.post("/predict", json=VALID_TX)
        assert "x-process-time-ms" in resp.headers


class TestPredictBatch:
    def test_returns_200_for_valid_batch(self, client):
        c, _, _ = client
        assert c.post("/predict/batch", json={"transactions": [VALID_TX]}).status_code == 200

    def test_response_schema(self, client):
        c, _, _ = client
        data = c.post("/predict/batch", json={"transactions": [VALID_TX]}).json()
        for field in (
            "total",
            "fraud_detected",
            "fraud_rate",
            "threshold",
            "model_version",
            "prediction_timestamp",
            "rollout_candidate_percent",
            "traffic_distribution",
            "predictions",
        ):
            assert field in data, f"Missing field: {field}"

    def test_empty_batch_returns_422(self, client):
        c, _, _ = client
        assert c.post("/predict/batch", json={"transactions": []}).status_code == 422

    def test_predictions_list_not_empty(self, client):
        c, _, _ = client
        data = c.post("/predict/batch", json={"transactions": [VALID_TX]}).json()
        assert len(data["predictions"]) > 0

    def test_batch_item_contains_lane_and_model_version(self, client):
        c, _, _ = client
        data = c.post("/predict/batch", json={"transactions": [VALID_TX]}).json()
        item = data["predictions"][0]
        for field in ("threshold", "served_by", "model_version", "prediction_timestamp"):
            assert field in item


class TestRolloutAdmin:
    def test_status_endpoint_returns_200(self, client):
        c, _, _ = client
        resp = c.get("/admin/rollout")
        assert resp.status_code == 200
        assert "traffic_percent_candidate" in resp.json()

    def test_ui_endpoint_returns_html(self, client):
        c, _, _ = client
        resp = c.get("/admin/rollout/ui")
        assert resp.status_code == 200
        assert "Canary Rollout Control" in resp.text

    def test_config_endpoint_updates_state(self, client):
        c, _, manager = client
        payload = {
            "enabled": True,
            "auto_advance": True,
            "candidate_models_dir": "models/candidate",
            "candidate_processed_dir": "data/processed",
            "rollout_steps": [10, 50, 100],
            "step_interval_minutes": 15,
        }
        resp = c.post("/admin/rollout/config", json=payload)
        assert resp.status_code == 200
        assert manager.get_rollout_status()["enabled"] is True
        assert manager.get_rollout_status()["step_interval_minutes"] == 15

    def test_advance_endpoint_returns_new_status(self, client):
        c, _, _ = client
        resp = c.post("/admin/rollout/advance")
        assert resp.status_code == 200
        assert resp.json()["traffic_percent_candidate"] == 25
