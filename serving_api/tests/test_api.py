"""
Tests for main.py — FastAPI endpoint behaviour.

Uses FastAPI TestClient with a mocked FraudDetector so tests run
without requiring actual model artifacts on disk.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from serving_api.app import main as main_module

app = main_module.app


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_TX = {
    "transaction_id":         "TX_TEST_001",
    "timestamp":              "2026-03-14 10:23:00",
    "customer_id":            "CUST_001",
    "amount":                 350000,
    "customer_tier":          "PRIORITY",
    "card_type":              "VISA",
    "card_tier":              "GOLD",
    "currency":               "VND",
    "merchant_name":          "Grab",
    "mcc_code":               4121,
    "merchant_category":      "Transport",
    "merchant_city":          "Ha Noi",
    "merchant_country":       "VN",
    "device_type":            "Mobile",
    "os":                     "iOS",
    "ip_country":             "VN",
    "distance_from_home_km":  2.5,
    "cvv_match":              "Y",
    "is_3d_secure":           "Y",
    "transaction_status":     "APPROVED",
    "tx_count_last_1h":       1,
    "tx_count_last_24h":      3,
    "time_since_last_tx_min": 120.0,
    "avg_amount_last_30d":    400000,
    "amount_ratio_vs_avg":    0.875,
    "is_new_device":          0,
    "is_new_merchant":        0,
    "card_bin":               411111,
    "account_age_days":       730,
    "is_weekend":             0,
    "hour_of_day":            10,
}

MOCK_SINGLE_RESULT = {
    "transaction_id": "TX_TEST_001",
    "fraud_score":    0.0231,
    "is_fraud_pred":  False,
    "threshold":      0.6788,
    "risk_level":     "LOW",
}


def make_mock_detector(
    single_result: dict | None = None,
    batch_df=None,
) -> MagicMock:
    """Build a MagicMock FraudDetector with sensible defaults."""
    import pandas as pd

    detector = MagicMock()
    detector.health_check.return_value = {
        "status":         "OK",
        "model_type":     "XGBClassifier",
        "feature_count":  31,
        "threshold":      0.6788,
        "best_iteration": 373,
        "loaded_at":      "2026-03-14T12:00:00+00:00",
    }
    detector.predict_single.return_value = single_result or MOCK_SINGLE_RESULT
    detector._threshold = 0.6788
    detector._feature_cols = ["amount", "transaction_hour", "segment_encoded"]

    if batch_df is None:
        batch_df = pd.DataFrame([{
            "transaction_id": "TX_TEST_001",
            "fraud_score":    0.0231,
            "is_fraud_pred":  0,
            "risk_level":     "LOW",
        }])
    detector.predict_batch.return_value = batch_df

    def fake_transform(df: pd.DataFrame) -> pd.DataFrame:
        timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
        return pd.DataFrame({
            "amount": pd.to_numeric(df["amount"], errors="coerce").fillna(0.0),
            "transaction_hour": timestamp.dt.hour.fillna(0).astype(float),
            "segment_encoded": 1.0,
        })

    detector._transform.side_effect = fake_transform

    return detector


@pytest.fixture()
def client():
    """TestClient with mocked FraudDetector injected via dependency override."""
    mock_detector = make_mock_detector()
    app.dependency_overrides[main_module.get_detector] = lambda: mock_detector
    app.dependency_overrides[main_module.get_optional_detector] = (
        lambda: mock_detector
    )

    with patch.object(main_module, "load_model"), patch.object(main_module, "unload_model"):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_detector

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
class TestRoot:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/")
        assert resp.status_code == 200

    def test_contains_version(self, client):
        c, _ = client
        data = resp = c.get("/").json()
        assert "version" in data

    def test_contains_endpoints(self, client):
        c, _ = client
        data = c.get("/").json()
        assert "predict" in data
        assert "health" in data


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["status"] == "OK"

    def test_response_schema(self, client):
        c, _ = client
        data = c.get("/health").json()
        for field in ("status", "model_type", "feature_count", "threshold",
                      "best_iteration", "loaded_at", "api_version",
                      "model_slot", "model_loaded"):
            assert field in data, f"Missing field: {field}"

    def test_feature_count_positive(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["feature_count"] > 0

    def test_threshold_in_range(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert 0 < data["threshold"] < 1


class TestDeployment:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/deployment")
        assert resp.status_code == 200

    def test_contains_runtime_fields(self, client):
        c, _ = client
        data = c.get("/deployment").json()
        for field in ("model_slot", "model_loaded", "models_dir", "processed_dir"):
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_returns_200_for_valid_input(self, client):
        c, _ = client
        resp = c.post("/predict", json=VALID_TX)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        c, _ = client
        data = c.post("/predict", json=VALID_TX).json()
        for field in ("transaction_id", "fraud_score", "is_fraud_pred",
                      "threshold", "risk_level"):
            assert field in data, f"Missing field: {field}"

    def test_fraud_score_in_range(self, client):
        c, _ = client
        data = c.post("/predict", json=VALID_TX).json()
        assert 0.0 <= data["fraud_score"] <= 1.0

    def test_risk_level_valid_values(self, client):
        c, _ = client
        data = c.post("/predict", json=VALID_TX).json()
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_missing_required_field_returns_422(self, client):
        c, _ = client
        bad_tx = {k: v for k, v in VALID_TX.items() if k != "amount"}
        resp = c.post("/predict", json=bad_tx)
        assert resp.status_code == 422

    def test_invalid_customer_tier_returns_422(self, client):
        c, _ = client
        bad_tx = {**VALID_TX, "customer_tier": "UNKNOWN_TIER"}
        resp = c.post("/predict", json=bad_tx)
        assert resp.status_code == 422

    def test_negative_amount_returns_422(self, client):
        c, _ = client
        bad_tx = {**VALID_TX, "amount": -100}
        resp = c.post("/predict", json=bad_tx)
        assert resp.status_code == 422

    def test_invalid_timestamp_returns_422(self, client):
        c, _ = client
        bad_tx = {**VALID_TX, "timestamp": "not-a-date"}
        resp = c.post("/predict", json=bad_tx)
        assert resp.status_code == 422

    def test_calls_detector_once(self, client):
        c, mock = client
        c.post("/predict", json=VALID_TX)
        mock.predict_single.assert_called_once()

    def test_transaction_id_echoed(self, client):
        c, _ = client
        data = c.post("/predict", json=VALID_TX).json()
        assert data["transaction_id"] == VALID_TX["transaction_id"]

    def test_process_time_header_present(self, client):
        c, _ = client
        resp = c.post("/predict", json=VALID_TX)
        assert "x-process-time-ms" in resp.headers


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------
class TestPredictBatch:
    def test_returns_200_for_valid_batch(self, client):
        c, _ = client
        resp = c.post("/predict/batch", json={"transactions": [VALID_TX]})
        assert resp.status_code == 200

    def test_response_schema(self, client):
        c, _ = client
        data = c.post("/predict/batch", json={"transactions": [VALID_TX]}).json()
        for field in ("total", "fraud_detected", "fraud_rate",
                      "threshold", "predictions"):
            assert field in data, f"Missing field: {field}"

    def test_empty_batch_returns_422(self, client):
        c, _ = client
        resp = c.post("/predict/batch", json={"transactions": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_exports_custom_prediction_metrics(self, client):
        c, _ = client
        c.post("/predict", json=VALID_TX)

        resp = c.get("/metrics")

        assert resp.status_code == 200
        assert "tcb_prediction_requests_total" in resp.text
        assert "tcb_prediction_score_bucket" in resp.text

    def test_exports_custom_drift_metrics(self, client):
        c, _ = client
        c.post("/predict/batch", json={"transactions": [VALID_TX]})

        resp = c.get("/metrics")

        assert resp.status_code == 200
        assert "tcb_drift_baseline_ready" in resp.text
        assert "tcb_drift_feature_score" in resp.text

    def test_total_matches_input_count(self, client):
        import pandas as pd
        c, mock = client
        # Return batch df matching 1 transaction
        mock.predict_batch.return_value = pd.DataFrame([{
            "transaction_id": "TX_TEST_001",
            "fraud_score":    0.05,
            "is_fraud_pred":  0,
            "risk_level":     "LOW",
        }])
        data = c.post("/predict/batch", json={"transactions": [VALID_TX]}).json()
        assert data["total"] == 1

    def test_fraud_rate_between_0_and_1(self, client):
        c, _ = client
        data = c.post("/predict/batch", json={"transactions": [VALID_TX]}).json()
        assert 0.0 <= data["fraud_rate"] <= 1.0

    def test_predictions_list_not_empty(self, client):
        c, _ = client
        data = c.post("/predict/batch", json={"transactions": [VALID_TX]}).json()
        assert len(data["predictions"]) > 0

    def test_prediction_item_schema(self, client):
        c, _ = client
        data = c.post("/predict/batch", json={"transactions": [VALID_TX]}).json()
        item = data["predictions"][0]
        for field in ("transaction_id", "fraud_score", "is_fraud_pred", "risk_level"):
            assert field in item, f"Missing field in prediction item: {field}"
