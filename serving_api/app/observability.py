from __future__ import annotations

import logging
from math import isfinite
from typing import Any

import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

from monitoring.evidently_ai.drift_monitor import DriftMonitor, DriftSnapshot

from .model_loader import PROCESSED_DIR

logger = logging.getLogger(__name__)

HTTP_REQUESTS_TOTAL = Counter(
    "tcb_http_requests_total",
    "Total HTTP requests handled by the FastAPI service.",
    ["method", "path", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "tcb_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
MODEL_LOADED = Gauge(
    "tcb_model_loaded",
    "Model load status. 1 means the fraud detector is ready.",
)
MODEL_THRESHOLD = Gauge(
    "tcb_model_threshold",
    "Decision threshold used by the currently loaded model.",
)
MODEL_FEATURE_COUNT = Gauge(
    "tcb_model_feature_count",
    "Number of model features used by the current detector.",
)
PREDICTION_REQUESTS_TOTAL = Counter(
    "tcb_prediction_requests_total",
    "Prediction API calls, split by endpoint.",
    ["endpoint"],
)
PREDICTION_SAMPLES_TOTAL = Counter(
    "tcb_prediction_samples_total",
    "Number of transactions scored by the API.",
    ["endpoint"],
)
PREDICTION_FRAUD_TOTAL = Counter(
    "tcb_prediction_fraud_total",
    "Number of transactions predicted as fraud.",
    ["endpoint"],
)
PREDICTION_RISK_LEVEL_TOTAL = Counter(
    "tcb_prediction_risk_level_total",
    "Prediction volume by risk level.",
    ["endpoint", "risk_level"],
)
PREDICTION_SCORE = Histogram(
    "tcb_prediction_score",
    "Distribution of fraud scores returned by the model.",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99),
)
PREDICTION_AMOUNT_VND = Histogram(
    "tcb_prediction_amount_vnd",
    "Distribution of transaction amounts scored by the API.",
    ["endpoint"],
    buckets=(
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        5_000_000,
        10_000_000,
        50_000_000,
    ),
)
DRIFT_BASELINE_READY = Gauge(
    "tcb_drift_baseline_ready",
    (
        "Drift baseline readiness. 1 means the monitor "
        "has a usable reference window."
    ),
)
DRIFT_REFERENCE_MODE = Gauge(
    "tcb_drift_reference_mode",
    "Active reference source for drift monitoring.",
    ["mode"],
)
DRIFT_REFERENCE_SAMPLES = Gauge(
    "tcb_drift_reference_samples",
    "Number of samples in the drift reference window.",
)
DRIFT_CURRENT_SAMPLES = Gauge(
    "tcb_drift_current_samples",
    "Number of samples in the live drift window.",
)
DRIFT_FEATURES_TOTAL = Gauge(
    "tcb_drift_features_total",
    "Number of features currently evaluated for drift.",
)
DRIFT_FEATURES_ALERTING = Gauge(
    "tcb_drift_features_alerting",
    "Number of monitored features above the drift alert threshold.",
)
DRIFT_RATIO = Gauge(
    "tcb_drift_ratio",
    "Share of monitored features currently alerting for drift.",
)
DRIFT_OVERALL_SCORE = Gauge(
    "tcb_drift_overall_score",
    "Mean drift score across monitored features.",
)
DRIFT_FEATURE_SCORE = Gauge(
    "tcb_drift_feature_score",
    "Per-feature drift score.",
    ["feature"],
)
DRIFT_FEATURE_ALERT = Gauge(
    "tcb_drift_feature_alert",
    (
        "Per-feature alert state. 1 means the feature "
        "is above the drift threshold."
    ),
    ["feature"],
)

_REFERENCE_MODES = ("train_parquet", "warmup_window", "missing")
_DRIFT_MONITOR = DriftMonitor(processed_dir=PROCESSED_DIR)


def bootstrap_observability(detector: Any | None) -> None:
    if detector is None:
        MODEL_LOADED.set(0)
        return

    MODEL_LOADED.set(1)

    threshold = getattr(detector, "_threshold", None)
    if isinstance(threshold, (float, int)):
        MODEL_THRESHOLD.set(float(threshold))

    feature_cols = getattr(detector, "_feature_cols", None)
    if isinstance(feature_cols, list):
        MODEL_FEATURE_COUNT.set(len(feature_cols))

    _apply_drift_snapshot(_DRIFT_MONITOR.bootstrap(detector))


def shutdown_observability() -> None:
    MODEL_LOADED.set(0)


def record_http_observation(
    method: str,
    path: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    labels = {
        "method": method,
        "path": path,
        "status": str(status_code),
    }
    HTTP_REQUESTS_TOTAL.labels(**labels).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method,
        path=path,
    ).observe(duration_seconds)


def record_prediction_observation(
    endpoint: str,
    raw_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    detector: Any,
) -> None:
    sample_count = len(predictions_df)
    PREDICTION_REQUESTS_TOTAL.labels(endpoint=endpoint).inc()
    PREDICTION_SAMPLES_TOTAL.labels(endpoint=endpoint).inc(sample_count)

    threshold = getattr(detector, "_threshold", None)
    if isinstance(threshold, (float, int)):
        MODEL_THRESHOLD.set(float(threshold))

    if "amount" in raw_df.columns:
        for amount in raw_df["amount"].dropna().tolist():
            amount_value = float(amount)
            if isfinite(amount_value) and amount_value > 0:
                PREDICTION_AMOUNT_VND.labels(
                    endpoint=endpoint
                ).observe(amount_value)

    if not predictions_df.empty:
        fraud_count = int(
            pd.to_numeric(
                predictions_df["is_fraud_pred"],
                errors="coerce",
            ).fillna(0).sum()
        )
        PREDICTION_FRAUD_TOTAL.labels(endpoint=endpoint).inc(fraud_count)

        for risk_level in (
            predictions_df["risk_level"]
            .fillna("UNKNOWN")
            .astype(str)
            .tolist()
        ):
            PREDICTION_RISK_LEVEL_TOTAL.labels(
                endpoint=endpoint,
                risk_level=risk_level,
            ).inc()

        for score in (
            pd.to_numeric(
                predictions_df["fraud_score"],
                errors="coerce",
            )
            .dropna()
            .tolist()
        ):
            score_value = float(score)
            if isfinite(score_value):
                PREDICTION_SCORE.labels(endpoint=endpoint).observe(score_value)

    _apply_drift_snapshot(
        _DRIFT_MONITOR.observe(raw_df=raw_df, detector=detector)
    )


def get_drift_snapshot() -> DriftSnapshot:
    return _DRIFT_MONITOR.snapshot()


def get_drift_alert_threshold() -> float:
    return _DRIFT_MONITOR.alert_threshold


def _apply_drift_snapshot(snapshot: DriftSnapshot) -> None:
    DRIFT_BASELINE_READY.set(1 if snapshot.ready else 0)
    DRIFT_REFERENCE_SAMPLES.set(snapshot.reference_samples)
    DRIFT_CURRENT_SAMPLES.set(snapshot.current_samples)
    DRIFT_FEATURES_TOTAL.set(snapshot.features_total)
    DRIFT_FEATURES_ALERTING.set(snapshot.features_alerting)
    DRIFT_RATIO.set(snapshot.drift_ratio)
    DRIFT_OVERALL_SCORE.set(snapshot.overall_score)

    for mode in _REFERENCE_MODES:
        DRIFT_REFERENCE_MODE.labels(mode=mode).set(
            1 if snapshot.reference_mode == mode else 0
        )

    DRIFT_FEATURE_SCORE.clear()
    DRIFT_FEATURE_ALERT.clear()
    for feature, score in snapshot.feature_scores.items():
        DRIFT_FEATURE_SCORE.labels(feature=feature).set(score)
        DRIFT_FEATURE_ALERT.labels(feature=feature).set(
            1 if score >= _DRIFT_MONITOR.alert_threshold else 0
        )
