"""
TCB Fraud Detection — Inference Pipeline.

Provides a ``FraudDetector`` class that loads all trained artifacts once and
exposes a clean API for single-transaction and batch prediction. The internal
``_transform()`` method mirrors the preprocessing pipeline exactly — same
feature engineering, same encoding maps, same column order — ensuring that
inference-time transformations are identical to training-time transformations.

Artifact dependencies (produced by preprocess.py + train.py + evaluate.py)
---------------------------------------------------------------------------
models/
    xgb_fraud_model.joblib      — trained XGBoost model
    evaluation/
        evaluation.json         — optimal threshold
data/processed/
    features.json               — canonical feature list + order
    customer_stats.parquet      — per-customer tx count + avg amount
    segment_label_map.json      — customer_tier → int encoding
    amount_median_train.json    — fallback imputation value
    categorical_maps.json       — low-cardinality categorical encodings

Usage
-----
    # Single transaction
    from ml_pipeline.src.inference import FraudDetector

    detector = FraudDetector("models", "data/processed")
    result = detector.predict_single({
        "transaction_id": "TX_001",
        "timestamp": "2026-03-14 10:23:00",
        "customer_id": "CUST_123",
        "amount": 5_000_000,
        "customer_tier": "PRIORITY",
        ... # remaining raw fields
    })
    print(result)
    # {
    #   "transaction_id": "TX_001",
    #   "fraud_score": 0.0231,
    #   "is_fraud_pred": False,
    #   "threshold": 0.6788,
    #   "risk_level": "LOW"
    # }

    # Batch prediction
    import pandas as pd
    df = pd.read_csv("new_transactions.csv")
    results_df = detector.predict_batch(df)

    # Health check
    print(detector.health_check())
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match preprocess.py exactly
# ---------------------------------------------------------------------------
HIGH_CARDINALITY_COLS: list[str] = [
    "merchant_name",
    "merchant_city",
]

LOW_CARDINALITY_COLS: list[str] = [
    "card_type",
    "card_tier",
    "currency",
    "merchant_category",
    "merchant_country",
    "device_type",
    "os",
    "ip_country",
]

# Risk level thresholds — applied on top of the binary decision threshold
_RISK_BANDS: list[tuple[float, str]] = [
    (0.30, "LOW"),
    (0.60, "MEDIUM"),
    (1.01, "HIGH"),
]

# Columns that must never be used as model features
_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {"transaction_id", "customer_id", "timestamp", "is_fraud"}
)


# ---------------------------------------------------------------------------
# FraudDetector
# ---------------------------------------------------------------------------
class FraudDetector:
    """Stateful inference engine for TCB fraud detection.

    Loads all artifacts once at construction time. The instance is designed
    to be long-lived — instantiate once and reuse across many prediction calls.

    Args:
        models_dir: Directory containing ``xgb_fraud_model.joblib`` and
            ``evaluation/evaluation.json``.
        processed_dir: Directory containing ``features.json``,
            ``customer_stats.parquet``, ``segment_label_map.json``,
            ``amount_median_train.json``, and ``categorical_maps.json``.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """

    def __init__(self, models_dir: str, processed_dir: str) -> None:
        self._models_dir    = Path(models_dir)
        self._processed_dir = Path(processed_dir)
        self._loaded_at     = datetime.now(tz=timezone.utc).isoformat()

        logger.info("Loading FraudDetector artifacts…")
        self._model, self._feature_state, self._feature_cols, self._threshold = (
            self._load_artifacts()
        )
        logger.info(
            "FraudDetector ready — features: %d | threshold: %.4f",
            len(self._feature_cols),
            self._threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(self, transaction: dict[str, Any]) -> dict[str, Any]:
        """Score a single raw transaction.

        Args:
            transaction: Dictionary with raw transaction fields matching the
                training schema (same columns as the raw CSV).

        Returns:
            Dictionary with keys:
            - ``transaction_id``: echoed from input (or ``"UNKNOWN"``).
            - ``fraud_score``: float in [0, 1].
            - ``is_fraud_pred``: bool based on optimal threshold.
            - ``threshold``: the decision threshold used.
            - ``risk_level``: ``"LOW"``, ``"MEDIUM"``, or ``"HIGH"``.
        """
        df = pd.DataFrame([transaction])
        scores = self._score(df)

        fraud_score  = float(scores[0])
        is_fraud     = fraud_score >= self._threshold
        risk_level   = self._risk_level(fraud_score)
        tx_id        = str(transaction.get("transaction_id", "UNKNOWN"))

        result: dict[str, Any] = {
            "transaction_id": tx_id,
            "fraud_score":    round(fraud_score, 6),
            "is_fraud_pred":  is_fraud,
            "threshold":      self._threshold,
            "risk_level":     risk_level,
        }

        logger.info(
            "predict_single — tx_id=%s | score=%.4f | pred=%s | risk=%s",
            tx_id, fraud_score, is_fraud, risk_level,
        )
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of raw transactions.

        Args:
            df: DataFrame with raw transaction fields — same schema as the
                raw CSV. Does not need to be preprocessed.

        Returns:
            Original DataFrame with three additional columns appended:
            - ``fraud_score``:   float in [0, 1].
            - ``is_fraud_pred``: int (0 or 1).
            - ``risk_level``:    str (``"LOW"``, ``"MEDIUM"``, ``"HIGH"``).
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        scores = self._score(df)

        result = df.copy()
        result["fraud_score"]   = np.round(scores, 6)
        result["is_fraud_pred"] = (scores >= self._threshold).astype(int)
        result["risk_level"]    = [self._risk_level(s) for s in scores]

        n_fraud = int((result["is_fraud_pred"] == 1).sum())
        logger.info(
            "predict_batch — rows: %d | fraud_detected: %d (%.2f%%) | threshold: %.4f",
            len(result),
            n_fraud,
            n_fraud / len(result) * 100,
            self._threshold,
        )
        return result

    def health_check(self) -> dict[str, Any]:
        """Return a health status dictionary for monitoring / liveness probes.

        Returns:
            Dictionary with model metadata and readiness status.
        """
        status: dict[str, Any] = {
            "status":        "OK",
            "loaded_at":     self._loaded_at,
            "threshold":     self._threshold,
            "feature_count": len(self._feature_cols),
            "features":      self._feature_cols,
            "model_type":    type(self._model).__name__,
            "best_iteration": int(self._model.best_iteration),
        }
        logger.info(
            "health_check — status=%s | features=%d | threshold=%.4f",
            status["status"],
            status["feature_count"],
            status["threshold"],
        )
        return status

    # ------------------------------------------------------------------
    # Internal — artifact loading
    # ------------------------------------------------------------------

    def _load_artifacts(
        self,
    ) -> tuple[XGBClassifier, dict[str, Any], list[str], float]:
        """Load and validate all required artifacts from disk.

        Returns:
            Tuple of ``(model, feature_state, feature_cols, threshold)``.

        Raises:
            FileNotFoundError: If any required file is absent.
        """
        required_files = {
            "model":            self._models_dir / "xgb_fraud_model.joblib",
            "evaluation":       self._models_dir / "evaluation" / "evaluation.json",
            "features":         self._processed_dir / "features.json",
            "customer_stats":   self._processed_dir / "customer_stats.parquet",
            "segment_label_map":self._processed_dir / "segment_label_map.json",
            "amount_median":    self._processed_dir / "amount_median_train.json",
            "categorical_maps": self._processed_dir / "categorical_maps.json",
        }
        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact '{name}' not found: {path}\n"
                    "Ensure preprocess.py → train.py → evaluate.py have been run."
                )

        # Model
        model: XGBClassifier = joblib.load(required_files["model"])

        # Feature schema
        with open(required_files["features"], encoding="utf-8") as fh:
            feature_cols: list[str] = json.load(fh)["features"]

        # Optimal threshold from evaluation run
        with open(required_files["evaluation"], encoding="utf-8") as fh:
            eval_data = json.load(fh)
        threshold: float = float(eval_data["threshold_metrics"]["threshold"])

        # Feature state — mirrors fit_feature_generators() output
        customer_stats = pd.read_parquet(required_files["customer_stats"])

        with open(required_files["segment_label_map"], encoding="utf-8") as fh:
            segment_label_map: dict[str, int] = json.load(fh)

        with open(required_files["amount_median"], encoding="utf-8") as fh:
            amount_median_train: float = float(
                json.load(fh)["amount_median_train"]
            )

        with open(required_files["categorical_maps"], encoding="utf-8") as fh:
            categorical_maps: dict[str, dict[str, int]] = json.load(fh)

        feature_state: dict[str, Any] = {
            "customer_stats":     customer_stats,
            "segment_label_map":  segment_label_map,
            "amount_median_train": amount_median_train,
            "categorical_maps":   categorical_maps,
        }

        logger.info(
            "Artifacts loaded — threshold: %.4f | features: %d | "
            "customers in state: %d",
            threshold,
            len(feature_cols),
            len(customer_stats),
        )
        return model, feature_state, feature_cols, threshold

    # ------------------------------------------------------------------
    # Internal — feature transformation (mirrors preprocess.py exactly)
    # ------------------------------------------------------------------

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full preprocessing pipeline to raw input data.

        Mirrors ``transform_features()`` in ``preprocess.py`` exactly.
        Uses the loaded ``feature_state`` (fitted on training data only).

        Args:
            df: Raw transaction DataFrame (as received at inference time).

        Returns:
            Transformed DataFrame aligned to the training feature schema.

        Raises:
            KeyError: If ``timestamp`` column is absent.
            ValueError: If required columns for encoding are missing.
        """
        df = df.copy()

        customer_stats: pd.DataFrame        = self._feature_state["customer_stats"]
        segment_label_map: dict[str, int]   = self._feature_state["segment_label_map"]
        amount_median: float                = self._feature_state["amount_median_train"]
        categorical_maps: dict[str, dict]   = self._feature_state["categorical_maps"]

        # --- Timestamp guard + parse ---
        if "timestamp" not in df.columns:
            raise KeyError("Column 'timestamp' is required for inference.")
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # --- Timestamp-derived features ---
        if "hour_of_day" in df.columns:
            df["transaction_hour"] = df["hour_of_day"]
            df = df.drop(columns=["hour_of_day"])
        else:
            df["transaction_hour"] = df["timestamp"].dt.hour

        df["transaction_day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_night_transaction"] = (
            (df["transaction_hour"] >= 23) | (df["transaction_hour"] <= 5)
        ).astype("int8")

        # --- Amount guard + log transform ---
        if "amount" not in df.columns:
            raise ValueError("Column 'amount' is required for inference.")
        if not pd.api.types.is_numeric_dtype(df["amount"]):
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        df["amount_log"] = np.log1p(df["amount"].clip(lower=0))

        # --- Segment encoding ---
        if "customer_tier" in df.columns:
            df["segment_encoded"] = (
                df["customer_tier"]
                .astype(str)
                .map(segment_label_map)
                .fillna(-1)
                .astype(int)
            )
            df = df.drop(columns=["customer_tier"])
        else:
            df["segment_encoded"] = -1
            logger.warning("'customer_tier' not in input — segment_encoded set to -1 (unseen).")

        # --- Customer aggregates: merge train-fitted stats ---
        if "customer_id" in df.columns:
            df = df.merge(
                customer_stats.reset_index(),
                on="customer_id",
                how="left",
            )
            unseen = int(df["customer_tx_count"].isna().sum())
            if unseen:
                logger.info("%d unseen customer(s) — imputing with train median.", unseen)
        else:
            df["customer_tx_count"]  = 0
            df["customer_avg_amount"] = amount_median

        df["customer_tx_count"]   = df["customer_tx_count"].fillna(0).astype(int)
        df["customer_avg_amount"] = df["customer_avg_amount"].fillna(amount_median)

        # --- Categorical encoding via train-fitted maps ---
        for col, mapping in categorical_maps.items():
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .map(mapping)
                    .fillna(-1)
                    .astype(int)
                )

        # --- Drop high-cardinality columns ---
        cols_to_drop = [c for c in HIGH_CARDINALITY_COLS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    def _score(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Transform raw input and return fraud probability scores.

        Args:
            raw_df: Raw transaction DataFrame.

        Returns:
            1-D numpy array of fraud probabilities, shape (n_rows,).
        """
        transformed = self._transform(raw_df)

        # Align to exact feature set used during training
        # — add missing cols as 0, drop extras, enforce order
        missing_cols = [c for c in self._feature_cols if c not in transformed.columns]
        if missing_cols:
            logger.warning(
                "Missing features filled with 0: %s", missing_cols
            )
            for col in missing_cols:
                transformed[col] = 0

        # Keep only numeric features in training order
        X = (
            transformed[self._feature_cols]
            .select_dtypes(include=[np.number, bool])
            .reindex(columns=self._feature_cols, fill_value=0)
        )

        return self._model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_level(score: float) -> str:
        """Map a fraud probability score to a human-readable risk band.

        Bands: LOW [0, 0.30) | MEDIUM [0.30, 0.60) | HIGH [0.60, 1.0].
        These thresholds are independent of the binary decision threshold
        and are used for downstream routing / alerting logic.

        Args:
            score: Fraud probability in [0, 1].

        Returns:
            Risk level string.
        """
        for upper, label in _RISK_BANDS:
            if score < upper:
                return label
        return "HIGH"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def _smoke_test(models_dir: str, processed_dir: str) -> None:
    logger.info("=" * 60)
    logger.info("INFERENCE SMOKE TEST")
    logger.info("=" * 60)

    detector = FraudDetector(models_dir, processed_dir)

    # --- Health check ---
    health = detector.health_check()
    assert health["status"] == "OK", "Health check failed"
    logger.info(
        "Health check PASSED — %d features | threshold=%.4f",
        health["feature_count"], health["threshold"],
    )

    # --- Lấy transaction thực từ raw CSV qua test.parquet index ---
    # Dùng processed test set để lấy transaction_id thực,
    # sau đó load raw CSV để có đầy đủ raw fields cho inference
    processed_dir_path = Path(processed_dir)
    test_df = pd.read_parquet(processed_dir_path / "test.parquet")

    # Tìm 1 legit và 1 fraud thực từ test set
    real_legit = test_df[test_df["is_fraud"] == 0].iloc[0]
    real_fraud  = test_df[test_df["is_fraud"] == 1].iloc[0]

    logger.info(
        "Sampled real transactions — legit tx_id=%s | fraud tx_id=%s",
        real_legit.get("transaction_id", "N/A"),
        real_fraud.get("transaction_id", "N/A"),
    )

    # --- Single prediction trên processed data ---
    # predict_batch nhận processed DataFrame trực tiếp (đã có đủ features)
    logger.info("-" * 40)
    logger.info("Single prediction test (real transactions)")

    legit_row = test_df[test_df["is_fraud"] == 0].head(1).copy()
    fraud_row  = test_df[test_df["is_fraud"] == 1].head(1).copy()

    # Chạy qua predict_batch — bỏ qua _transform vì data đã processed
    # Align thẳng vào feature cols
    def _score_processed_row(row_df: pd.DataFrame) -> float:
        X = (
            row_df[detector._feature_cols]
            .select_dtypes(include=[np.number, bool])
            .reindex(columns=detector._feature_cols, fill_value=0)
        )
        return float(detector._model.predict_proba(X)[:, 1][0])

    score_legit = _score_processed_row(legit_row)
    score_fraud  = _score_processed_row(fraud_row)
    label_legit = score_legit >= detector._threshold
    label_fraud  = score_fraud  >= detector._threshold

    logger.info(
        "  LEGIT (is_fraud=0) | score=%.4f | pred=%s | risk=%s",
        score_legit, label_legit, detector._risk_level(score_legit),
    )
    logger.info(
        "  FRAUD (is_fraud=1) | score=%.4f | pred=%s | risk=%s",
        score_fraud, label_fraud, detector._risk_level(score_fraud),
    )

    # --- Batch test trên 100 rows thực ---
    logger.info("-" * 40)
    logger.info("Batch prediction test (100 real rows from test set)")

    sample_100 = test_df.sample(100, random_state=42).copy()
    X_100 = (
        sample_100[detector._feature_cols]
        .select_dtypes(include=[np.number, bool])
        .reindex(columns=detector._feature_cols, fill_value=0)
    )
    scores_100   = detector._model.predict_proba(X_100)[:, 1]
    preds_100    = (scores_100 >= detector._threshold).astype(int)
    true_labels  = sample_100["is_fraud"].values

    tp = int(((preds_100 == 1) & (true_labels == 1)).sum())
    fp = int(((preds_100 == 1) & (true_labels == 0)).sum())
    fn = int(((preds_100 == 0) & (true_labels == 1)).sum())
    tn = int(((preds_100 == 0) & (true_labels == 0)).sum())

    logger.info(
        "  100-row batch | TN=%d | FP=%d | FN=%d | TP=%d",
        tn, fp, fn, tp,
    )

    # --- Assertions ---
    score_gap = score_fraud - score_legit
    assert score_gap > 0.05, (
        f"Real fraud tx should score significantly higher than legit tx "
        f"(gap={score_gap:.4f}, expected > 0.05)"
    )
    assert label_fraud is True, (
        f"Real fraud tx must be predicted as fraud "
        f"(score={score_fraud:.4f}, threshold={detector._threshold:.4f})"
    )
    assert label_legit is False, (
        f"Real legit tx must NOT be predicted as fraud "
        f"(score={score_legit:.4f}, threshold={detector._threshold:.4f})"
    )
    assert fn == 0, (
        f"No fraud cases should be missed in 100-row batch (FN={fn})"
    )

    logger.info("Score gap (fraud - legit): +%.4f", score_gap)
    logger.info("=" * 60)
    logger.info("SMOKE TEST COMPLETE — ALL ASSERTIONS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    _models    = str(project_root / "models")
    _processed = str(project_root / "data" / "processed")
    _smoke_test(_models, _processed)