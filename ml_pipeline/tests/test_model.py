from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier


# ===========================================================================
# Shared synthetic data helpers
# ===========================================================================

def _make_feature_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(5, int(n * 0.05))
    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    rng.shuffle(labels)

    timestamps = pd.date_range("2026-01-01", periods=n, freq="15min")

    return pd.DataFrame({
        "transaction_id": [f"TX_{i:06d}" for i in range(n)],
        "customer_id": [f"CUST_{i % 20:05d}" for i in range(n)],
        "timestamp": timestamps,
        "is_weekend": rng.integers(0, 2, size=n).astype(float),
        "card_bin": rng.integers(400000, 500000, size=n).astype(float),
        "card_type": rng.integers(0, 3, size=n).astype(float),
        "card_tier": rng.integers(0, 3, size=n).astype(float),
        "account_age_days": rng.integers(30, 3650, size=n).astype(float),
        "amount": rng.uniform(10_000, 5_000_000, size=n),
        "currency": rng.integers(0, 2, size=n).astype(float),
        "mcc_code": rng.integers(1000, 9999, size=n).astype(float),
        "merchant_category": rng.integers(0, 5, size=n).astype(float),
        "merchant_country": rng.integers(0, 4, size=n).astype(float),
        "device_type": rng.integers(0, 3, size=n).astype(float),
        "os": rng.integers(0, 4, size=n).astype(float),
        "ip_country": rng.integers(0, 4, size=n).astype(float),
        "distance_from_home_km": rng.exponential(200, size=n),
        "cvv_match": rng.integers(0, 2, size=n).astype(float),
        "is_3d_secure": rng.integers(0, 2, size=n).astype(float),
        "transaction_status": rng.integers(0, 2, size=n).astype(float),
        "tx_count_last_1h": rng.integers(1, 10, size=n).astype(float),
        "tx_count_last_24h": rng.integers(1, 60, size=n).astype(float),
        "time_since_last_tx_min": rng.exponential(500, size=n),
        "avg_amount_last_30d": rng.uniform(100_000, 3_000_000, size=n),
        "amount_ratio_vs_avg": rng.uniform(0.1, 5.0, size=n),
        "is_new_device": rng.integers(0, 2, size=n).astype(float),
        "is_new_merchant": rng.integers(0, 2, size=n).astype(float),
        "amount_log": rng.uniform(9, 15, size=n),
        "transaction_hour": rng.integers(0, 24, size=n).astype(float),
        "transaction_day_of_week": rng.integers(0, 7, size=n).astype(float),
        "is_night_transaction": rng.integers(0, 2, size=n).astype(float),
        "segment_encoded": rng.integers(0, 4, size=n).astype(float),
        "customer_tx_count": rng.integers(1, 50, size=n).astype(float),
        "customer_avg_amount": rng.uniform(100_000, 3_000_000, size=n),
        "is_fraud": labels,
    })


_FEATURE_COLS = [
    "is_weekend", "card_bin", "card_type", "card_tier",
    "account_age_days", "amount", "currency", "mcc_code",
    "merchant_category", "merchant_country", "device_type", "os",
    "ip_country", "distance_from_home_km", "cvv_match", "is_3d_secure",
    "transaction_status", "tx_count_last_1h", "tx_count_last_24h",
    "time_since_last_tx_min", "avg_amount_last_30d", "amount_ratio_vs_avg",
    "is_new_device", "is_new_merchant", "transaction_hour",
    "transaction_day_of_week", "is_night_transaction", "amount_log",
    "segment_encoded", "customer_tx_count", "customer_avg_amount",
]


def _make_tiny_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train a tiny real XGBoost model for testing."""
    model = XGBClassifier(
        n_estimators=10,
        max_depth=2,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
        scale_pos_weight=float((y == 0).sum()) / max(float(y.sum()), 1),
    )
    model.fit(X, y)
    return model


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def processed_df() -> pd.DataFrame:
    """Full processed synthetic DataFrame."""
    return _make_feature_df(n=300)


@pytest.fixture()
def train_df(processed_df: pd.DataFrame) -> pd.DataFrame:
    """First 240 rows as training set."""
    return processed_df.iloc[:240].copy()


@pytest.fixture()
def test_df(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Last 60 rows as test set."""
    return processed_df.iloc[240:].copy()


@pytest.fixture()
def X_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """Training feature matrix."""
    return train_df[_FEATURE_COLS].copy()


@pytest.fixture()
def y_train(train_df: pd.DataFrame) -> pd.Series:
    """Training labels."""
    return train_df["is_fraud"].astype(int)


@pytest.fixture()
def X_test(test_df: pd.DataFrame) -> pd.DataFrame:
    """Test feature matrix."""
    return test_df[_FEATURE_COLS].copy()


@pytest.fixture()
def y_test(test_df: pd.DataFrame) -> pd.Series:
    """Test labels."""
    return test_df["is_fraud"].astype(int)


@pytest.fixture()
def trained_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Real tiny XGBoost model trained on synthetic data."""
    return _make_tiny_model(X_train, y_train)


@pytest.fixture()
def artifact_dir(
    tmp_path: Path,
    trained_model: XGBClassifier,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Path:
    """Populate a temporary directory with all required pipeline artifacts."""
    # Save model
    joblib.dump(trained_model, tmp_path / "xgb_fraud_model.joblib")

    # Save metrics
    metrics = {
        "roc_auc": 0.85, "pr_auc": 0.60, "f1": 0.55,
        "precision": 0.60, "recall": 0.50,
        "threshold": 0.35, "confusion_matrix": [[55, 3], [2, 0]],
        "best_iteration": 9,
    }
    with open(tmp_path / "metrics.json", "w") as fh:
        json.dump(metrics, fh)

    # processed/ sub-dir
    proc = tmp_path / "processed"
    proc.mkdir()
    train_df.to_parquet(proc / "train.parquet", index=False)
    test_df.to_parquet(proc / "test.parquet", index=False)

    with open(proc / "features.json", "w") as fh:
        json.dump({"target": "is_fraud", "features": _FEATURE_COLS}, fh)

    # Feature state artifacts for FraudDetector
    customer_stats = (
        train_df.groupby("customer_id")["amount"]
        .agg(customer_tx_count="count", customer_avg_amount="mean")
    )
    customer_stats.to_parquet(proc / "customer_stats.parquet")

    seg_map = {"INSPIRE": 0, "MASS": 1, "PRIVATE": 2, "PRIORITY": 3}
    with open(proc / "segment_label_map.json", "w") as fh:
        json.dump(seg_map, fh)

    with open(proc / "amount_median_train.json", "w") as fh:
        json.dump({"amount_median_train": float(train_df["amount"].median())}, fh)

    with open(proc / "categorical_maps.json", "w") as fh:
        json.dump({}, fh)

    # Evaluation sub-dir + evaluation.json for FraudDetector
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    eval_report = {
        "evaluated_at": "2026-01-01T00:00:00+00:00",
        "threshold_metrics": {
            "threshold": 0.35, "precision": 0.6, "recall": 0.5,
            "f1": 0.55, "pr_auc": 0.60, "roc_auc": 0.85,
            "confusion_matrix": [[55, 3], [2, 0]],
        },
        "baseline_comparison": {"status": "PASS", "metrics": {}},
        "overall_status": "PASS",
    }
    with open(eval_dir / "evaluation.json", "w") as fh:
        json.dump(eval_report, fh)

    return tmp_path


# ===========================================================================
# Tests — train.py: compute_class_weight
# ===========================================================================

class TestComputeClassWeight:
    """Tests for compute_class_weight."""

    def test_returns_correct_ratio(self, y_train: pd.Series) -> None:
        from ml_pipeline.src.train import compute_class_weight

        spw = compute_class_weight(y_train)
        n_neg = int((y_train == 0).sum())
        n_pos = int(y_train.sum())
        assert abs(spw - n_neg / n_pos) < 1e-9

    def test_raises_when_no_positives(self) -> None:
        from ml_pipeline.src.train import compute_class_weight

        y_all_legit = pd.Series([0] * 100)
        with pytest.raises(ValueError, match="No positive"):
            compute_class_weight(y_all_legit)

    def test_returns_float(self, y_train: pd.Series) -> None:
        from ml_pipeline.src.train import compute_class_weight

        result = compute_class_weight(y_train)
        assert isinstance(result, float)


# ===========================================================================
# Tests — train.py: prepare_features
# ===========================================================================

class TestPrepareFeatures:
    """Tests for prepare_features."""

    def test_shapes_match(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.train import prepare_features

        X_tr, y_tr, X_te, y_te = prepare_features(train_df, test_df, _FEATURE_COLS)
        assert X_tr.shape == (len(train_df), len(_FEATURE_COLS))
        assert X_te.shape == (len(test_df), len(_FEATURE_COLS))
        assert len(y_tr) == len(train_df)
        assert len(y_te) == len(test_df)

    def test_missing_feature_raises(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.train import prepare_features

        bad_cols = _FEATURE_COLS + ["nonexistent_col"]
        with pytest.raises(ValueError, match="missing from train set"):
            prepare_features(train_df, test_df, bad_cols)

    def test_target_excluded_from_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.train import prepare_features

        X_tr, _, _, _ = prepare_features(train_df, test_df, _FEATURE_COLS)
        assert "is_fraud" not in X_tr.columns


# ===========================================================================
# Tests — train.py: filter_numeric_features
# ===========================================================================

class TestFilterNumericFeatures:
    """Tests for filter_numeric_features."""

    def test_drops_object_column(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        from ml_pipeline.src.train import filter_numeric_features

        X_train_with_str = X_train.copy()
        X_test_with_str = X_test.copy()
        X_train_with_str["text_col"] = "hello"
        X_test_with_str["text_col"] = "world"

        X_tr_num, X_te_num = filter_numeric_features(X_train_with_str, X_test_with_str)
        assert "text_col" not in X_tr_num.columns
        assert "text_col" not in X_te_num.columns

    def test_preserves_all_numeric(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        from ml_pipeline.src.train import filter_numeric_features

        X_tr_num, _ = filter_numeric_features(X_train, X_test)
        assert X_tr_num.shape[1] == X_train.shape[1]


# ===========================================================================
# Tests — train.py: evaluate_model
# ===========================================================================

class TestEvaluateModel:
    """Tests for evaluate_model."""

    def test_returns_all_required_keys(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.train import evaluate_model

        metrics = evaluate_model(trained_model, X_test, y_test, threshold=0.5)
        for key in ("roc_auc", "pr_auc", "f1", "precision", "recall",
                    "threshold", "confusion_matrix", "best_iteration"):
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_in_valid_range(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.train import evaluate_model

        metrics = evaluate_model(trained_model, X_test, y_test, threshold=0.5)
        for key in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"

    def test_confusion_matrix_shape(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.train import evaluate_model

        metrics = evaluate_model(trained_model, X_test, y_test, threshold=0.5)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2


# ===========================================================================
# Tests — train.py: find_optimal_threshold
# ===========================================================================

class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold."""

    def test_returns_float_in_unit_interval(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.train import find_optimal_threshold

        t = find_optimal_threshold(trained_model, X_test, y_test, min_recall=0.5)
        assert 0.0 <= t <= 1.0

    def test_fallback_when_strict_constraints(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Should not raise even with impossible constraints — falls back gracefully."""
        from ml_pipeline.src.train import find_optimal_threshold

        t = find_optimal_threshold(
            trained_model, X_test, y_test,
            min_recall=0.9999, max_threshold=0.0001,
        )
        assert isinstance(t, float)


# ===========================================================================
# Tests — train.py: get_feature_importance
# ===========================================================================

class TestGetFeatureImportance:
    """Tests for get_feature_importance."""

    def test_returns_dataframe_with_correct_cols(
        self,
        trained_model: XGBClassifier,
        X_train: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.train import get_feature_importance

        df = get_feature_importance(trained_model, X_train.columns.tolist())
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "importance" in df.columns

    def test_top_n_respected(
        self,
        trained_model: XGBClassifier,
        X_train: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.train import get_feature_importance

        df = get_feature_importance(trained_model, X_train.columns.tolist(), top_n=5)
        assert len(df) <= 5


# ===========================================================================
# Tests — train.py: save_artifacts
# ===========================================================================

class TestSaveArtifacts:
    """Tests for save_artifacts."""

    def test_creates_expected_files(
        self,
        trained_model: XGBClassifier,
        tmp_path: Path,
    ) -> None:
        from ml_pipeline.src.train import save_artifacts, get_feature_importance

        metrics = {"roc_auc": 0.9, "pr_auc": 0.7, "f1": 0.6}
        fi = get_feature_importance(trained_model, _FEATURE_COLS, top_n=5)
        save_artifacts(trained_model, metrics, fi, str(tmp_path))

        assert (tmp_path / "xgb_fraud_model.joblib").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "feature_importance.csv").exists()

    def test_metrics_json_content(
        self,
        trained_model: XGBClassifier,
        tmp_path: Path,
    ) -> None:
        from ml_pipeline.src.train import save_artifacts, get_feature_importance

        metrics = {"roc_auc": 0.91, "pr_auc": 0.72}
        fi = get_feature_importance(trained_model, _FEATURE_COLS, top_n=3)
        save_artifacts(trained_model, metrics, fi, str(tmp_path))

        with open(tmp_path / "metrics.json") as fh:
            loaded = json.load(fh)
        assert loaded["roc_auc"] == 0.91


# ===========================================================================
# Tests — train.py: load_data
# ===========================================================================

class TestLoadData:
    """Tests for load_data."""

    def test_returns_train_test_features(self, artifact_dir: Path) -> None:
        from ml_pipeline.src.train import load_data

        proc_dir = str(artifact_dir / "processed")
        train, test, features = load_data(proc_dir)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(features, list)
        assert len(features) == len(_FEATURE_COLS)

    def test_raises_when_artifacts_missing(self, tmp_path: Path) -> None:
        from ml_pipeline.src.train import load_data

        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path))


# ===========================================================================
# Tests — train.py: run_training (integration, MLflow mocked)
# ===========================================================================

class TestRunTraining:
    """Integration tests for run_training with MLflow mocked."""

    def test_run_training_returns_metrics(self, artifact_dir: Path) -> None:
        from ml_pipeline.src.train import run_training

        proc_dir = str(artifact_dir / "processed")
        models_dir = str(artifact_dir / "model_output")

        mock_run_id = MagicMock()
        mock_run_id.info.run_id = "test-run-id-001"
        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/test-run-id-001/model"

        with patch("ml_pipeline.src.train.mlflow") as mock_mlflow, \
             patch("ml_pipeline.src.train.register_model_from_run", return_value=1), \
             patch("ml_pipeline.src.train.transition_model_version_stage"):
            mock_mlflow.start_run.return_value.__enter__ = lambda s: s
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.active_run.return_value = mock_run_id
            mock_mlflow.xgboost.log_model.return_value = mock_model_info
            mock_mlflow.set_experiment.return_value = None
            mock_mlflow.set_tracking_uri.return_value = None
            os.environ["MLFLOW_REGISTER_MODEL"] = "false"

            metrics = run_training(proc_dir, models_dir)

        assert "pr_auc" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_run_training_creates_model_file(self, artifact_dir: Path) -> None:
        from ml_pipeline.src.train import run_training

        proc_dir = str(artifact_dir / "processed")
        models_dir = str(artifact_dir / "model_output2")

        mock_run_id = MagicMock()
        mock_run_id.info.run_id = "test-run-id-002"
        mock_model_info = MagicMock()
        mock_model_info.model_uri = "runs:/test-run-id-002/model"

        with patch("ml_pipeline.src.train.mlflow") as mock_mlflow, \
             patch("ml_pipeline.src.train.register_model_from_run", return_value=1), \
             patch("ml_pipeline.src.train.transition_model_version_stage"):
            mock_mlflow.start_run.return_value.__enter__ = lambda s: s
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.active_run.return_value = mock_run_id
            mock_mlflow.xgboost.log_model.return_value = mock_model_info
            os.environ["MLFLOW_REGISTER_MODEL"] = "false"

            run_training(proc_dir, models_dir)

        assert (Path(models_dir) / "xgb_fraud_model.joblib").exists()


# ===========================================================================
# Tests — evaluate.py: compare_baseline
# ===========================================================================

class TestCompareBaseline:
    """Tests for compare_baseline."""

    def test_pass_when_current_better(self) -> None:
        from ml_pipeline.src.evaluate import compare_baseline

        current = {"pr_auc": 0.80, "f1": 0.75, "recall": 0.90}
        baseline = {"pr_auc": 0.75, "f1": 0.70, "recall": 0.85}
        result = compare_baseline(current, baseline)
        assert result["status"] == "PASS"

    def test_fail_when_current_much_worse(self) -> None:
        from ml_pipeline.src.evaluate import compare_baseline

        current = {"pr_auc": 0.50, "f1": 0.40, "recall": 0.60}
        baseline = {"pr_auc": 0.80, "f1": 0.75, "recall": 0.90}
        result = compare_baseline(current, baseline)
        assert result["status"] == "FAIL"

    def test_pass_within_tolerance(self) -> None:
        from ml_pipeline.src.evaluate import compare_baseline

        # Default tolerance: pr_auc=0.02, f1=0.03, recall=0.02
        current = {"pr_auc": 0.79, "f1": 0.73, "recall": 0.89}
        baseline = {"pr_auc": 0.80, "f1": 0.75, "recall": 0.90}
        result = compare_baseline(current, baseline)
        assert result["status"] == "PASS"

    def test_custom_tolerance(self) -> None:
        from ml_pipeline.src.evaluate import compare_baseline

        current = {"pr_auc": 0.70}
        baseline = {"pr_auc": 0.80}
        result = compare_baseline(current, baseline, tolerance={"pr_auc": 0.15})
        assert result["status"] == "PASS"

    def test_returns_metric_details(self) -> None:
        from ml_pipeline.src.evaluate import compare_baseline

        current = {"pr_auc": 0.80, "f1": 0.70, "recall": 0.88}
        baseline = {"pr_auc": 0.75, "f1": 0.68, "recall": 0.86}
        result = compare_baseline(current, baseline)
        assert "metrics" in result
        assert "pr_auc" in result["metrics"]
        detail = result["metrics"]["pr_auc"]
        assert "baseline" in detail and "current" in detail and "delta" in detail


# ===========================================================================
# Tests — evaluate.py: evaluate_threshold
# ===========================================================================

class TestEvaluateThreshold:
    """Tests for evaluate_threshold."""

    def test_returns_all_metric_keys(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_threshold

        result = evaluate_threshold(trained_model, X_test, y_test, min_recall=0.3)
        for key in ("threshold", "precision", "recall", "f1", "pr_auc", "roc_auc"):
            assert key in result

    def test_values_in_unit_interval(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_threshold

        result = evaluate_threshold(trained_model, X_test, y_test, min_recall=0.3)
        for key in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
            assert 0.0 <= result[key] <= 1.0

    def test_saves_pr_curve_png(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tmp_path: Path,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_threshold

        evaluate_threshold(
            trained_model, X_test, y_test,
            min_recall=0.3, output_dir=tmp_path,
        )
        assert (tmp_path / "pr_curve.png").exists()

    def test_no_png_when_output_dir_none(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_threshold

        # Should complete without error when output_dir=None
        result = evaluate_threshold(
            trained_model, X_test, y_test,
            min_recall=0.3, output_dir=None,
        )
        assert isinstance(result, dict)


# ===========================================================================
# Tests — evaluate.py: evaluate_segments
# ===========================================================================

class TestEvaluateSegments:
    """Tests for evaluate_segments."""

    def test_returns_dataframe_with_all_row(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_df: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_segments

        seg_df = evaluate_segments(
            trained_model, X_test, y_test, test_df,
            threshold=0.5, segment_col="segment_encoded",
        )
        assert isinstance(seg_df, pd.DataFrame)
        assert "ALL" in seg_df["segment"].values

    def test_fallback_when_segment_col_missing(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_df: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_segments

        seg_df = evaluate_segments(
            trained_model, X_test, y_test, test_df,
            threshold=0.5, segment_col="nonexistent_col",
        )
        # Only "ALL" row returned
        assert len(seg_df) == 1
        assert seg_df["segment"].iloc[0] == "ALL"


# ===========================================================================
# Tests — evaluate.py: save_evaluation_report
# ===========================================================================

class TestSaveEvaluationReport:
    """Tests for save_evaluation_report."""

    def test_creates_evaluation_json_and_segment_csv(self, tmp_path: Path) -> None:
        from ml_pipeline.src.evaluate import save_evaluation_report

        threshold_metrics = {
            "threshold": 0.35, "precision": 0.6, "recall": 0.8,
            "f1": 0.7, "pr_auc": 0.65, "roc_auc": 0.88,
        }
        segment_report = pd.DataFrame([{
            "segment": "ALL", "n_samples": 60, "n_fraud": 3,
            "fraud_rate": 5.0, "precision": 0.6, "recall": 0.8, "f1": 0.7, "pr_auc": 0.65,
        }])
        comparison = {"status": "PASS", "metrics": {}}

        save_evaluation_report(threshold_metrics, segment_report, comparison, str(tmp_path))

        assert (tmp_path / "evaluation.json").exists()
        assert (tmp_path / "segment_report.csv").exists()

        with open(tmp_path / "evaluation.json") as fh:
            report = json.load(fh)
        assert report["overall_status"] == "PASS"


# ===========================================================================
# Tests — evaluate.py: explain_shap (mocked)
# ===========================================================================

class TestExplainShap:
    """Tests for explain_shap — SHAP is mocked to avoid slow computation."""

    def test_creates_shap_output_files(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        from ml_pipeline.src.evaluate import explain_shap

        n = len(X_test)
        fake_shap = np.zeros((n, len(_FEATURE_COLS)))
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = fake_shap
        mock_explainer.expected_value = 0.0

        with patch("ml_pipeline.src.evaluate.shap.TreeExplainer", return_value=mock_explainer), \
             patch("ml_pipeline.src.evaluate.shap.summary_plot"), \
             patch("ml_pipeline.src.evaluate.shap.plots.waterfall"), \
             patch("ml_pipeline.src.evaluate.shap.Explanation", return_value=MagicMock()):
            explain_shap(trained_model, X_test, _FEATURE_COLS, output_dir=tmp_path, sample_size=20)

        assert (tmp_path / "shap_summary.png").exists()
        assert (tmp_path / "shap_waterfall.png").exists()


# ===========================================================================
# Tests — inference.py: FraudDetector._risk_level
# ===========================================================================

class TestRiskLevel:
    """Tests for FraudDetector._risk_level static method."""

    def test_low_risk(self) -> None:
        from ml_pipeline.src.inference import FraudDetector

        assert FraudDetector._risk_level(0.05) == "LOW"
        assert FraudDetector._risk_level(0.0) == "LOW"
        assert FraudDetector._risk_level(0.29) == "LOW"

    def test_medium_risk(self) -> None:
        from ml_pipeline.src.inference import FraudDetector

        assert FraudDetector._risk_level(0.30) == "MEDIUM"
        assert FraudDetector._risk_level(0.50) == "MEDIUM"
        assert FraudDetector._risk_level(0.59) == "MEDIUM"

    def test_high_risk(self) -> None:
        from ml_pipeline.src.inference import FraudDetector

        assert FraudDetector._risk_level(0.60) == "HIGH"
        assert FraudDetector._risk_level(0.99) == "HIGH"
        assert FraudDetector._risk_level(1.0) == "HIGH"


# ===========================================================================
# Tests — inference.py: FraudDetector (full constructor + predictions)
# ===========================================================================

@pytest.fixture()
def detector(artifact_dir: Path) -> "FraudDetector":  # noqa: F821
    """Build a real FraudDetector from the artifact_dir fixture."""
    from ml_pipeline.src.inference import FraudDetector

    return FraudDetector(
        models_dir=str(artifact_dir),
        processed_dir=str(artifact_dir / "processed"),
    )


class TestFraudDetector:
    """Tests for FraudDetector class."""

    def test_loads_without_error(self, detector: "FraudDetector") -> None:  # noqa: F821
        assert detector is not None

    def test_health_check_status_ok(self, detector: "FraudDetector") -> None:  # noqa: F821
        health = detector.health_check()
        assert health["status"] == "OK"
        assert health["feature_count"] > 0
        assert isinstance(health["threshold"], float)

    def test_health_check_has_required_keys(self, detector: "FraudDetector") -> None:  # noqa: F821
        health = detector.health_check()
        for key in ("status", "loaded_at", "threshold", "feature_count",
                    "features", "model_type", "best_iteration"):
            assert key in health, f"Missing health key: {key}"

    def test_predict_single_returns_dict(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        row = test_df.iloc[0].to_dict()
        result = detector.predict_single(row)
        assert isinstance(result, dict)
        for key in ("transaction_id", "fraud_score", "is_fraud_pred", "threshold", "risk_level"):
            assert key in result

    def test_predict_single_score_in_unit_interval(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        row = test_df.iloc[0].to_dict()
        result = detector.predict_single(row)
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_predict_single_risk_level_valid(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        row = test_df.iloc[0].to_dict()
        result = detector.predict_single(row)
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_predict_single_unknown_tx_id(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        row = test_df.iloc[0].to_dict()
        row.pop("transaction_id", None)
        result = detector.predict_single(row)
        assert result["transaction_id"] == "UNKNOWN"

    def test_predict_batch_returns_dataframe(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        result = detector.predict_batch(test_df.head(10))
        assert isinstance(result, pd.DataFrame)
        for col in ("fraud_score", "is_fraud_pred", "risk_level"):
            assert col in result.columns

    def test_predict_batch_row_count_preserved(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        batch = test_df.head(15)
        result = detector.predict_batch(batch)
        assert len(result) == 15

    def test_predict_batch_empty_raises(
        self,
        detector: "FraudDetector",  # noqa: F821
    ) -> None:
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            detector.predict_batch(empty_df)

    def test_predict_batch_scores_in_unit_interval(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        result = detector.predict_batch(test_df.head(20))
        assert (result["fraud_score"] >= 0.0).all()
        assert (result["fraud_score"] <= 1.0).all()

    def test_fraud_detector_missing_artifacts_raises(self, tmp_path: Path) -> None:
        from ml_pipeline.src.inference import FraudDetector

        with pytest.raises(FileNotFoundError):
            FraudDetector(str(tmp_path), str(tmp_path))

# ===========================================================================
# Tests — inference.py: FraudDetector._transform
# ===========================================================================

class TestFraudDetectorTransform:
    """Tests for the internal _transform method."""

    def test_transform_adds_derived_features(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        sample = test_df.head(5).copy()
        # Simulate raw input by removing derived features
        for col in ("amount_log", "transaction_hour", "is_night_transaction",
                    "transaction_day_of_week"):
            if col in sample.columns:
                sample = sample.drop(columns=[col])

        # Need raw fields for _transform
        sample["customer_tier"] = "MASS"
        transformed = detector._transform(sample)
        assert "amount_log" in transformed.columns

    def test_transform_raises_without_timestamp(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        sample = test_df.head(3).drop(columns=["timestamp"])
        with pytest.raises(KeyError, match="timestamp"):
            detector._transform(sample)

    def test_transform_fills_unseen_customer(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        sample = test_df.head(3).copy()
        # Use a customer_id guaranteed to not exist in training stats
        sample["customer_id"] = "TOTALLY_NEW_CUSTOMER_XYZ_999"
        transformed = detector._transform(sample)
        assert (transformed["customer_tx_count"] == 0).all()

# ===========================================================================
# Tests — evaluate.py: configure_mlflow
# ===========================================================================

class TestConfigureMlflow:
    """Tests for configure_mlflow."""

    @patch("ml_pipeline.src.mlflow_utils.mlflow")
    def test_defaults(self, mock_mlflow: MagicMock) -> None:
        from ml_pipeline.src.mlflow_utils import configure_mlflow

        with patch.dict(os.environ, {}, clear=True):
            name = configure_mlflow("train")
        mock_mlflow.set_experiment.assert_called_once()
        assert name.startswith("train-")

    @patch("ml_pipeline.src.mlflow_utils.mlflow")
    def test_custom_run_name(self, mock_mlflow: MagicMock) -> None:
        from ml_pipeline.src.mlflow_utils import configure_mlflow

        env = {"MLFLOW_RUN_NAME": "my-custom-run"}
        with patch.dict(os.environ, env, clear=True):
            name = configure_mlflow("eval")
        assert name == "my-custom-run"

    @patch("ml_pipeline.src.mlflow_utils.mlflow")
    def test_tracking_uri_set(self, mock_mlflow: MagicMock) -> None:
        from ml_pipeline.src.mlflow_utils import configure_mlflow

        env = {"MLFLOW_TRACKING_URI": "http://localhost:5000"}
        with patch.dict(os.environ, env, clear=True):
            configure_mlflow("eval")
        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "http://localhost:5000"
        )

# ===========================================================================
# Tests — evaluate.py: build_mlflow_tags
# ===========================================================================

class TestBuildMlflowTags:
    """Tests for build_mlflow_tags."""

    def test_filters_empty_values(self) -> None:
        from ml_pipeline.src.mlflow_utils import build_mlflow_tags

        with patch.dict(os.environ, {}, clear=True):
            tags = build_mlflow_tags(
                "eval",
                models_dir="/m",
                processed_dir="/p",
                evaluation_dir="/e",
            )
        # Empty env vars should be filtered out
        assert "pipeline.stage" in tags
        assert tags["pipeline.stage"] == "eval"
        assert all(v for v in tags.values())

    def test_includes_manual_source(self) -> None:
        from ml_pipeline.src.mlflow_utils import build_mlflow_tags

        env = {"PIPELINE_SOURCE": "airflow"}
        with patch.dict(os.environ, env, clear=True):
            tags = build_mlflow_tags(
                "train",
                models_dir="/models",
                processed_dir="/proc",
                evaluation_dir="/eval",
            )
        assert tags["pipeline.source"] == "airflow"

# ===========================================================================
# Tests — evaluate.py: evaluate_threshold
# ===========================================================================

class TestEvaluateThreshold:
    """Tests for evaluate_threshold."""

    def test_returns_required_keys(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_threshold

        result = evaluate_threshold(
            trained_model, X_test, y_test,
            min_recall=0.10, max_threshold=0.90,
        )
        for key in ("threshold", "precision", "recall",
                     "f1", "pr_auc", "roc_auc", "confusion_matrix"):
            assert key in result, f"Missing key: {key}"

    def test_metrics_in_valid_range(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_threshold

        result = evaluate_threshold(
            trained_model, X_test, y_test,
            min_recall=0.10, max_threshold=0.90,
        )
        assert 0.0 <= result["threshold"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_with_output_dir(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tmp_path: Path,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_threshold

        result = evaluate_threshold(
            trained_model, X_test, y_test,
            min_recall=0.10, max_threshold=0.90,
            output_dir=tmp_path,
        )
        assert (tmp_path / "pr_curve.png").exists()
        assert result["threshold"] > 0

    def test_relaxed_recall(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_threshold

        # min_recall=1.0 may force relaxation
        result = evaluate_threshold(
            trained_model, X_test, y_test,
            min_recall=1.0, max_threshold=0.01,
        )
        assert "threshold" in result

# ===========================================================================
# Tests — evaluate.py: evaluate_segments
# ===========================================================================

class TestEvaluateSegments:
    """Tests for evaluate_segments."""

    def test_returns_all_segment_and_overall(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_df: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_segments

        report = evaluate_segments(
            trained_model, X_test, y_test,
            test_df, threshold=0.5,
        )
        assert isinstance(report, pd.DataFrame)
        assert "ALL" in report["segment"].values

    def test_missing_segment_col(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_df: pd.DataFrame,
    ) -> None:
        from ml_pipeline.src.evaluate import evaluate_segments

        df_no_seg = test_df.drop(
            columns=["segment_encoded"], errors="ignore"
        )
        report = evaluate_segments(
            trained_model, X_test, y_test,
            df_no_seg, threshold=0.5,
            segment_col="nonexistent_col",
        )
        # Should return only the ALL row
        assert len(report) == 1
        assert report.iloc[0]["segment"] == "ALL"

# ===========================================================================
# Tests — inference.py: predict_single
# ===========================================================================

class TestPredictSingle:
    """Tests for FraudDetector.predict_single."""

    def test_returns_required_keys(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        row = test_df.iloc[0].to_dict()
        result = detector.predict_single(row)
        for key in ("transaction_id", "fraud_score",
                     "is_fraud_pred", "threshold", "risk_level"):
            assert key in result

    def test_score_in_unit_interval(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        row = test_df.iloc[0].to_dict()
        result = detector.predict_single(row)
        assert 0.0 <= result["fraud_score"] <= 1.0

    def test_risk_level_is_valid(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        row = test_df.iloc[0].to_dict()
        result = detector.predict_single(row)
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

# ===========================================================================
# Tests — inference.py: predict_batch (extended)
# ===========================================================================

class TestPredictBatchExtended:
    """Extended tests for FraudDetector.predict_batch."""

    def test_empty_raises(
        self,
        detector: "FraudDetector",  # noqa: F821
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            detector.predict_batch(pd.DataFrame())

    def test_output_has_risk_level(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        result = detector.predict_batch(test_df.head(5))
        assert "risk_level" in result.columns
        assert set(result["risk_level"].unique()).issubset(
            {"LOW", "MEDIUM", "HIGH"}
        )

    def test_is_fraud_pred_binary(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        result = detector.predict_batch(test_df.head(5))
        assert set(result["is_fraud_pred"].unique()).issubset({0, 1})

# ===========================================================================
# Tests — inference.py: _risk_level
# ===========================================================================

class TestRiskLevel:
    """Tests for FraudDetector._risk_level."""

    def test_low(
        self,
        detector: "FraudDetector",  # noqa: F821
    ) -> None:
        assert detector._risk_level(0.1) == "LOW"

    def test_medium(
        self,
        detector: "FraudDetector",  # noqa: F821
    ) -> None:
        assert detector._risk_level(0.5) == "MEDIUM"

    def test_high(
        self,
        detector: "FraudDetector",  # noqa: F821
    ) -> None:
        assert detector._risk_level(0.9) == "HIGH"

# ===========================================================================
# Tests — evaluate.py: load_artifacts
# ===========================================================================

class TestLoadArtifacts:
    """Tests for load_artifacts."""

    def test_loads_successfully(
        self,
        artifact_dir: Path,
    ) -> None:
        from ml_pipeline.src.evaluate import load_artifacts

        model, X_test, y_test, feat_cols, baseline = load_artifacts(
            str(artifact_dir), str(artifact_dir / "processed"),
        )
        assert hasattr(model, "predict_proba")
        assert isinstance(X_test, pd.DataFrame)
        assert len(X_test) == len(y_test)
        assert isinstance(feat_cols, list)
        assert len(feat_cols) > 0
        assert isinstance(baseline, dict)

    def test_missing_model_raises(
        self, tmp_path: Path,
    ) -> None:
        from ml_pipeline.src.evaluate import load_artifacts

        with pytest.raises(FileNotFoundError, match="model"):
            load_artifacts(str(tmp_path), str(tmp_path))

    def test_missing_features_raises(
        self,
        artifact_dir: Path,
    ) -> None:
        from ml_pipeline.src.evaluate import load_artifacts

        # Remove a feature from features.json to cause mismatch
        feat_path = artifact_dir / "processed" / "features.json"
        with open(feat_path) as fh:
            meta = json.load(fh)
        meta["features"].append("nonexistent_column_xyz")
        with open(feat_path, "w") as fh:
            json.dump(meta, fh)

        with pytest.raises(ValueError, match="missing"):
            load_artifacts(
                str(artifact_dir),
                str(artifact_dir / "processed"),
            )

# ===========================================================================
# Tests — evaluate.py: save_evaluation_report
# ===========================================================================

class TestSaveEvaluationReport:
    """Tests for save_evaluation_report."""

    def test_creates_files(self, tmp_path: Path) -> None:
        from ml_pipeline.src.evaluate import save_evaluation_report

        metrics = {
            "threshold": 0.35, "precision": 0.6,
            "recall": 0.5, "f1": 0.55,
            "pr_auc": 0.60, "roc_auc": 0.85,
        }
        comparison = {"status": "PASS", "metrics": {}}
        seg_df = pd.DataFrame([{
            "segment": "ALL", "n_samples": 100,
            "n_fraud": 5, "fraud_rate": 5.0,
            "precision": 0.8, "recall": 0.6,
            "f1": 0.69, "pr_auc": 0.7,
        }])

        save_evaluation_report(
            metrics, seg_df, comparison,
            str(tmp_path),
        )
        assert (tmp_path / "evaluation.json").exists()
        assert (tmp_path / "segment_report.csv").exists()

        with open(tmp_path / "evaluation.json") as fh:
            report = json.load(fh)
        assert report["overall_status"] == "PASS"


# ===========================================================================
# Tests — evaluate.py: explain_shap
# ===========================================================================

class TestExplainShap:
    """Tests for explain_shap."""

    def test_creates_shap_plots(
        self,
        trained_model: XGBClassifier,
        X_test: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        from ml_pipeline.src.evaluate import explain_shap

        explain_shap(
            trained_model, X_test,
            X_test.columns.tolist(),
            output_dir=tmp_path,
            sample_size=20,
        )
        assert (tmp_path / "shap_summary.png").exists()
        assert (tmp_path / "shap_waterfall.png").exists()

# ===========================================================================
# Tests — inference.py: _transform edge cases
# ===========================================================================

class TestTransformEdgeCases:
    """Tests for _transform edge-case paths."""

    def test_timestamp_string_coercion(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        sample = test_df.head(3).copy()
        # Convert timestamp to string to trigger coercion
        sample["timestamp"] = sample["timestamp"].astype(str)
        transformed = detector._transform(sample)
        assert "transaction_hour" in transformed.columns

    def test_no_customer_id(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        sample = test_df.head(3).copy()
        sample = sample.drop(columns=["customer_id"])
        transformed = detector._transform(sample)
        assert "customer_tx_count" in transformed.columns
        assert (transformed["customer_tx_count"] == 0).all()

    def test_amount_string_coercion(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        sample = test_df.head(3).copy()
        sample["amount"] = sample["amount"].astype(str)
        transformed = detector._transform(sample)
        assert "amount_log" in transformed.columns

    def test_hour_of_day_rename(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        sample = test_df.head(3).copy()
        if "transaction_hour" in sample.columns:
            sample = sample.rename(
                columns={"transaction_hour": "hour_of_day"}
            )
        transformed = detector._transform(sample)
        assert "transaction_hour" in transformed.columns
        assert "hour_of_day" not in transformed.columns

    def test_categorical_encoding(
        self,
        artifact_dir: Path,
        test_df: pd.DataFrame,
    ) -> None:
        """Covers _transform lines 407-414."""
        from ml_pipeline.src.inference import FraudDetector

        # Write a non-empty categorical_maps so the loop executes
        cat_maps = {"currency": {"VND": 0, "USD": 1}}
        proc = artifact_dir / "processed"
        with open(proc / "categorical_maps.json", "w") as fh:
            json.dump(cat_maps, fh)

        det = FraudDetector(
            str(artifact_dir),
            str(proc),
        )
        sample = test_df.head(3).copy()
        # Give currency string values matching the map
        sample["currency"] = "VND"
        transformed = det._transform(sample)
        assert (transformed["currency"] == 0).all()

    def test_high_cardinality_drop(
        self,
        detector: "FraudDetector",  # noqa: F821
        test_df: pd.DataFrame,
    ) -> None:
        """Covers _transform lines 416-419."""
        sample = test_df.head(3).copy()
        sample["merchant_name"] = "SomeShop"
        sample["merchant_city"] = "Hanoi"
        transformed = detector._transform(sample)
        assert "merchant_name" not in transformed.columns
        assert "merchant_city" not in transformed.columns

    def test_score_fills_missing_features(
        self,
        artifact_dir: Path,
        test_df: pd.DataFrame,
    ) -> None:
        """Covers _score lines 436-442."""
        from ml_pipeline.src.inference import FraudDetector

        proc = artifact_dir / "processed"
        # Add a fake extra feature to features.json
        feat_path = proc / "features.json"
        with open(feat_path) as fh:
            meta = json.load(fh)
        meta["features"].append("invented_feat_xyz")
        with open(feat_path, "w") as fh:
            json.dump(meta, fh)

        det = FraudDetector(str(artifact_dir), str(proc))
        # Mock predict_proba to avoid XGBoost feature mismatch
        dummy = np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
        with patch.object(
            det._model, "predict_proba",
            return_value=dummy,
        ):
            scores = det._score(test_df.head(3).copy())
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

# ===========================================================================
# Tests — train.py: configure_mlflow (covers lines 79, 89)
# ===========================================================================

class TestTrainConfigureMlflow:
    """Tests for train.configure_mlflow (imported from mlflow_utils)."""

    @patch("ml_pipeline.src.mlflow_utils.mlflow")
    def test_tracking_uri(
        self, mock_mlflow: MagicMock,
    ) -> None:
        from ml_pipeline.src.mlflow_utils import configure_mlflow

        env = {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}
        with patch.dict(os.environ, env, clear=True):
            configure_mlflow("train")
        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "http://mlflow:5000"
        )

    @patch("ml_pipeline.src.mlflow_utils.mlflow")
    def test_custom_run_name(
        self, mock_mlflow: MagicMock,
    ) -> None:
        from ml_pipeline.src.mlflow_utils import configure_mlflow

        env = {"MLFLOW_RUN_NAME": "custom-run"}
        with patch.dict(os.environ, env, clear=True):
            name = configure_mlflow("train")
        assert name == "custom-run"