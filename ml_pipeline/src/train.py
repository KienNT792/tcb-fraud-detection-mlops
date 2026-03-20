"""
TCB Fraud Detection — Model Training Pipeline.

Loads processed artifacts from the preprocessing stage, trains an XGBoost
classifier, evaluates performance with fraud-appropriate metrics (F1, PR-AUC,
ROC-AUC), and persists all model artifacts to disk.

Pipeline order
--------------
load_data → prepare_features → compute_class_weight → train_model
→ evaluate_model → save_artifacts

Usage
-----
    python -m ml_pipeline.src.train
    # OR
    from ml_pipeline.src.train import run_training
    run_training("data/processed", "models")
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
# Columns that must never be used as model features
_IDENTIFIER_COLS: frozenset[str] = frozenset(
    {"transaction_id", "customer_id", "timestamp", "is_fraud"}
)

MODEL_FILENAME = "xgb_fraud_model.joblib"
METRICS_FILENAME = "metrics.json"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.csv"


# Step 1 — Load Processed Data
def load_data(
    processed_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the processed train/test Parquet files and feature schema.

    Reads the artifacts produced by the preprocessing pipeline:
    - ``train.parquet``, ``test.parquet``: processed splits.
    - ``features.json``: canonical feature list
      (excludes identifiers & target).

    Args:
        processed_dir: Path to the ``data/processed/`` directory.

    Returns:
        Tuple of ``(train_df, test_df, feature_cols)`` where *feature_cols* is
        the ordered list of model input features.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    base = Path(processed_dir)

    for fname in ("train.parquet", "test.parquet", "features.json"):
        if not (base / fname).exists():
            raise FileNotFoundError(
                f"Missing required artifact: {base / fname}. "
                "Run the preprocessing pipeline first."
            )

    logger.info("Loading processed artifacts from: %s", base)

    train = pd.read_parquet(base / "train.parquet")
    test = pd.read_parquet(base / "test.parquet")

    with open(base / "features.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    feature_cols: list[str] = meta["features"]

    logger.info(
        "Loaded — Train: %s | Test: %s | Features: %d",
        train.shape,
        test.shape,
        len(feature_cols),
    )
    return train, test, feature_cols


# Step 2 — Prepare Feature Matrices
def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extract X/y matrices from train and test DataFrames.

    Only columns defined in *feature_cols* are used as model inputs.
    Any identifier or target column is excluded.

    Args:
        train: Processed training DataFrame.
        test: Processed test DataFrame.
        feature_cols: Ordered list of feature names from ``features.json``.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)``.

    Raises:
        ValueError: If any feature in *feature_cols* is absent from the data.
    """
    # Guard against schema drift — fail loudly rather than silently drop cols
    missing_train = [c for c in feature_cols if c not in train.columns]
    missing_test = [c for c in feature_cols if c not in test.columns]
    if missing_train:
        raise ValueError(f"Features missing from train set: {missing_train}")
    if missing_test:
        raise ValueError(f"Features missing from test set: {missing_test}")

    X_train: pd.DataFrame = train[feature_cols].copy()
    y_train: pd.Series = train["is_fraud"].astype(int)
    X_test: pd.DataFrame = test[feature_cols].copy()
    y_test: pd.Series = test["is_fraud"].astype(int)

    logger.info(
        "Feature matrices — X_train: %s | X_test: %s",
        X_train.shape,
        X_test.shape,
    )
    logger.info(
        "Class distribution — Train: %d fraud / %d legit (%.2f%%) | "
        "Test: %d fraud / %d legit (%.2f%%)",
        y_train.sum(), len(y_train) - y_train.sum(), y_train.mean() * 100,
        y_test.sum(), len(y_test) - y_test.sum(), y_test.mean() * 100,
    )
    return X_train, y_train, X_test, y_test


# Step 3 — Compute Class Weight
def compute_class_weight(y_train: pd.Series) -> float:
    """Compute XGBoost scale_pos_weight to compensate for class imbalance.

    XGBoost uses ``scale_pos_weight = n_negative / n_positive`` to up-weight
    the minority (fraud) class during training. This is the recommended
    approach for imbalanced binary classification without resampling.

    Args:
        y_train: Binary target series (0 = legit, 1 = fraud).

    Returns:
        ``scale_pos_weight`` float value (n_negative / n_positive).

    Raises:
        ValueError: If the positive class is absent from *y_train*.
    """
    n_positive = int(y_train.sum())
    n_negative = int((y_train == 0).sum())

    if n_positive == 0:
        raise ValueError("No positive (fraud) samples found in y_train.")

    spw = n_negative / n_positive
    logger.info(
        "Class weight — Negative: %d | Positive: %d | scale_pos_weight: %.4f",
        n_negative,
        n_positive,
        spw,
    )
    return spw


# Step 3a — Filter Numeric Features (XGBoost compatibility)
def filter_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    before = X_train.shape[1]
    X_train_num = X_train.select_dtypes(include=[np.number, bool]).copy()
    numeric_cols: list[str] = X_train_num.columns.tolist()
    X_test_num = X_test[numeric_cols].copy()
    after = len(numeric_cols)

    dropped = before - after
    logger.info(
        (
            "Feature filtering — total: %d | numeric: %d | "
            "dropped (non-numeric): %d"
        ),
        before,
        after,
        dropped,
    )
    if dropped:
        non_numeric = [c for c in X_train.columns if c not in numeric_cols]
        logger.info("Dropped columns: %s", non_numeric)

    return X_train_num, X_test_num


# Step 4 — Train Model
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_pos_weight: float,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> XGBClassifier:
    """Train an XGBoost binary classifier for fraud detection.

    Uses ``binary:logistic`` objective, AUC evaluation metric, and early
    stopping on the test set to prevent over-fitting. The *scale_pos_weight*
    parameter handles class imbalance natively in the loss function.

    Args:
        X_train: Training feature matrix.
        y_train: Training binary labels.
        X_test: Validation feature matrix (used for early stopping only).
        y_test: Validation binary labels.
        scale_pos_weight: Ratio of negative to positive class samples.
        n_estimators: Maximum number of boosting rounds. Defaults to 1000.
        early_stopping_rounds: Stop if no improvement after this many rounds.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Fitted ``XGBClassifier`` instance.
    """
    logger.info(
        (
            "Training XGBoost — max_rounds: %d | early_stopping: %d | "
            "scale_pos_weight: %.4f"
        ),
        n_estimators,
        early_stopping_rounds,
        scale_pos_weight,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        # PR-AUC is more informative than ROC-AUC for imbalanced data.
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    best_round = model.best_iteration + 1
    logger.info(
        "Training complete — best round: %d | elapsed: %.1fs",
        best_round,
        elapsed,
    )
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute a comprehensive fraud-detection evaluation report.

    Computes metrics appropriate for imbalanced classification tasks.
    Accuracy is intentionally omitted; PR-AUC and F1 are the primary signals.

    Metrics computed:
    - ``roc_auc``:    ROC-AUC score.
    - ``pr_auc``:     Precision-Recall AUC
      (primary metric for imbalanced data).
    - ``f1``:         F1-score at the given threshold.
    - ``precision``:  Precision at the given threshold.
    - ``recall``:     Recall (sensitivity) at the given threshold.
    - ``confusion_matrix``: 2×2 confusion matrix as nested list.

    Args:
        model: Fitted XGBoost model.
        X_test: Test feature matrix.
        y_test: True binary labels.
        threshold: Decision threshold for converting probabilities to labels.
            Defaults to 0.5.

    Returns:
        Dictionary of metric names to values.
    """
    y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]
    y_pred: np.ndarray = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics: dict[str, Any] = {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "threshold": threshold,
        "confusion_matrix": cm,
        "best_iteration": int(model.best_iteration),
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    # --- Structured evaluation report ---
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info("  ROC-AUC   : %.4f", roc_auc)
    logger.info(
        "  PR-AUC    : %.4f  ← primary metric (imbalanced data)",
        pr_auc,
    )
    logger.info("  F1-Score  : %.4f", f1)
    logger.info("  Precision : %.4f", precision)
    logger.info("  Recall    : %.4f", recall)
    logger.info("  Threshold : %.2f", threshold)
    logger.info("-" * 40)
    logger.info("  Confusion Matrix (threshold=%.2f):", threshold)
    logger.info("    TN=%d | FP=%d | FN=%d | TP=%d", tn, fp, fn, tp)
    logger.info("  Best XGBoost iteration: %d", model.best_iteration)
    logger.info("=" * 60)

    return metrics


def get_feature_importance(
    model: XGBClassifier,
    feature_cols: list[str],
    top_n: int = 35,
) -> pd.DataFrame:
    """Extract and rank feature importances from the trained XGBoost model.

    Uses the ``gain`` importance type, which measures the average improvement
    in loss from splits using each feature — more meaningful than ``weight``
    for fraud detection.

    Args:
        model: Fitted XGBoost model.
        feature_cols: Ordered list of feature names matching training columns.
        top_n: Number of top features to return. Defaults to 35.

    Returns:
        DataFrame with columns ``feature`` and ``importance``, sorted by
        descending importance.
    """
    raw_importance = model.get_booster().get_score(importance_type="gain")

    rows = [
        {"feature": feat, "importance": raw_importance.get(feat, 0.0)}
        for feat in feature_cols
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )

    logger.info("Top 10 features by gain:")
    for _, row in df.head(10).iterrows():
        logger.info("  %-35s %.2f", row["feature"], row["importance"])

    return df


def save_artifacts(
    model: XGBClassifier,
    metrics: dict[str, Any],
    feature_importance: pd.DataFrame,
    models_dir: str,
) -> None:
    """Persist all model artifacts to the models directory.

    Saved files:
    - ``xgb_fraud_model.joblib``: serialized XGBoost model.
    - ``metrics.json``: evaluation metrics dictionary.
    - ``feature_importance.csv``: feature importance ranking.

    Args:
        model: Fitted XGBoost model.
        metrics: Evaluation metrics from *evaluate_model*.
        feature_importance: DataFrame from *get_feature_importance*.
        models_dir: Output directory (created if absent).
    """
    out = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = out / MODEL_FILENAME
    joblib.dump(model, model_path)
    logger.info("Model saved → %s", model_path)

    # Metrics
    metrics_path = out / METRICS_FILENAME
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    # Feature importance
    fi_path = out / FEATURE_IMPORTANCE_FILENAME
    feature_importance.to_csv(fi_path, index=False)
    logger.info("Feature importance saved → %s", fi_path)


def find_optimal_threshold(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_recall: float = 0.95,
    max_threshold: float = 0.7,
) -> float:
    from sklearn.metrics import precision_recall_curve

    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # Apply both constraints:
    # recall >= min_recall and threshold <= max_threshold.
    valid_mask = (recalls[:-1] >= min_recall) & (thresholds <= max_threshold)

    if not valid_mask.any():
        # Fallback to recall-only if max_threshold is too strict.
        logger.warning(
            "Không tìm được threshold <= %.2f với recall >= %.2f, "
            "fallback sang recall-only",
            max_threshold,
            min_recall,
        )
        valid_mask = recalls[:-1] >= min_recall

    if not valid_mask.any():
        optimal_threshold = float(thresholds[recalls[:-1].argmax()])
    else:
        best_idx = precisions[:-1][valid_mask].argmax()
        optimal_threshold = float(thresholds[valid_mask][best_idx])

    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score

    logger.info(
        "Optimal threshold: %.4f → Precision: %.4f | Recall: %.4f | F1: %.4f",
        optimal_threshold,
        precision_score(y_test, y_pred_opt, zero_division=0),
        recall_score(y_test, y_pred_opt, zero_division=0),
        f1_score(y_test, y_pred_opt, zero_division=0),
    )
    return optimal_threshold


def run_training(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("TCB FRAUD DETECTION — TRAINING PIPELINE START")
    logger.info("Processed data: %s", processed_dir)
    logger.info("Model output:   %s", models_dir)
    logger.info("=" * 60)

    train, test, feature_cols = load_data(processed_dir)
    X_train, y_train, X_test, y_test = prepare_features(
        train,
        test,
        feature_cols,
    )
    scale_pos_weight = compute_class_weight(y_train)

    # Drop non-numeric columns — XGBoost requires int/float/bool features
    X_train, X_test = filter_numeric_features(X_train, X_test)
    numeric_feature_cols: list[str] = X_train.columns.tolist()

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "model_type": "xgboost",
                "scale_pos_weight": round(scale_pos_weight, 4),
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 1000,
                "early_stopping_rounds": 50,
                "eval_metric": "aucpr",
                "feature_count": len(numeric_feature_cols),
            }
        )

        model = train_model(X_train, y_train, X_test, y_test, scale_pos_weight)
        optimal_threshold = find_optimal_threshold(
            model,
            X_test,
            y_test,
            min_recall=0.95,
        )
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            threshold=optimal_threshold,
        )

        # Log the selected operating threshold to MLflow.
        mlflow.log_param("decision_threshold", round(optimal_threshold, 4))
        feature_importance = get_feature_importance(
            model,
            numeric_feature_cols,
        )
        save_artifacts(model, metrics, feature_importance, models_dir)

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

        # Log artifacts to MLflow
        out = Path(models_dir)
        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact(str(out / METRICS_FILENAME))
        mlflow.log_artifact(str(out / FEATURE_IMPORTANCE_FILENAME))

        logger.info(
            "MLflow run logged — run_id: %s",
            mlflow.active_run().info.run_id,
        )

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(
        "PR-AUC: %.4f | F1: %.4f | ROC-AUC: %.4f",
        metrics["pr_auc"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    _processed = str(project_root / "data" / "processed")
    _models = str(project_root / "models")
    run_training(_processed, _models)
