from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/CI
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {"transaction_id", "customer_id", "timestamp", "is_fraud"}
)

_REGRESSION_TOLERANCE: dict[str, float] = {
    "pr_auc":    0.02,   # PR-AUC không được giảm quá 2 điểm
    "f1":        0.03,   # F1 không được giảm quá 3 điểm
    "recall":    0.02,   # Recall không được giảm quá 2 điểm
}

def load_artifacts(
    models_dir: str,
    processed_dir: str,
) -> tuple[XGBClassifier, pd.DataFrame, pd.Series, list[str], dict[str, Any]]:

    m_path = Path(models_dir)
    p_path = Path(processed_dir)

    required = {
        "model":    m_path / "xgb_fraud_model.joblib",
        "metrics":  m_path / "metrics.json",
        "test":     p_path / "test.parquet",
        "features": p_path / "features.json",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact '{name}' not found: {path}\n"
                "Run preprocess.py then train.py first."
            )

    logger.info("Loading artifacts from: %s + %s", m_path, p_path)

    model: XGBClassifier = joblib.load(required["model"])

    test = pd.read_parquet(required["test"])

    with open(required["features"], encoding="utf-8") as fh:
        meta = json.load(fh)
    feature_cols: list[str] = meta["features"]

    with open(required["metrics"], encoding="utf-8") as fh:
        baseline_metrics: dict[str, Any] = json.load(fh)

    # Guard: verify all features present
    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        raise ValueError(f"Features missing from test set: {missing}")

    # Keep only numeric features (mirrors filter_numeric_features in train.py)
    X_test = test[feature_cols].select_dtypes(include=[np.number, bool]).copy()
    y_test = test["is_fraud"].astype(int)
    numeric_feature_cols = X_test.columns.tolist()

    logger.info(
        "Artifacts loaded — X_test: %s | Fraud: %d / %d (%.2f%%)",
        X_test.shape,
        y_test.sum(),
        len(y_test),
        y_test.mean() * 100,
    )
    return model, X_test, y_test, numeric_feature_cols, baseline_metrics


def evaluate_threshold(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_recall: float = 0.95,
    max_threshold: float = 0.70,
    output_dir: Path | None = None,
) -> dict[str, Any]:

    y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]

    # Full PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # ROC curve for ROC-AUC
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc  = float(average_precision_score(y_test, y_prob))

    # Find optimal threshold: recall >= min_recall AND threshold <= max_threshold
    valid = (recalls[:-1] >= min_recall) & (thresholds <= max_threshold)
    if not valid.any():
        logger.warning(
            "No threshold satisfies recall>=%.2f AND threshold<=%.2f — "
            "relaxing max_threshold constraint.",
            min_recall, max_threshold,
        )
        valid = recalls[:-1] >= min_recall

    if not valid.any():
        best_idx = int(recalls[:-1].argmax())
    else:
        best_idx = int(precisions[:-1][valid].argmax())
        # Map back to global index
        best_idx = int(np.where(valid)[0][precisions[:-1][valid].argmax()])

    opt_threshold = float(thresholds[best_idx])
    y_pred_opt = (y_prob >= opt_threshold).astype(int)

    opt_metrics = {
        "threshold":  round(opt_threshold, 4),
        "precision":  round(float(precision_score(y_test, y_pred_opt, zero_division=0)), 6),
        "recall":     round(float(recall_score(y_test, y_pred_opt, zero_division=0)), 6),
        "f1":         round(float(f1_score(y_test, y_pred_opt, zero_division=0)), 6),
        "pr_auc":     round(pr_auc, 6),
        "roc_auc":    round(roc_auc, 6),
        "confusion_matrix": confusion_matrix(y_test, y_pred_opt).tolist(),
    }

    tn, fp, fn, tp = (
        opt_metrics["confusion_matrix"][0][0],
        opt_metrics["confusion_matrix"][0][1],
        opt_metrics["confusion_matrix"][1][0],
        opt_metrics["confusion_matrix"][1][1],
    )

    logger.info("=" * 60)
    logger.info("THRESHOLD ANALYSIS")
    logger.info("=" * 60)
    logger.info("  Optimal threshold : %.4f", opt_threshold)
    logger.info("  Precision         : %.4f", opt_metrics["precision"])
    logger.info("  Recall            : %.4f", opt_metrics["recall"])
    logger.info("  F1                : %.4f", opt_metrics["f1"])
    logger.info("  PR-AUC            : %.4f", pr_auc)
    logger.info("  ROC-AUC           : %.4f", roc_auc)
    logger.info("  TN=%d | FP=%d | FN=%d | TP=%d", tn, fp, fn, tp)
    logger.info("=" * 60)

    # --- Plot PR curve ---
    if output_dir is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PR curve
        ax = axes[0]
        ax.plot(recalls, precisions, color="#1D9E75", linewidth=2, label=f"PR curve (AUC={pr_auc:.4f})")
        ax.scatter(
            opt_metrics["recall"], opt_metrics["precision"],
            color="#D85A30", s=80, zorder=5,
            label=f"Optimal @ threshold={opt_threshold:.3f}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Threshold sweep — F1, Precision, Recall vs threshold
        ax2 = axes[1]
        sweep_thresholds = thresholds
        sweep_precision  = precisions[:-1]
        sweep_recall     = recalls[:-1]
        sweep_f1 = 2 * sweep_precision * sweep_recall / np.where(
            (sweep_precision + sweep_recall) == 0, 1,
            sweep_precision + sweep_recall
        )
        ax2.plot(sweep_thresholds, sweep_precision, label="Precision", color="#185FA5")
        ax2.plot(sweep_thresholds, sweep_recall,    label="Recall",    color="#1D9E75")
        ax2.plot(sweep_thresholds, sweep_f1,        label="F1",        color="#BA7517", linestyle="--")
        ax2.axvline(opt_threshold, color="#D85A30", linestyle=":", linewidth=1.5,
                    label=f"Optimal={opt_threshold:.3f}")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Score")
        ax2.set_title("Metrics vs Decision Threshold")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        save_path = output_dir / "pr_curve.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("PR curve saved → %s", save_path)

    return opt_metrics

def evaluate_segments(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    threshold: float,
    segment_col: str = "segment_encoded",
) -> pd.DataFrame:

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    results = []

    # Overall row first
    results.append({
        "segment":   "ALL",
        "n_samples": len(y_test),
        "n_fraud":   int(y_test.sum()),
        "fraud_rate": round(float(y_test.mean() * 100), 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "pr_auc":    round(float(average_precision_score(y_test, y_prob)), 4),
    })

    if segment_col not in test_df.columns:
        logger.warning("Column '%s' not found — skipping segment breakdown.", segment_col)
        return pd.DataFrame(results)

    segments = sorted(test_df[segment_col].unique())
    for seg in segments:
        mask = test_df[segment_col].values == seg
        if mask.sum() == 0:
            continue
        y_t = y_test.values[mask]
        y_p = y_pred[mask]
        y_pb = y_prob[mask]

        n_fraud = int(y_t.sum())
        row: dict[str, Any] = {
            "segment":    int(seg),
            "n_samples":  int(mask.sum()),
            "n_fraud":    n_fraud,
            "fraud_rate": round(float(y_t.mean() * 100), 2),
            "precision":  round(float(precision_score(y_t, y_p, zero_division=0)), 4),
            "recall":     round(float(recall_score(y_t, y_p, zero_division=0)), 4),
            "f1":         round(float(f1_score(y_t, y_p, zero_division=0)), 4),
            "pr_auc":     round(float(average_precision_score(y_t, y_pb)) if n_fraud > 0 else 0.0, 4),
        }
        results.append(row)

    df_seg = pd.DataFrame(results)

    logger.info("=" * 60)
    logger.info("SEGMENT EVALUATION (threshold=%.4f)", threshold)
    logger.info("=" * 60)
    for _, row in df_seg.iterrows():
        logger.info(
            "  %-8s | n=%6d | fraud=%4d (%.2f%%) | "
            "P=%.4f | R=%.4f | F1=%.4f",
            str(row["segment"]),
            row["n_samples"],
            row["n_fraud"],
            row["fraud_rate"],
            row["precision"],
            row["recall"],
            row["f1"],
        )
    logger.info("=" * 60)

    return df_seg.sort_values("f1", ascending=True).reset_index(drop=True)

def explain_shap(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    sample_size: int = 2000,
    top_n: int = 20,
) -> None:

    logger.info("Computing SHAP values (sample_size=%d)…", sample_size)

    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # --- Summary (beeswarm) plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, sample,
        feature_names=feature_cols,
        max_display=top_n,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Importance — Beeswarm", fontsize=13, pad=12)
    plt.tight_layout()
    summary_path = output_dir / "shap_summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP summary saved → %s", summary_path)

    # --- Waterfall plot for highest-risk prediction ---
    y_prob_sample = model.predict_proba(sample)[:, 1]
    highest_risk_idx = int(y_prob_sample.argmax())

    explanation = shap.Explanation(
        values        = shap_values[highest_risk_idx],
        base_values   = explainer.expected_value,
        data          = sample.iloc[highest_risk_idx].values,
        feature_names = feature_cols,
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — Highest Risk Prediction (score={y_prob_sample[highest_risk_idx]:.4f})",
        fontsize=12, pad=12,
    )
    plt.tight_layout()
    waterfall_path = output_dir / "shap_waterfall.png"
    fig2.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("SHAP waterfall saved → %s", waterfall_path)

def compare_baseline(
    current_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    tolerance: dict[str, float] | None = None,
) -> dict[str, Any]:

    if tolerance is None:
        tolerance = _REGRESSION_TOLERANCE

    comparison: dict[str, Any] = {"status": "PASS", "metrics": {}}

    for metric, tol in tolerance.items():
        current_val  = current_metrics.get(metric, 0.0)
        baseline_val = baseline_metrics.get(metric, 0.0)
        delta        = current_val - baseline_val
        passed       = delta >= -tol

        comparison["metrics"][metric] = {
            "baseline": round(float(baseline_val), 6),
            "current":  round(float(current_val), 6),
            "delta":    round(float(delta), 6),
            "tolerance": tol,
            "status":   "PASS" if passed else "FAIL",
        }

        if not passed:
            comparison["status"] = "FAIL"

    logger.info("=" * 60)
    logger.info("BASELINE COMPARISON — Overall: %s", comparison["status"])
    logger.info("=" * 60)
    for metric, info in comparison["metrics"].items():
        flag = "✓" if info["status"] == "PASS" else "✗"
        logger.info(
            "  %s %-10s | baseline=%.4f | current=%.4f | delta=%+.4f | tol=%.2f",
            flag,
            metric,
            info["baseline"],
            info["current"],
            info["delta"],
            info["tolerance"],
        )
    logger.info("=" * 60)

    return comparison


def save_evaluation_report(
    threshold_metrics: dict[str, Any],
    segment_report: pd.DataFrame,
    comparison: dict[str, Any],
    output_dir: str,
) -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "evaluated_at":    datetime.now(tz=timezone.utc).isoformat(),
        "threshold_metrics": threshold_metrics,
        "baseline_comparison": comparison,
        "overall_status":  comparison["status"],
    }

    report_path = out / "evaluation.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Evaluation report saved → %s", report_path)

    seg_path = out / "segment_report.csv"
    segment_report.to_csv(seg_path, index=False)
    logger.info("Segment report saved    → %s", seg_path)


def run_evaluation(
    models_dir: str = "models",
    processed_dir: str = "data/processed",
    evaluation_dir: str = "models/evaluation",
    min_recall: float = 0.95,
    max_threshold: float = 0.70,
) -> dict[str, Any]:

    logger.info("=" * 60)
    logger.info("TCB FRAUD DETECTION — EVALUATION PIPELINE START")
    logger.info("Models dir:     %s", models_dir)
    logger.info("Processed dir:  %s", processed_dir)
    logger.info("Evaluation dir: %s", evaluation_dir)
    logger.info("=" * 60)

    out = Path(evaluation_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1 — Load
    model, X_test, y_test, feature_cols, baseline_metrics = load_artifacts(
        models_dir, processed_dir
    )

    # Load full test DF for segment analysis (needs segment_encoded column)
    test_df = pd.read_parquet(Path(processed_dir) / "test.parquet")

    with mlflow.start_run(run_name="evaluation"):

        # Step 2 — Threshold analysis
        threshold_metrics = evaluate_threshold(
            model, X_test, y_test,
            min_recall=min_recall,
            max_threshold=max_threshold,
            output_dir=out,
        )
        optimal_threshold = threshold_metrics["threshold"]

        # Step 3 — Segment breakdown
        segment_report = evaluate_segments(
            model, X_test, y_test, test_df,
            threshold=optimal_threshold,
        )

        # Step 4 — SHAP
        explain_shap(model, X_test, feature_cols, output_dir=out)

        # Step 5 — Baseline comparison
        comparison = compare_baseline(threshold_metrics, baseline_metrics)

        # Step 6 — Save report
        save_evaluation_report(threshold_metrics, segment_report, comparison, evaluation_dir)

        # Step 7 — MLflow logging
        mlflow.log_metrics({
            "eval_pr_auc":    threshold_metrics["pr_auc"],
            "eval_roc_auc":   threshold_metrics["roc_auc"],
            "eval_f1":        threshold_metrics["f1"],
            "eval_precision": threshold_metrics["precision"],
            "eval_recall":    threshold_metrics["recall"],
            "eval_threshold": optimal_threshold,
        })
        mlflow.set_tag("evaluation_status", comparison["status"])
        mlflow.set_tag("min_recall_target", str(min_recall))

        for artifact_file in [
            "evaluation.json",
            "segment_report.csv",
            "pr_curve.png",
            "shap_summary.png",
            "shap_waterfall.png",
        ]:
            artifact_path = out / artifact_file
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        logger.info("MLflow evaluation run logged — run_id: %s", mlflow.active_run().info.run_id)

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE COMPLETE — Status: %s", comparison["status"])
    logger.info(
        "PR-AUC: %.4f | F1: %.4f | Recall: %.4f | Threshold: %.4f",
        threshold_metrics["pr_auc"],
        threshold_metrics["f1"],
        threshold_metrics["recall"],
        optimal_threshold,
    )
    logger.info("=" * 60)

    return {
        "evaluated_at":      datetime.now(tz=timezone.utc).isoformat(),
        "threshold_metrics": threshold_metrics,
        "baseline_comparison": comparison,
        "overall_status":    comparison["status"],
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    _models    = str(project_root / "models")
    _processed = str(project_root / "data" / "processed")
    _eval_dir  = str(project_root / "models" / "evaluation")
    run_evaluation(_models, _processed, _eval_dir)