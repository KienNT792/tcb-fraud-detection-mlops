from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from infrastructure.pipeline import PIPELINE_CONFIG

try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
except Exception:
    DataDriftPreset = None
    Report = None

from monitoring.alerts import send_webhook


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_basic_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict[str, Any]:
    common_cols = [col for col in reference_df.columns if col in current_df.columns]
    numeric_cols = [
        col for col in common_cols
        if pd.api.types.is_numeric_dtype(reference_df[col]) and pd.api.types.is_numeric_dtype(current_df[col])
    ]

    drifted_features: list[dict[str, Any]] = []
    for col in numeric_cols:
        ref_mean = float(reference_df[col].fillna(0).mean())
        cur_mean = float(current_df[col].fillna(0).mean())
        scale = max(abs(ref_mean), 1.0)
        delta_ratio = abs(cur_mean - ref_mean) / scale
        if delta_ratio >= 0.2:
            drifted_features.append(
                {
                    "feature": col,
                    "reference_mean": round(ref_mean, 6),
                    "current_mean": round(cur_mean, 6),
                    "delta_ratio": round(delta_ratio, 6),
                }
            )

    drift_share = round(len(drifted_features) / max(len(numeric_cols), 1), 6)
    return {
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "numeric_features_checked": len(numeric_cols),
        "drifted_features": drifted_features,
        "drift_share": drift_share,
        "drift_detected": drift_share > 0.2,
    }


def maybe_write_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
) -> str | None:
    if Report is None or DataDriftPreset is None:
        return None

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    html_path = output_dir / "evidently_drift_report.html"
    report.save_html(str(html_path))
    return str(html_path)


def run_monitoring(
    reference_path: str,
    current_path: str,
    output_dir: str,
    webhook_url: str | None = None,
) -> dict[str, Any]:
    ref_path = Path(reference_path)
    cur_path = Path(current_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_df = load_frame(ref_path)
    current_df = load_frame(cur_path)

    summary = compute_basic_drift(reference_df, current_df)
    evidently_report = maybe_write_evidently_report(reference_df, current_df, out_dir)
    summary["reference_path"] = str(ref_path)
    summary["current_path"] = str(cur_path)
    summary["evidently_report"] = evidently_report

    summary_path = out_dir / "drift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if summary["drift_detected"] and webhook_url:
        send_webhook(
            webhook_url,
            {
                "event": "data_drift_detected",
                "drift_share": summary["drift_share"],
                "output": str(summary_path),
            },
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data drift checks for the fraud detection pipeline.")
    parser.add_argument("--reference-path", default=str(PIPELINE_CONFIG.paths.processed_dir / "train.parquet"))
    parser.add_argument("--current-path", default=str(PIPELINE_CONFIG.paths.processed_dir / "test.parquet"))
    parser.add_argument("--output-dir", default=str(PIPELINE_CONFIG.paths.drift_output_dir))
    parser.add_argument("--webhook-url", default=os.getenv("ALERT_WEBHOOK_URL"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_monitoring(
        reference_path=args.reference_path,
        current_path=args.current_path,
        output_dir=args.output_dir,
        webhook_url=args.webhook_url,
    )


if __name__ == "__main__":
    main()
