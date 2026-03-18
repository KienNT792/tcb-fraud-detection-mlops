from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.pipeline import PIPELINE_CONFIG
from ml_pipeline.src.preprocess import (
    load_dataset,
    run_preprocessing,
    validate_schema,
)


def make_demo_raw_df(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2026-01-01")
    timestamps = pd.date_range(start, periods=n, freq="15min")
    tiers = rng.choice(["PRIVATE", "PRIORITY", "INSPIRE", "MASS"], size=n)
    cvv_vals = rng.choice(["Y", "N", "N/A"], size=n, p=[0.8, 0.1, 0.1])
    is_3ds = rng.choice(["Y", "N", "N/A"], size=n, p=[0.75, 0.15, 0.1])
    statuses = rng.choice(["APPROVED", "DECLINED"], size=n, p=[0.92, 0.08])
    is_fraud = (rng.random(n) < 0.03).astype(int)
    customer_ids = [f"TCB_CUST_{idx:05d}" for idx in range(200)]

    return pd.DataFrame(
        {
            "transaction_id": [f"txn-{idx:06d}" for idx in range(n)],
            "timestamp": timestamps.astype(str),
            "hour_of_day": timestamps.hour,
            "is_weekend": timestamps.dayofweek.isin([5, 6]).astype(int),
            "customer_id": rng.choice(customer_ids, size=n),
            "card_bin": rng.integers(400000, 520000, size=n),
            "card_type": rng.choice(["CREDIT", "DEBIT"], size=n),
            "card_tier": rng.choice(["PLATINUM", "SPARK", "EVERYDAY"], size=n),
            "account_age_days": rng.integers(1, 3000, size=n),
            "amount": np.abs(rng.normal(500_000, 1_000_000, size=n)).clip(1_000),
            "currency": "VND",
            "merchant_name": rng.choice(["WinMart", "Shopee", "Grab"], size=n),
            "mcc_code": rng.integers(4000, 9000, size=n),
            "merchant_category": rng.choice(["Groceries", "Electronics", "Travel"], size=n),
            "merchant_city": rng.choice(["Ha Noi", "TP Ho Chi Minh", "Da Nang"], size=n),
            "merchant_country": rng.choice(["VN", "US", "SG"], size=n),
            "device_type": rng.choice(["Mobile", "Desktop", "POS"], size=n),
            "os": rng.choice(["Android", "iOS", "Windows"], size=n),
            "ip_country": rng.choice(["VN", "US", "SG"], size=n),
            "distance_from_home_km": rng.exponential(200, size=n),
            "cvv_match": cvv_vals,
            "is_3d_secure": is_3ds,
            "transaction_status": statuses,
            "tx_count_last_1h": rng.integers(1, 10, size=n),
            "tx_count_last_24h": rng.integers(1, 60, size=n),
            "time_since_last_tx_min": rng.exponential(2000, size=n),
            "avg_amount_last_30d": np.abs(rng.normal(2_000_000, 1_000_000, size=n)),
            "amount_ratio_vs_avg": rng.uniform(0.01, 5.0, size=n),
            "is_new_device": rng.integers(0, 2, size=n),
            "is_new_merchant": rng.integers(0, 2, size=n),
            "customer_tier": tiers,
            "is_fraud": is_fraud,
        }
    )


def materialize_feature_ready(processed_dir: Path, feature_ready_dir: Path) -> None:
    feature_ready_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        "features.json",
        "customer_stats.parquet",
        "segment_label_map.json",
        "amount_median_train.json",
        "categorical_maps.json",
    ]
    for filename in files_to_copy:
        src = processed_dir / filename
        dst = feature_ready_dir / filename
        if src.exists():
            shutil.copy2(src, dst)

    manifest = {
        "source_processed_dir": str(processed_dir),
        "feature_ready_dir": str(feature_ready_dir),
        "files": files_to_copy,
    }
    with open(feature_ready_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def bootstrap(
    raw_path: Path,
    validated_dir: Path,
    processed_dir: Path,
    feature_ready_dir: Path,
    if_missing: bool,
) -> None:
    if if_missing and (processed_dir / "features.json").exists():
        return

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    validated_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    feature_ready_dir.mkdir(parents=True, exist_ok=True)

    demo_df = make_demo_raw_df()
    demo_df.to_csv(raw_path, index=False)

    loaded = load_dataset(str(raw_path))
    validate_schema(loaded)
    loaded.to_parquet(validated_dir / "validated_transactions.parquet", index=False)

    run_preprocessing(str(raw_path), str(processed_dir))
    materialize_feature_ready(processed_dir, feature_ready_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap demo raw/processed artifacts for local runs.")
    parser.add_argument("--raw-path", default=str(PIPELINE_CONFIG.paths.demo_raw_path))
    parser.add_argument("--validated-dir", default=str(PIPELINE_CONFIG.paths.validated_dir))
    parser.add_argument("--processed-dir", default=str(PIPELINE_CONFIG.paths.processed_dir))
    parser.add_argument("--feature-ready-dir", default=str(PIPELINE_CONFIG.paths.feature_ready_dir))
    parser.add_argument("--if-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap(
        raw_path=Path(args.raw_path),
        validated_dir=Path(args.validated_dir),
        processed_dir=Path(args.processed_dir),
        feature_ready_dir=Path(args.feature_ready_dir),
        if_missing=args.if_missing,
    )


if __name__ == "__main__":
    main()
