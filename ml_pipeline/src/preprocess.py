"""
TCB Fraud Detection — Data Preprocessing Pipeline.

This module implements the full preprocessing pipeline for the TCB credit card
fraud detection dataset. It follows MLOps best practices: schema validation,
data-leakage-free feature engineering via a fit/transform split, time-based
train/test splitting, and feature metadata persistence.

Pipeline order
--------------
load_dataset → validate_schema → clean_data → split_dataset
→ fit_feature_generators(train) → transform_features(train)
→ transform_features(test) → save_processed_data → save_feature_metadata

Usage
-----
    python -m ml_pipeline.src.preprocess
    # OR
    from ml_pipeline.src.preprocess import run_preprocessing
    train, test = run_preprocessing("data/raw/tcb_credit_fraud_dataset.csv",
                                    "data/processed")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
# Constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: list[str] = [
    "transaction_id",
    "timestamp",
    "customer_id",
    "amount",
    "customer_tier",
    "is_fraud",
]

# Mapping for Y/N/N/A fields → int
_YN_MAP: dict[str, int] = {"Y": 1, "N": 0, "N/A": 0}

# Transaction status → int
_STATUS_MAP: dict[str, int] = {"APPROVED": 1, "DECLINED": 0}

# Raw behavioural features present in the dataset — must be preserved through cleaning
BEHAVIOURAL_FEATURES: list[str] = [
    "tx_count_last_1h",
    "tx_count_last_24h",
    "time_since_last_tx_min",
    "avg_amount_last_30d",
    "amount_ratio_vs_avg",
    "distance_from_home_km",
    "is_new_device",
    "is_new_merchant",
    "cvv_match",
    "is_3d_secure",
]

# Derived feature names added by transform_features()
DERIVED_FEATURES: list[str] = [
    "transaction_hour",
    "transaction_day_of_week",
    "is_night_transaction",
    "amount_log",
    "segment_encoded",
    "customer_tx_count",
    "customer_avg_amount",
]

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
# Columns excluded from the feature list when saving metadata
_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {"transaction_id", "customer_id", "timestamp", "is_fraud"}
)


def analyze_dataset(path: str) -> None:
    """Load the raw CSV and log a structured analysis report.

    Inspects the dataset for basic statistics, class distribution, segment
    distribution, missing values, and amount outliers. Does not modify the file.

    Args:
        path: Absolute or relative path to the raw CSV file.

    Raises:
        FileNotFoundError: If the CSV file does not exist at *path*.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("=" * 60)
    logger.info("DATASET ANALYSIS REPORT")
    logger.info("Path: %s", path)
    logger.info("=" * 60)

    df = pd.read_csv(path, low_memory=False)

    # Basic shape
    logger.info("Rows: %d | Columns: %d", *df.shape)

    # Missing values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if cols_with_nulls.empty:
        logger.info("Missing values: NONE")
    else:
        logger.info("Missing values detected:\n%s", cols_with_nulls.to_string())

    # Class distribution
    if "is_fraud" in df.columns:
        fraud_counts = df["is_fraud"].value_counts()
        fraud_rate = df["is_fraud"].mean() * 100
        logger.info(
            "Class distribution — Legit: %d | Fraud: %d | Fraud rate: %.2f%%",
            fraud_counts.get(0, 0),
            fraud_counts.get(1, 0),
            fraud_rate,
        )
    else:
        logger.warning("Column 'is_fraud' not found in dataset.")

    # Segment distribution
    if "customer_tier" in df.columns:
        logger.info(
            "Customer tier distribution:\n%s",
            df["customer_tier"].value_counts().to_string(),
        )

    # Amount outlier detection
    if "amount" in df.columns:
        amt = pd.to_numeric(df["amount"], errors="coerce")
        p99 = amt.quantile(0.99)
        p999 = amt.quantile(0.999)
        extreme = (amt > p99).sum()
        logger.info(
            "Amount — min: %s | median: %s | 99th pct: %s | 99.9th pct: %s | "
            "rows above 99th pct: %d",
            f"{amt.min():,.0f}",
            f"{amt.median():,.0f}",
            f"{p99:,.0f}",
            f"{p999:,.0f}",
            extreme,
        )

    # Timestamp format check
    if "timestamp" in df.columns:
        sample = df["timestamp"].dropna().iloc[:5].tolist()
        logger.info("Timestamp samples: %s", sample)

    # CVV / 3DS value distribution
    for col in ("cvv_match", "is_3d_secure"):
        if col in df.columns:
            logger.info(
                "%s unique values: %s",
                col,
                df[col].value_counts().to_dict(),
            )

    # Dtypes summary
    logger.info("Column dtypes:\n%s", df.dtypes.to_string())
    logger.info("=" * 60)


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset and apply initial type casting.

    Preserves all 33 raw columns including behavioural features. Does not
    apply any cleaning or feature engineering.

    Args:
        path: File path to the raw CSV.

    Returns:
        DataFrame with correctly typed columns.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    logger.info("Loading dataset from: %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Numeric casts
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").astype("Int64")

    # Categorical
    df["customer_tier"] = df["customer_tier"].astype("category")
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["customer_id"] = df["customer_id"].astype(str)

    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the DataFrame conforms to the expected pipeline schema.

    Checks required column presence, timestamp type, numeric amount, and
    binary is_fraud values. This validation gate prevents silent downstream
    failures caused by schema drift.

    Args:
        df: DataFrame to validate (typically the output of *load_dataset*).

    Raises:
        ValueError: If any schema constraint is violated.
    """
    errors: list[str] = []

    # 1. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. timestamp — must be datetime64 and have no nulls
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            errors.append(
                "Column 'timestamp' is not datetime64. "
                "Call load_dataset() before validate_schema()."
            )
        elif df["timestamp"].isna().any():
            null_ts = int(df["timestamp"].isna().sum())
            errors.append(f"Column 'timestamp' has {null_ts} null value(s).")

    # 3. amount — must be numeric and non-negative
    if "amount" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["amount"]):
            errors.append("Column 'amount' is not numeric.")
        elif (df["amount"] < 0).any():
            neg_count = int((df["amount"] < 0).sum())
            errors.append(f"Column 'amount' has {neg_count} negative value(s).")

    # 4. is_fraud — must be binary {0, 1}
    if "is_fraud" in df.columns:
        unique_vals = set(df["is_fraud"].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            errors.append(
                f"Column 'is_fraud' contains unexpected values: {unique_vals - {0, 1}}"
            )

    # 5. transaction_id — must have no duplicates
    if "transaction_id" in df.columns:
        n_dupes = int(df["transaction_id"].duplicated().sum())
        if n_dupes:
            errors.append(
                f"Column 'transaction_id' has {n_dupes} duplicate value(s)."
            )

    # 6. customer_id — must not be null
    if "customer_id" in df.columns:
        null_cust = int(df["customer_id"].isna().sum())
        if null_cust:
            errors.append(f"Column 'customer_id' has {null_cust} null value(s).")

    if errors:
        msg = "Schema validation failed:\n  " + "\n  ".join(errors)
        raise ValueError(msg)

    logger.info("Schema validation passed (%d rows, %d columns).", *df.shape)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values according to domain-specific rules.

    Rules applied:
    - ``os``: fill nulls with ``"UNKNOWN"``.
    - ``is_3d_secure``: fill nulls with ``"N"`` (conservative default).
    - Numeric behavioural features: fill nulls with ``0``.

    Args:
        df: DataFrame to impute (in-place copy).

    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()
    total_nulls_before = int(df.isnull().sum().sum())

    if "os" in df.columns:
        df["os"] = df["os"].fillna("UNKNOWN")

    if "is_3d_secure" in df.columns:
        df["is_3d_secure"] = df["is_3d_secure"].fillna("N")

    for col in BEHAVIOURAL_FEATURES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)

    total_nulls_after = int(df.isnull().sum().sum())
    logger.info(
        "Missing value imputation — nulls: %d → %d (resolved: %d)",
        total_nulls_before,
        total_nulls_after,
        total_nulls_before - total_nulls_after,
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data cleaning transformations to the raw DataFrame.

    Operations: duplicate removal, missing-value imputation, Y/N/N/A encoding,
    transaction status encoding, and binary column casting.

    Args:
        df: Raw DataFrame from *load_dataset*.

    Returns:
        Cleaned DataFrame.
    """
    rows_before = len(df)
    fraud_rate_before = float(df["is_fraud"].mean())

    # Impute missing values via dedicated helper
    df = handle_missing_values(df)

    # Remove duplicate transactions
    df = df.drop_duplicates(subset=["transaction_id"]).copy()
    duplicates_removed = rows_before - len(df)
    logger.info(
        "Duplicates removed: %d (rows: %d → %d)",
        duplicates_removed,
        rows_before,
        len(df),
    )

    # Encode Y/N/N/A string fields to int8
    for col in ("cvv_match", "is_3d_secure"):
        if col in df.columns:
            df[col] = df[col].map(_YN_MAP).fillna(0).astype("int8")

    # Encode transaction status
    if "transaction_status" in df.columns:
        df["transaction_status"] = (
            df["transaction_status"].map(_STATUS_MAP).fillna(0).astype("int8")
        )

    # Ensure is_fraud is a plain Python int
    df["is_fraud"] = df["is_fraud"].fillna(0).astype(int)

    logger.info(
        "Cleaning complete — Rows: %d → %d | Fraud rate: %.2f%% → %.2f%%",
        rows_before,
        len(df),
        fraud_rate_before * 100,
        float(df["is_fraud"].mean()) * 100,
    )
    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically into train and test sets.

    Sorts by timestamp ascending, then slices the first *train_ratio* fraction
    as training data and the remainder as test data. This preserves temporal
    ordering and prevents future-data leakage.

    Feature engineering is intentionally performed AFTER this split so that
    customer-level aggregates can be fit on training data only.

    Args:
        df: Cleaned DataFrame with a valid 'timestamp' column.
        train_ratio: Fraction of rows to use for training. Defaults to 0.8.

    Returns:
        A tuple ``(train_df, test_df)``.

    Raises:
        ValueError: If *train_ratio* is not in the range (0, 1).
    """
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)

    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()

    logger.info(
        "Time-based split — Train: %d rows (fraud %.2f%%) | "
        "Test: %d rows (fraud %.2f%%)",
        len(train),
        train["is_fraud"].mean() * 100,
        len(test),
        test["is_fraud"].mean() * 100,
    )
    logger.info(
        "Train window: %s → %s",
        train["timestamp"].min(),
        train["timestamp"].max(),
    )
    logger.info(
        "Test window:  %s → %s",
        test["timestamp"].min(),
        test["timestamp"].max(),
    )

    return train, test


def fit_feature_generators(train_df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Fitting feature generators on training data (%d rows)…", len(train_df))

    # Customer-level aggregates (fit on TRAIN ONLY)
    customer_stats: pd.DataFrame = (
        train_df.groupby("customer_id", observed=True)["amount"]
        .agg(count="count", mean="mean")
        .rename(columns={"count": "customer_tx_count", "mean": "customer_avg_amount"})
    )

    # Segment encoding map (fit on TRAIN categories)
    tiers = sorted(train_df["customer_tier"].dropna().unique().tolist())
    segment_label_map: dict[str, int] = {tier: idx for idx, tier in enumerate(tiers)}

    # Fallback imputation value for unseen customers
    amount_median_train: float = float(train_df["amount"].median())

    # Categorical encoding maps — fit on TRAIN ONLY
    categorical_maps: dict[str, dict[str, int]] = {}
    for col in LOW_CARDINALITY_COLS:
        if col in train_df.columns:
            # Lấy unique values từ train, sort để reproducible
            unique_vals = sorted(train_df[col].dropna().astype(str).unique().tolist())
            categorical_maps[col] = {val: idx for idx, val in enumerate(unique_vals)}
            logger.info(
                "Categorical map fitted — %s: %d unique values",
                col, len(unique_vals),
            )
    logger.info(
        "Feature generators fitted — unique customers: %d | segment map: %s",
        len(customer_stats),
        segment_label_map,
    )

    return {
        "customer_stats": customer_stats,
        "segment_label_map": segment_label_map,
        "amount_median_train": amount_median_train,
        "categorical_maps": categorical_maps,
    }


def transform_features(
    df: pd.DataFrame,
    feature_state: dict[str, Any],
) -> pd.DataFrame:

    df = df.copy()

    customer_stats: pd.DataFrame = feature_state["customer_stats"]
    segment_label_map: dict[str, int] = feature_state["segment_label_map"]
    amount_median_train: float = feature_state["amount_median_train"]
    categorical_maps: dict[str, dict[str, int]] = feature_state.get("categorical_maps", {})

    # --- Timestamp guard ---
    if "timestamp" not in df.columns:
        raise KeyError("Column 'timestamp' is required for feature engineering.")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # --- Timestamp-derived features ---
    # Reuse raw 'hour_of_day' if present to avoid duplicate feature
    if "hour_of_day" in df.columns:
        df["transaction_hour"] = df["hour_of_day"]
        df = df.drop(columns=["hour_of_day"])
        logger.info("Reused 'hour_of_day' as 'transaction_hour' (duplicate avoided)")
    else:
        df["transaction_hour"] = df["timestamp"].dt.hour

    df["transaction_day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night_transaction"] = (
        (df["transaction_hour"] >= 23) | (df["transaction_hour"] <= 5)
    ).astype("int8")

    # --- Amount log transform (guard: ensure numeric, clip negatives) ---
    if not pd.api.types.is_numeric_dtype(df["amount"]):
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["amount_log"] = np.log1p(df["amount"].clip(lower=0))

    # --- Segment encoding via train-fitted label map ---
    df["segment_encoded"] = (
        df["customer_tier"]
        .astype(str)
        .map(segment_label_map)
        .fillna(-1)
        .astype(int)
    )

    # --- Customer aggregates: merge train-fitted stats, impute unseen ---
    df = df.merge(
        customer_stats.reset_index(),
        on="customer_id",
        how="left",
    )
    unseen = int(df["customer_tx_count"].isna().sum())
    df["customer_tx_count"] = df["customer_tx_count"].fillna(0).astype(int)
    df["customer_avg_amount"] = df["customer_avg_amount"].fillna(amount_median_train)

    # --- Categorical encoding via train-fitted maps ---
    for col, mapping in categorical_maps.items():
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .map(mapping)
                .fillna(-1)     # -1 = unseen category in test set
                .astype(int)
            )

    # --- Drop high-cardinality columns (too many unique values to encode) ---
    cols_to_drop = [c for c in HIGH_CARDINALITY_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped high-cardinality columns: %s", cols_to_drop)

    # --- Drop customer_tier — already encoded as segment_encoded above ---
    if "customer_tier" in df.columns:
        df = df.drop(columns=["customer_tier"])
        logger.info("Dropped 'customer_tier' — replaced by 'segment_encoded'")

    logger.info(
        "Features transformed — %d rows | %d columns | %d unseen customers imputed",
        len(df),
        df.shape[1],
        unseen,
    )
    return df


def save_processed_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str,
) -> None:

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train.parquet"
    test_path = out_path / "test.parquet"

    train.to_parquet(train_path, index=False, engine="pyarrow")
    test.to_parquet(test_path, index=False, engine="pyarrow")

    logger.info(
        "Saved train.parquet (%d rows) → %s", len(train), train_path
    )
    logger.info(
        "Saved test.parquet  (%d rows) → %s", len(test), test_path
    )


def save_feature_metadata(
    feature_cols: list[str],
    output_dir: str,
    target_col: str = "is_fraud",
) -> None:

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "target": target_col,
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    metadata_path = out_path / "features.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Feature metadata → %s (%d features)", metadata_path, len(feature_cols))


def save_feature_state(feature_state: dict[str, Any], output_dir: str) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Customer stats as Parquet
    stats: pd.DataFrame = feature_state["customer_stats"]
    stats_path = out_path / "customer_stats.parquet"
    stats.to_parquet(stats_path, engine="pyarrow")

    # Segment label map as JSON
    seg_path = out_path / "segment_label_map.json"
    with open(seg_path, "w", encoding="utf-8") as fh:
        json.dump(feature_state["segment_label_map"], fh, indent=2)

    # Median amount as JSON
    median_path = out_path / "amount_median_train.json"
    with open(median_path, "w", encoding="utf-8") as fh:
        json.dump({"amount_median_train": feature_state["amount_median_train"]}, fh)

    # Categorical maps as JSON
    cat_path = out_path / "categorical_maps.json"
    with open(cat_path, "w", encoding="utf-8") as fh:
        json.dump(feature_state.get("categorical_maps", {}), fh, indent=2)
    logger.info("Categorical maps saved → %s", cat_path)


def run_preprocessing(
    data_path: str = "data/raw/tcb_credit_fraud_dataset.csv",
    output_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
   
    logger.info("=" * 60)
    logger.info("TCB FRAUD DETECTION — PREPROCESSING PIPELINE START")
    logger.info("Input:  %s", data_path)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)
    analyze_dataset(data_path)

    df = load_dataset(data_path)
    validate_schema(df)
    df = clean_data(df)
    train, test = split_dataset(df, train_ratio=0.8)
    feature_state = fit_feature_generators(train)
    train = transform_features(train, feature_state)
    test = transform_features(test, feature_state)
    save_processed_data(train, test, output_dir)
    save_feature_state(feature_state, output_dir)
    feature_cols = [c for c in train.columns if c not in _NON_FEATURE_COLS]
    save_feature_metadata(feature_cols, output_dir)

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("Train: %s | Test: %s | Features: %d", train.shape, test.shape, len(feature_cols))
    logger.info("=" * 60)

    return train, test


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    _data_path = str(project_root / "data" / "raw" / "tcb_credit_fraud_dataset.csv")
    _output_dir = str(project_root / "data" / "processed")
    run_preprocessing(_data_path, _output_dir)
