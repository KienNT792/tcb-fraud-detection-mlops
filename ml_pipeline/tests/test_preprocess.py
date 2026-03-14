"""
Tests for the TCB Fraud Detection preprocessing pipeline.

All tests operate on a synthetic in-memory fixture that closely mirrors the
real dataset schema. This ensures fast execution without depending on the
presence of the actual raw CSV file.

Coverage targets: >80% of ml_pipeline/src/preprocess.py

Run with:
    pytest ml_pipeline/tests/test_preprocess.py -v
    pytest ml_pipeline/tests/test_preprocess.py --cov=ml_pipeline.src.preprocess
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the public API under test
# ---------------------------------------------------------------------------
from ml_pipeline.src.preprocess import (
    analyze_dataset,
    clean_data,
    fit_feature_generators,
    handle_missing_values,
    load_dataset,
    run_preprocessing,
    save_feature_metadata,
    save_feature_state,
    save_processed_data,
    split_dataset,
    transform_features,
    validate_schema,
    _NON_FEATURE_COLS,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_synthetic_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic DataFrame that mirrors the real TCB dataset schema.

    The fixture contains exactly the same column names and representative
    value ranges as the production CSV without depending on any file on disk.

    Args:
        n: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        Synthetic transaction DataFrame.
    """
    rng = np.random.default_rng(seed)

    # Chronological timestamps spanning 30 days
    start = pd.Timestamp("2026-01-01")
    timestamps = pd.date_range(start, periods=n, freq="15min")

    tiers = rng.choice(["PRIVATE", "PRIORITY", "INSPIRE", "MASS"], size=n)
    cvv_vals = rng.choice(["Y", "N", "N/A"], size=n, p=[0.8, 0.1, 0.1])
    is_3ds = rng.choice(["Y", "N", "N/A"], size=n, p=[0.75, 0.1, 0.15])
    statuses = rng.choice(["APPROVED", "DECLINED"], size=n, p=[0.9, 0.1])

    # Fraud label — ~3% rate
    is_fraud = (rng.random(n) < 0.03).astype(int)

    # Generate 50 unique customers so some appear in both train and test
    cust_ids = [f"TCB_CUST_{i:05d}" for i in range(50)]

    df = pd.DataFrame(
        {
            "transaction_id": [f"txn-{i:06d}" for i in range(n)],
            "timestamp": timestamps,
            "hour_of_day": timestamps.hour,
            "is_weekend": timestamps.dayofweek.isin([5, 6]).astype(int),
            "customer_id": rng.choice(cust_ids, size=n),
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
    return df


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Provide a synthetic raw DataFrame (pre-cleaning)."""
    return _make_synthetic_df(n=100)


@pytest.fixture()
def cleaned_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Provide a cleaned DataFrame."""
    return clean_data(raw_df.copy())


@pytest.fixture()
def split_dfs(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide train and test DataFrames from the cleaned fixture."""
    return split_dataset(cleaned_df, train_ratio=0.8)


@pytest.fixture()
def feature_state(split_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> dict:
    """Provide a feature_state dict fitted on the training split."""
    train, _ = split_dfs
    return fit_feature_generators(train)


@pytest.fixture()
def transformed_dfs(
    split_dfs: tuple[pd.DataFrame, pd.DataFrame],
    feature_state: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide fully transformed train and test DataFrames."""
    train, test = split_dfs
    return (
        transform_features(train, feature_state),
        transform_features(test, feature_state),
    )


@pytest.fixture()
def raw_csv_path(raw_df: pd.DataFrame, tmp_path: Path) -> str:
    """Write the synthetic DataFrame to a temporary CSV and return its path."""
    csv_path = tmp_path / "synthetic_dataset.csv"
    # Convert timestamp back to string so load_dataset can parse it
    df_to_save = raw_df.copy()
    df_to_save["timestamp"] = df_to_save["timestamp"].astype(str)
    df_to_save.to_csv(csv_path, index=False)
    return str(csv_path)


# ===========================================================================
# Tests — load_dataset
# ===========================================================================

def test_load_dataset_returns_dataframe(raw_csv_path: str) -> None:
    """load_dataset should return a non-empty pandas DataFrame."""
    df = load_dataset(raw_csv_path)
    assert isinstance(df, pd.DataFrame), "Expected pd.DataFrame"
    assert len(df) > 0, "DataFrame should not be empty"


def test_load_dataset_timestamp_parsed(raw_csv_path: str) -> None:
    """load_dataset should parse 'timestamp' into datetime64."""
    df = load_dataset(raw_csv_path)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), (
        "Column 'timestamp' should be datetime64 after load_dataset"
    )


def test_load_dataset_file_not_found() -> None:
    """load_dataset should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path/dataset.csv")


# ===========================================================================
# Tests — validate_schema
# ===========================================================================

def test_validate_schema_passes(raw_df: pd.DataFrame) -> None:
    """validate_schema should not raise for a well-formed DataFrame."""
    validate_schema(raw_df)  # should not raise


def test_validate_schema_fails_missing_col(raw_df: pd.DataFrame) -> None:
    """validate_schema should raise ValueError when a required column is absent."""
    df_missing = raw_df.drop(columns=["is_fraud"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df_missing)


def test_validate_schema_fails_non_binary_target(raw_df: pd.DataFrame) -> None:
    """validate_schema should raise ValueError if is_fraud has values outside {0,1}."""
    df_bad = raw_df.copy()
    df_bad["is_fraud"] = df_bad["is_fraud"].replace({0: 99})
    with pytest.raises(ValueError, match="is_fraud"):
        validate_schema(df_bad)


# ===========================================================================
# Tests — clean_data
# ===========================================================================

def test_fraud_column_exists_and_binary(raw_df: pd.DataFrame) -> None:
    """After clean_data, 'is_fraud' must exist and contain only {0, 1}."""
    cleaned = clean_data(raw_df)
    assert "is_fraud" in cleaned.columns, "'is_fraud' column missing after cleaning"
    unique_vals = set(cleaned["is_fraud"].unique())
    assert unique_vals.issubset({0, 1}), (
        f"is_fraud should only contain {{0, 1}}, got {unique_vals}"
    )


def test_clean_data_no_nulls_in_critical_cols(raw_df: pd.DataFrame) -> None:
    """After clean_data, critical columns must have no null values."""
    critical = ["transaction_id", "timestamp", "customer_id", "amount", "is_fraud"]
    cleaned = clean_data(raw_df)
    for col in critical:
        if col in cleaned.columns:
            assert cleaned[col].isnull().sum() == 0, (
                f"Column '{col}' has null values after clean_data"
            )


def test_clean_data_removes_duplicates(raw_df: pd.DataFrame) -> None:
    """clean_data should remove duplicated transaction_id rows."""
    # Introduce 5 exact duplicates
    dup_rows = raw_df.iloc[:5].copy()
    df_with_dups = pd.concat([raw_df, dup_rows], ignore_index=True)
    cleaned = clean_data(df_with_dups)
    assert len(cleaned) == len(raw_df), (
        f"Expected {len(raw_df)} rows after dedup, got {len(cleaned)}"
    )


def test_clean_data_encodes_yn_fields(raw_df: pd.DataFrame) -> None:
    """cvv_match and is_3d_secure should be integer after clean_data."""
    cleaned = clean_data(raw_df)
    for col in ("cvv_match", "is_3d_secure"):
        if col in cleaned.columns:
            assert pd.api.types.is_integer_dtype(cleaned[col]), (
                f"Column '{col}' should be integer after encoding, "
                f"got {cleaned[col].dtype}"
            )


# ===========================================================================
# Tests — handle_missing_values
# ===========================================================================

def test_handle_missing_values_resolves_nulls() -> None:
    """handle_missing_values should fill os, is_3d_secure, and numeric cols."""
    df = pd.DataFrame({
        "os": [None, "Windows"],
        "is_3d_secure": [None, "Y"],
        "tx_count_last_1h": [np.nan, 3.0],
        "distance_from_home_km": [np.nan, 10.0],
    })
    result = handle_missing_values(df)
    assert result["os"].iloc[0] == "UNKNOWN"
    assert result["is_3d_secure"].iloc[0] == "N"
    assert result["tx_count_last_1h"].iloc[0] == 0.0
    assert result["distance_from_home_km"].iloc[0] == 0.0


def test_handle_missing_values_does_not_mutate_original(raw_df: pd.DataFrame) -> None:
    """handle_missing_values should return a copy and not modify the input."""
    original_nulls = int(raw_df.isnull().sum().sum())
    _ = handle_missing_values(raw_df)
    assert int(raw_df.isnull().sum().sum()) == original_nulls, (
        "handle_missing_values must not mutate the original DataFrame"
    )


# ===========================================================================
# Tests — split_dataset
# ===========================================================================

def test_split_ratio(cleaned_df: pd.DataFrame) -> None:
    """Train set should contain approximately 80% of total rows."""
    train, test = split_dataset(cleaned_df, train_ratio=0.8)
    total = len(cleaned_df)
    actual_ratio = len(train) / total
    assert abs(actual_ratio - 0.8) < 0.02, (
        f"Train ratio expected ~0.80, got {actual_ratio:.3f}"
    )
    assert len(train) + len(test) == total, "Train + test must equal total rows"


def test_split_timestamp_integrity(cleaned_df: pd.DataFrame) -> None:
    """All train timestamps must be strictly before all test timestamps.

    This guards the chronological ordering assumption that prevents future
    data from leaking into the training set.
    """
    train, test = split_dataset(cleaned_df, train_ratio=0.8)
    assert train["timestamp"].max() < test["timestamp"].min(), (
        "Temporal integrity violated: max(train.timestamp) >= min(test.timestamp). "
        "This indicates the split is NOT chronological."
    )


def test_split_invalid_ratio(cleaned_df: pd.DataFrame) -> None:
    """split_dataset should raise ValueError for out-of-range train_ratio."""
    with pytest.raises(ValueError):
        split_dataset(cleaned_df, train_ratio=1.5)


# ===========================================================================
# Tests — fit_feature_generators + transform_features
# ===========================================================================

def test_feature_columns_exist(transformed_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Transformed DataFrames must contain all mandatory derived feature columns."""
    train, _ = transformed_dfs
    expected_cols = [
        "transaction_hour",
        "transaction_day_of_week",
        "is_night_transaction",
        "amount_log",
        "segment_encoded",
        "customer_tx_count",
        "customer_avg_amount",
    ]
    for col in expected_cols:
        assert col in train.columns, (
            f"Expected derived feature '{col}' missing from transformed train set"
        )


def test_behavioural_features_preserved(
    transformed_dfs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Raw behavioural features must be preserved after transform_features."""
    train, _ = transformed_dfs
    preserved = [
        "tx_count_last_1h",
        "avg_amount_last_30d",
        "amount_ratio_vs_avg",
        "is_new_device",
        "is_new_merchant",
        "distance_from_home_km",
    ]
    for col in preserved:
        assert col in train.columns, (
            f"Behavioural feature '{col}' was dropped — it must be preserved."
        )


def test_no_leakage_customer_aggregates(
    split_dfs: tuple[pd.DataFrame, pd.DataFrame],
    feature_state: dict,
) -> None:
    """Customer aggregates must be based solely on training set transactions.

    Approach: Compare the customer_tx_count in feature_state for a specific
    customer against the count computed from the FULL dataset. If there is
    leakage, the counts will be inflated by test-set transactions.
    """
    train, test = split_dfs

    # Find a customer that appears in BOTH splits
    common_customers = set(train["customer_id"]) & set(test["customer_id"])
    if not common_customers:
        pytest.skip("No common customers between train and test — cannot verify leakage")

    cust = next(iter(common_customers))
    train_count = (train["customer_id"] == cust).sum()
    full_count = (
        (train["customer_id"] == cust).sum()
        + (test["customer_id"] == cust).sum()
    )

    customer_stats: pd.DataFrame = feature_state["customer_stats"]
    if cust not in customer_stats.index:
        pytest.skip(f"Customer {cust} not in feature_state — cannot verify")

    fitted_count = int(customer_stats.loc[cust, "customer_tx_count"])

    assert fitted_count == train_count, (
        f"Leakage detected! Customer '{cust}' has {full_count} total transactions "
        f"but feature_state has {fitted_count} (expected train-only count: {train_count}). "
        "Customer aggregates must only reflect the training set."
    )


def test_no_missing_values_after_full_pipeline(
    transformed_dfs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """After the full pipeline, critical columns must have zero NaN values."""
    critical = [
        "amount_log",
        "transaction_hour",
        "is_night_transaction",
        "segment_encoded",
        "customer_tx_count",
        "customer_avg_amount",
        "is_fraud",
    ]
    for split_name, df in zip(("train", "test"), transformed_dfs):
        for col in critical:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                assert null_count == 0, (
                    f"{split_name}.{col} has {null_count} NaN values after pipeline"
                )


# ===========================================================================
# Tests — save_processed_data + save_feature_metadata
# ===========================================================================

def test_save_feature_state_creates_artifacts(
    feature_state: dict,
    tmp_path: Path,
) -> None:
    """save_feature_state should create customer_stats.parquet, segment_label_map.json,
    and amount_median_train.json in the output directory."""
    save_feature_state(feature_state, str(tmp_path))

    assert (tmp_path / "customer_stats.parquet").exists(), "customer_stats.parquet missing"
    assert (tmp_path / "segment_label_map.json").exists(), "segment_label_map.json missing"
    assert (tmp_path / "amount_median_train.json").exists(), "amount_median_train.json missing"

    # Verify parquet integrity
    stats = pd.read_parquet(tmp_path / "customer_stats.parquet")
    assert "customer_tx_count" in stats.columns
    assert "customer_avg_amount" in stats.columns

    # Verify segment map JSON
    import json
    with open(tmp_path / "segment_label_map.json") as fh:
        seg_map = json.load(fh)
    assert isinstance(seg_map, dict)
    assert len(seg_map) > 0

    # Verify median JSON
    with open(tmp_path / "amount_median_train.json") as fh:
        median_data = json.load(fh)
    assert "amount_median_train" in median_data
    assert isinstance(median_data["amount_median_train"], float)


def test_save_processed_data_creates_parquet(
    transformed_dfs: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """save_processed_data should create train.parquet and test.parquet."""
    train, test = transformed_dfs
    save_processed_data(train, test, str(tmp_path))

    assert (tmp_path / "train.parquet").exists(), "train.parquet not created"
    assert (tmp_path / "test.parquet").exists(), "test.parquet not created"

    # Verify parquet files are readable and have matching row counts
    train_loaded = pd.read_parquet(tmp_path / "train.parquet")
    test_loaded = pd.read_parquet(tmp_path / "test.parquet")

    assert len(train_loaded) == len(train), "train.parquet row count mismatch"
    assert len(test_loaded) == len(test), "test.parquet row count mismatch"


def test_feature_metadata_saved(
    transformed_dfs: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """save_feature_metadata should create a valid features.json file."""
    train, _ = transformed_dfs
    non_feature = {"transaction_id", "customer_id", "timestamp", "is_fraud"}
    feature_cols = [c for c in train.columns if c not in non_feature]

    save_feature_metadata(feature_cols, str(tmp_path))

    metadata_path = tmp_path / "features.json"
    assert metadata_path.exists(), "features.json was not created"

    with open(metadata_path, encoding="utf-8") as fh:
        metadata = json.load(fh)

    assert "target" in metadata, "'target' key missing from features.json"
    assert metadata["target"] == "is_fraud", "Target should be 'is_fraud'"
    assert "features" in metadata, "'features' key missing from features.json"
    assert isinstance(metadata["features"], list), "'features' should be a list"
    assert len(metadata["features"]) == len(feature_cols), "Feature count mismatch"
    assert "created_at" in metadata, "'created_at' key missing from features.json"


# ===========================================================================
# Tests — analyze_dataset integration
# ===========================================================================

def test_analyze_dataset_runs_without_error(raw_csv_path: str) -> None:
    """analyze_dataset should complete without raising any exception."""
    analyze_dataset(raw_csv_path)  # should not raise


def test_analyze_dataset_file_not_found() -> None:
    """analyze_dataset should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        analyze_dataset("/nonexistent/missing.csv")


# ===========================================================================
# Tests — run_preprocessing (end-to-end)
# ===========================================================================

def test_run_preprocessing_end_to_end(raw_csv_path: str, tmp_path: Path) -> None:
    """run_preprocessing should produce valid train/test DataFrames and output files."""
    train, test = run_preprocessing(raw_csv_path, str(tmp_path))

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) > 0 and len(test) > 0

    # All expected output files must exist
    for fname in (
        "train.parquet",
        "test.parquet",
        "features.json",
        "customer_stats.parquet",
        "segment_label_map.json",
        "amount_median_train.json",
    ):
        assert (tmp_path / fname).exists(), f"Missing output file: {fname}"

    # Temporal integrity
    assert train["timestamp"].max() < test["timestamp"].min(), (
        "End-to-end split violated temporal ordering"
    )

    # Derived features present
    for col in ("transaction_hour", "amount_log", "segment_encoded", "is_night_transaction"):
        assert col in train.columns, f"Missing derived feature: {col}"
