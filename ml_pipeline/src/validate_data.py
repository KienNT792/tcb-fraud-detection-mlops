from __future__ import annotations

import argparse
from pathlib import Path

from infrastructure.pipeline import PIPELINE_CONFIG

from .mlflow_utils import write_json
from .preprocess import load_dataset, validate_schema


def run_validation(data_path: str, validated_dir: str) -> dict[str, int]:
    df = load_dataset(data_path)
    validate_schema(df)

    out = Path(validated_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / "validated_transactions.parquet"
    df.to_parquet(parquet_path, index=False)

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_transaction_ids": int(df["transaction_id"].duplicated().sum()),
        "null_timestamps": int(df["timestamp"].isna().sum()),
    }
    write_json(out / "validation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw fraud dataset and persist a validated zone.")
    parser.add_argument("--data-path", default=str(PIPELINE_CONFIG.paths.demo_raw_path))
    parser.add_argument("--validated-dir", default=str(PIPELINE_CONFIG.paths.validated_dir))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(args.data_path, args.validated_dir)


if __name__ == "__main__":
    main()
