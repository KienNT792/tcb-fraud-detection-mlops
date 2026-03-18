from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from infrastructure.pipeline import PIPELINE_CONFIG

from .inference import FraudDetector


def run_batch_scoring(
    input_path: str,
    output_dir: str,
    models_dir: str = str(PIPELINE_CONFIG.paths.models_dir),
    processed_dir: str = str(PIPELINE_CONFIG.paths.processed_dir),
) -> Path | None:
    src = Path(input_path)
    if not src.exists():
        return None

    if src.suffix.lower() == ".parquet":
        raw_df = pd.read_parquet(src)
    else:
        raw_df = pd.read_csv(src)

    detector = FraudDetector(models_dir, processed_dir)
    scored = detector.predict_batch(raw_df)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"batch_scored_{timestamp}.parquet"
    scored.to_parquet(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch scoring on a CSV or Parquet file.")
    parser.add_argument("--input-path", default=str(PIPELINE_CONFIG.paths.batch_input_path))
    parser.add_argument("--output-dir", default=str(PIPELINE_CONFIG.paths.scored_dir))
    parser.add_argument("--models-dir", default=str(PIPELINE_CONFIG.paths.models_dir))
    parser.add_argument("--processed-dir", default=str(PIPELINE_CONFIG.paths.processed_dir))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch_scoring(
        input_path=args.input_path,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        processed_dir=args.processed_dir,
    )


if __name__ == "__main__":
    main()
