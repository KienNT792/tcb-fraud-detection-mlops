from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from infrastructure.pipeline import PIPELINE_CONFIG

from .mlflow_utils import write_json


FILES_TO_COPY = [
    "features.json",
    "customer_stats.parquet",
    "segment_label_map.json",
    "amount_median_train.json",
    "categorical_maps.json",
]


def materialize_feature_ready(processed_dir: str, output_dir: str) -> dict[str, list[str]]:
    src_dir = Path(processed_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for filename in FILES_TO_COPY:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, dst_dir / filename)
            copied.append(filename)

    payload = {
        "processed_dir": str(src_dir),
        "feature_ready_dir": str(dst_dir),
        "copied_files": copied,
    }
    write_json(dst_dir / "manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy processed artifacts into the feature-ready zone.")
    parser.add_argument("--processed-dir", default=str(PIPELINE_CONFIG.paths.processed_dir))
    parser.add_argument("--output-dir", default=str(PIPELINE_CONFIG.paths.feature_ready_dir))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_feature_ready(args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
