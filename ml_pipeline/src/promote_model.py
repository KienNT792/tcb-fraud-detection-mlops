from __future__ import annotations

import argparse
import logging

from ml_pipeline.src.model_registry import (
    find_latest_version_by_run,
    transition_model_version_stage,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a registered MLflow model"
            " version to a target stage."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="tcb-fraud-xgboost",
        help="MLflow registered model name.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["Staging", "Production", "Archived", "None"],
        help="Target stage for promotion.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Explicit model version to promote.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Training run_id used to resolve latest registered version.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    version = args.version

    if version is None:
        if not args.run_id:
            raise ValueError("Either --version or --run-id must be provided.")
        resolved = find_latest_version_by_run(
            model_name=args.model_name,
            run_id=args.run_id,
        )
        if resolved is None:
            raise RuntimeError(
                f"No model version found for"
                f" model={args.model_name}"
                f" run_id={args.run_id}"
            )
        version = resolved

    transition_model_version_stage(
        model_name=args.model_name,
        version=version,
        stage=args.stage,
        archive_existing_versions=True,
    )
    logger.info(
        "Promotion complete | model=%s | version=%s | stage=%s",
        args.model_name,
        version,
        args.stage,
    )


if __name__ == "__main__":
    from ml_pipeline.src.logging_config import setup_logging
    setup_logging()

    main()

