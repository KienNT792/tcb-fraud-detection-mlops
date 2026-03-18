from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_list_env(name: str, default: list[int]) -> list[int]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return values or default


@dataclass(frozen=True)
class PathConfig:
    raw_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("RAW_DATA_DIR", "data/raw")))
    validated_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("VALIDATED_DATA_DIR", "data/validated"))
    )
    processed_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("PROCESSED_DIR", "data/processed"))
    )
    feature_ready_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("FEATURE_READY_DIR", "data/feature-ready"))
    )
    models_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("MODELS_DIR", "models")))
    artifacts_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    )
    incoming_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("INCOMING_DATA_DIR", "data/incoming"))
    )
    scored_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("SCORED_DATA_DIR", "data/scored"))
    )

    @property
    def demo_raw_path(self) -> Path:
        return self.raw_dir / "demo_transactions.csv"

    @property
    def validated_dataset_path(self) -> Path:
        return self.validated_dir / "validated_transactions.parquet"

    @property
    def evaluation_dir(self) -> Path:
        return self.models_dir / "evaluation"

    @property
    def deployment_manifest_path(self) -> Path:
        return self.artifacts_dir / "deployment" / "release_manifest.json"

    @property
    def drift_output_dir(self) -> Path:
        return self.artifacts_dir / "monitoring"

    @property
    def batch_input_path(self) -> Path:
        return self.incoming_dir / "batch_scoring_input.csv"


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI",
            f"file://{(PROJECT_ROOT / 'ml_pipeline' / 'mlruns').resolve()}",
        )
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "tcb-fraud-detection")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_MODEL_NAME", "tcb-fraud-detector")
    )
    dataset_version: str = field(
        default_factory=lambda: os.getenv("DATASET_VERSION", "unversioned")
    )
    config_version: str = field(
        default_factory=lambda: os.getenv("CONFIG_VERSION", "v1")
    )
    pipeline_trigger: str = field(
        default_factory=lambda: os.getenv("PIPELINE_TRIGGER", "manual")
    )
    pipeline_actor: str = field(
        default_factory=lambda: os.getenv("PIPELINE_ACTOR", "unknown")
    )
    min_recall: float = field(
        default_factory=lambda: float(os.getenv("MIN_RECALL_THRESHOLD", "0.95"))
    )


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    )
    input_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_INPUT_TOPIC", "transactions.raw")
    )
    output_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_OUTPUT_TOPIC", "transactions.scored")
    )


@dataclass(frozen=True)
class AirflowConfig:
    project_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AIRFLOW_PROJECT_DIR", "/opt/airflow/project"))
    )
    training_schedule: str = field(
        default_factory=lambda: os.getenv("AIRFLOW_TRAINING_SCHEDULE", "@daily")
    )
    monitoring_schedule: str = field(
        default_factory=lambda: os.getenv("AIRFLOW_MONITORING_SCHEDULE", "0 * * * *")
    )
    batch_schedule: str = field(
        default_factory=lambda: os.getenv("AIRFLOW_BATCH_SCHEDULE", "*/15 * * * *")
    )
    owner: str = field(default_factory=lambda: os.getenv("AIRFLOW_OWNER", "mlops"))
    retries: int = field(default_factory=lambda: int(os.getenv("AIRFLOW_RETRIES", "2")))
    retry_delay_minutes: int = field(
        default_factory=lambda: int(os.getenv("AIRFLOW_RETRY_DELAY_MINUTES", "5"))
    )
    start_date: datetime = field(default_factory=lambda: datetime(2026, 1, 1))

    @property
    def retry_delay(self) -> timedelta:
        return timedelta(minutes=self.retry_delay_minutes)


@dataclass(frozen=True)
class ServingConfig:
    rollout_config_path: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("ROLLOUT_CONFIG_PATH", "artifacts/rollout/rollout_config.json")
        )
    )
    candidate_models_dir: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("CANDIDATE_MODELS_DIR", "models/candidate")
        )
    )
    candidate_processed_dir: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("CANDIDATE_PROCESSED_DIR", "data/processed")
        )
    )
    rollout_steps: list[int] = field(
        default_factory=lambda: _parse_int_list_env("ROLLOUT_STEPS", [10, 25, 50, 100])
    )
    rollout_step_interval_minutes: int = field(
        default_factory=lambda: int(os.getenv("ROLLOUT_STEP_INTERVAL_MINUTES", "30"))
    )
    rollout_auto_advance: bool = field(
        default_factory=lambda: _parse_bool_env("ROLLOUT_AUTO_ADVANCE", False)
    )
    rollout_auto_promote: bool = field(
        default_factory=lambda: _parse_bool_env("ROLLOUT_AUTO_PROMOTE", False)
    )


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path = PROJECT_ROOT
    paths: PathConfig = field(default_factory=PathConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    airflow: AirflowConfig = field(default_factory=AirflowConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


PIPELINE_CONFIG = PipelineConfig()
