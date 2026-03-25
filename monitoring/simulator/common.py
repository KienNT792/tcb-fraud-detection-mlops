from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import load_dotenv

from ml_pipeline.src.evaluate import run_evaluation
from ml_pipeline.src.model_registry import (
    find_latest_version_by_stage,
    transition_model_version_stage,
)
from ml_pipeline.src.preprocess import run_preprocessing
from ml_pipeline.src.registry_metadata import (
    REGISTRY_METADATA_FILENAME,
    read_registry_metadata,
)
from ml_pipeline.src.runtime_bundle import PROCESSED_RUNTIME_FILES
from ml_pipeline.src.train import run_training

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ROOT_USER", ""))
os.environ.setdefault(
    "AWS_SECRET_ACCESS_KEY",
    os.getenv("MINIO_ROOT_PASSWORD", ""),
)
if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
    raise EnvironmentError(
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (or MINIO_ROOT_USER / "
        "MINIO_ROOT_PASSWORD) must be set. Configure them in .env file."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOADBALANCER_URL = os.getenv("SIMULATOR_TARGET_URL", "http://127.0.0.1:8000").rstrip("/")
STABLE_URL = os.getenv("SIMULATOR_STABLE_URL", "http://127.0.0.1:8002").rstrip("/")
CANDIDATE_URL = os.getenv("SIMULATOR_CANDIDATE_URL", "http://127.0.0.1:8003").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("SIMULATOR_REQUEST_TIMEOUT_SECONDS", "15"))
DEFAULT_THRESHOLD = float(
    os.getenv("SIMULATOR_DRIFT_THRESHOLD", os.getenv("DRIFT_ALERT_THRESHOLD", "0.2"))
)
DOCKER_COMPOSE_CANDIDATE_PROFILE = os.getenv(
    "DOCKER_COMPOSE_CANDIDATE_PROFILE",
    "candidate",
)
CANDIDATE_SERVICE_NAME = os.getenv("CANDIDATE_SERVICE_NAME", "fastapi-candidate")
CANDIDATE_CONTAINER_NAME = os.getenv(
    "CANDIDATE_CONTAINER_NAME",
    "tcb-fastapi-candidate",
)
PROMETHEUS_FILE_SD_DIR = REPO_ROOT / "monitoring" / "prometheus" / "file_sd"
PROMETHEUS_CANDIDATE_TARGET_FILE = PROMETHEUS_FILE_SD_DIR / "candidate.json"
PROMETHEUS_CANDIDATE_SCRAPE_TARGET = os.getenv(
    "PROMETHEUS_CANDIDATE_SCRAPE_TARGET",
    "fastapi-candidate:8000",
)

MODELS_ROOT = REPO_ROOT / "models"
VERSIONS_DIR = MODELS_ROOT / "versions"
CANDIDATE_DIR = MODELS_ROOT / "deployments" / "candidate"
CANDIDATE_PROCESSED_DIR = CANDIDATE_DIR / "processed"
CANARY_SPLIT_CONFIG = REPO_ROOT / "monitoring" / "loadbalancer" / "canary_split.conf"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DATASET_PATH = REPO_ROOT / "data" / "raw" / "tcb_credit_fraud_dataset.csv"
TRAINING_ARTIFACTS = ("train.parquet", "test.parquet", "features.json")

MODEL_FILENAME = "xgb_fraud_model.joblib"
METRICS_FILENAME = "metrics.json"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.csv"
EVALUATION_DIRNAME = "evaluation"
EVALUATION_FILENAME = "evaluation.json"
MANIFEST_FILENAME = "model_manifest.json"
ROOT_ARTIFACTS = (
    MODEL_FILENAME,
    METRICS_FILENAME,
    FEATURE_IMPORTANCE_FILENAME,
)
PROMOTE_TARGET_STAGE = os.getenv("MLFLOW_PROMOTE_STAGE", "Production")
RUNTIME_BUNDLE_ARTIFACT_PATH = os.getenv(
    "MLFLOW_RUNTIME_BUNDLE_PATH",
    "runtime_bundle",
)
ROLLBACK_REGISTRY_NONE = "none"
ROLLBACK_REGISTRY_ARCHIVE_CURRENT = "archive-current-production"
ROLLBACK_REGISTRY_RESTORE_PREVIOUS = "restore-previous-production"
ROLLBACK_REGISTRY_ACTIONS = (
    ROLLBACK_REGISTRY_NONE,
    ROLLBACK_REGISTRY_ARCHIVE_CURRENT,
    ROLLBACK_REGISTRY_RESTORE_PREVIOUS,
)


@dataclass(slots=True)
class ScenarioResult:
    name: str
    target_url: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    elapsed_seconds: float

    @property
    def achieved_rps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_requests / self.elapsed_seconds


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _random_id(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"{prefix}_{suffix}"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    tmp_path.replace(path)


def _manifest_path(model_dir: Path) -> Path:
    return model_dir / MANIFEST_FILENAME


def ensure_prometheus_scrape_layout() -> None:
    PROMETHEUS_FILE_SD_DIR.mkdir(parents=True, exist_ok=True)
    if not PROMETHEUS_CANDIDATE_TARGET_FILE.exists():
        _write_json(PROMETHEUS_CANDIDATE_TARGET_FILE, [])


def set_candidate_scrape_target(enabled: bool) -> None:
    ensure_prometheus_scrape_layout()
    payload: list[dict[str, Any]]
    if enabled:
        payload = [{"targets": [PROMETHEUS_CANDIDATE_SCRAPE_TARGET]}]
    else:
        payload = []
    _write_json(PROMETHEUS_CANDIDATE_TARGET_FILE, payload)


def read_manifest(model_dir: Path) -> dict[str, Any] | None:
    manifest_path = _manifest_path(model_dir)
    if not manifest_path.exists():
        return None
    return _read_json(manifest_path)


def _reset_runtime_bundle(
    target_dir: Path,
    *,
    processed_dir: Path | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in (*ROOT_ARTIFACTS, MANIFEST_FILENAME, REGISTRY_METADATA_FILENAME):
        artifact_path = target_dir / filename
        if artifact_path.exists():
            artifact_path.unlink()

    evaluation_dir = target_dir / EVALUATION_DIRNAME
    if evaluation_dir.exists():
        shutil.rmtree(evaluation_dir)

    resolved_processed_dir = processed_dir or (target_dir / "processed")
    if resolved_processed_dir.exists():
        for filename in PROCESSED_RUNTIME_FILES:
            artifact_path = resolved_processed_dir / filename
            if artifact_path.exists():
                artifact_path.unlink()
        if (
            resolved_processed_dir != PROCESSED_DIR
            and resolved_processed_dir.exists()
            and not any(resolved_processed_dir.iterdir())
        ):
            resolved_processed_dir.rmdir()


def copy_model_artifacts(
    source_dir: Path,
    target_dir: Path,
    *,
    reset_target: bool = True,
) -> None:
    source_path = source_dir.resolve()
    target_path = target_dir.resolve()
    if not (source_path / MODEL_FILENAME).exists():
        raise FileNotFoundError(
            f"Model artifact not found at {source_path / MODEL_FILENAME}"
        )

    if reset_target:
        _reset_runtime_bundle(target_path)

    for filename in ROOT_ARTIFACTS:
        src = source_path / filename
        if src.exists():
            shutil.copy2(src, target_path / filename)

    evaluation_src = source_path / EVALUATION_DIRNAME
    if evaluation_src.exists():
        shutil.copytree(
            evaluation_src,
            target_path / EVALUATION_DIRNAME,
            dirs_exist_ok=True,
        )

    registry_metadata = read_registry_metadata(source_path)
    if registry_metadata:
        (target_path / REGISTRY_METADATA_FILENAME).write_text(
            json.dumps(registry_metadata, indent=2),
            encoding="utf-8",
        )


def copy_processed_artifacts(source_dir: Path, target_dir: Path) -> None:
    source_path = source_dir.resolve()
    target_path = target_dir.resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    for filename in PROCESSED_RUNTIME_FILES:
        src = source_path / filename
        if not src.exists():
            raise FileNotFoundError(
                f"Processed artifact not found at {src}"
            )
        shutil.copy2(src, target_path / filename)


def copy_runtime_bundle(
    *,
    source_model_dir: Path,
    source_processed_dir: Path,
    target_model_dir: Path,
    target_processed_dir: Path,
) -> None:
    _reset_runtime_bundle(target_model_dir, processed_dir=target_processed_dir)
    copy_model_artifacts(
        source_model_dir,
        target_model_dir,
        reset_target=False,
    )
    copy_processed_artifacts(source_processed_dir, target_processed_dir)


def collect_model_summary(model_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "models_dir": str(model_dir),
        "metrics": {},
        "evaluation": {},
    }

    metrics_path = model_dir / METRICS_FILENAME
    if metrics_path.exists():
        summary["metrics"] = _read_json(metrics_path)

    evaluation_path = model_dir / EVALUATION_DIRNAME / EVALUATION_FILENAME
    if evaluation_path.exists():
        evaluation = _read_json(evaluation_path)
        summary["evaluation"] = {
            "overall_status": evaluation.get("overall_status"),
            "threshold_metrics": evaluation.get("threshold_metrics", {}),
            "baseline_comparison": evaluation.get("baseline_comparison", {}),
            "evaluated_at": evaluation.get("evaluated_at"),
        }

    return summary


def registry_manifest_fields(model_dir: Path) -> dict[str, Any]:
    registry_metadata = read_registry_metadata(model_dir)
    if not registry_metadata:
        return {}
    return {
        "registry_model_name": registry_metadata.get("model_name"),
        "registry_version": str(registry_metadata.get("version")),
        "registry_stage": registry_metadata.get("stage"),
        "run_id": registry_metadata.get("run_id"),
        "runtime_bundle_path": registry_metadata.get("runtime_bundle_artifact_path"),
    }


def current_stable_registry_target() -> tuple[str, int] | None:
    stable_manifest = read_manifest(MODELS_ROOT) or {}
    model_name = stable_manifest.get("registry_model_name")
    version = stable_manifest.get("registry_version")
    if not model_name or version in (None, ""):
        return None
    return str(model_name), int(str(version))


def previous_stable_registry_target() -> tuple[str, int] | None:
    stable_manifest = read_manifest(MODELS_ROOT) or {}
    model_name = stable_manifest.get("previous_registry_model_name") or stable_manifest.get(
        "registry_model_name"
    )
    version = stable_manifest.get("previous_registry_version")
    if model_name and version not in (None, ""):
        return str(model_name), int(str(version))

    current_target = current_stable_registry_target()
    if current_target is None:
        return None

    current_model_name, current_version = current_target
    fallback_version = find_latest_version_by_stage(
        model_name=current_model_name,
        stage="Archived",
        exclude_versions={current_version},
    )
    if fallback_version is None:
        return None
    return current_model_name, fallback_version


def sync_registry_version_to_stable(
    *,
    model_name: str,
    version: int,
) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    command = [sys.executable, "scripts/runtime_bundle_registry.py"]
    if tracking_uri:
        command.extend(["--tracking-uri", tracking_uri])
    command.extend(
        [
            "download-version",
            "--model-name",
            model_name,
            "--version",
            str(version),
            "--artifact-path",
            RUNTIME_BUNDLE_ARTIFACT_PATH,
            "--output-root",
            str(REPO_ROOT),
            "--models-output-dir",
            str(MODELS_ROOT),
            "--processed-output-dir",
            str(PROCESSED_DIR),
            "--manifest-slot",
            "stable",
        ]
    )
    _run_command(command)


def write_manifest(
    target_dir: Path,
    *,
    slot: str,
    model_id: str,
    source_model_dir: Path,
    metrics_summary: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "slot": slot,
        "model_id": model_id,
        "source_model_dir": str(source_model_dir.resolve()),
        "updated_at": _utc_now(),
        "metrics_summary": metrics_summary or collect_model_summary(target_dir),
    }
    if extra:
        manifest.update(extra)
    _write_json(_manifest_path(target_dir), manifest)
    return manifest


def _slot_payload(slot: str, model_dir: Path) -> dict[str, Any]:
    manifest = read_manifest(model_dir)
    if manifest is None:
        return {
            "slot": slot,
            "status": "empty",
            "model_id": None,
            "models_dir": str(model_dir),
            "manifest_path": str(_manifest_path(model_dir)),
        }

    return {
        "slot": slot,
        "status": "ready",
        "model_id": manifest.get("model_id"),
        "models_dir": str(model_dir),
        "manifest_path": str(_manifest_path(model_dir)),
        "source_model_dir": manifest.get("source_model_dir"),
        "updated_at": manifest.get("updated_at"),
        "metrics_summary": manifest.get("metrics_summary", {}),
    }


def _write_canary_split_config(candidate_percentage: int) -> None:
    CANARY_SPLIT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if candidate_percentage <= 0:
        rules = "  * stable;\n"
    elif candidate_percentage >= 100:
        rules = "  * candidate;\n"
    else:
        rules = f"  {candidate_percentage}% candidate;\n  * stable;\n"
    CANARY_SPLIT_CONFIG.write_text(
        (
            'split_clients "${remote_addr}${msec}${request_length}" $rollout_slot {\n'
            f"{rules}"
            "}\n"
        ),
        encoding="utf-8",
    )


def _run_command(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(command)
            + " | "
            + (result.stderr.strip() or result.stdout.strip() or "unknown error")
        )
    return result


def _run_compose(
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "compose",
        "--profile",
        DOCKER_COMPOSE_CANDIDATE_PROFILE,
        *args,
    ]
    return _run_command(command, check=check)


def candidate_manifest_exists() -> bool:
    return _manifest_path(CANDIDATE_DIR).exists()


def is_candidate_service_running() -> bool:
    result = _run_command(
        ["docker", "inspect", "-f", "{{.State.Running}}", CANDIDATE_CONTAINER_NAME],
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def ensure_candidate_service_running() -> None:
    if not candidate_manifest_exists():
        raise FileNotFoundError(
            f"Candidate manifest not found at {_manifest_path(CANDIDATE_DIR)}"
        )

    if is_candidate_service_running():
        set_candidate_scrape_target(True)
        return

    logger.info("Starting candidate service | service=%s", CANDIDATE_SERVICE_NAME)
    _run_compose("up", "-d", CANDIDATE_SERVICE_NAME)
    wait_for_http_ready(f"{CANDIDATE_URL}/health")
    set_candidate_scrape_target(True)


def stop_candidate_service() -> None:
    set_candidate_scrape_target(False)
    _run_compose("stop", CANDIDATE_SERVICE_NAME, check=False)
    _run_compose("rm", "-f", CANDIDATE_SERVICE_NAME, check=False)


def clear_candidate_slot(*, stop_service: bool = True) -> None:
    _reset_runtime_bundle(CANDIDATE_DIR, processed_dir=CANDIDATE_PROCESSED_DIR)
    set_candidate_scrape_target(False)
    if stop_service:
        stop_candidate_service()


def bootstrap_runtime_layout() -> None:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_prometheus_scrape_layout()

    if (MODELS_ROOT / MODEL_FILENAME).exists() and read_manifest(MODELS_ROOT) is None:
        write_manifest(
            MODELS_ROOT,
            slot="stable",
            model_id=os.getenv("DEFAULT_STABLE_MODEL_ID", "stable-bootstrap"),
            source_model_dir=MODELS_ROOT,
            metrics_summary=collect_model_summary(MODELS_ROOT),
        )

    if not CANARY_SPLIT_CONFIG.exists():
        _write_canary_split_config(0)


def get_canary_percentage() -> int:
    bootstrap_runtime_layout()
    for line in CANARY_SPLIT_CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.endswith("% candidate;"):
            return int(stripped.split("%", 1)[0])
        if stripped == "* candidate;":
            return 100
        if stripped == "* stable;":
            return 0
    return 0


def reload_loadbalancer() -> None:
    command = ["docker", "compose", "exec", "-T", "loadbalancer", "nginx", "-s", "reload"]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to reload nginx loadbalancer: "
            + (result.stderr.strip() or result.stdout.strip() or "unknown error")
        )


def set_canary_percentage(candidate_percentage: int, *, reload: bool = True) -> dict[str, Any]:
    bootstrap_runtime_layout()
    resolved_percentage = max(0, min(100, int(candidate_percentage)))
    if resolved_percentage > 0:
        if read_manifest(CANDIDATE_DIR) is None:
            raise RuntimeError("No candidate model is staged for rollout.")
        ensure_candidate_service_running()
    _write_canary_split_config(resolved_percentage)
    if reload:
        reload_loadbalancer()
    return fetch_rollout_state()


def rollback_canary(
    *,
    registry_action: str = ROLLBACK_REGISTRY_NONE,
) -> dict[str, Any]:
    bootstrap_runtime_layout()
    if registry_action not in ROLLBACK_REGISTRY_ACTIONS:
        raise ValueError(
            "registry_action must be one of: "
            + ", ".join(ROLLBACK_REGISTRY_ACTIONS)
        )

    rollback_result: dict[str, Any] = {
        "traffic_reset": True,
        "candidate_stopped": True,
        "candidate_cleared": True,
        "registry_action": registry_action,
        "registry_result": "skipped",
    }

    set_canary_percentage(0)
    candidate_manifest = read_manifest(CANDIDATE_DIR) or {}
    stable_manifest = read_manifest(MODELS_ROOT) or {}

    if registry_action == ROLLBACK_REGISTRY_ARCHIVE_CURRENT:
        current_target = current_stable_registry_target()
        if current_target is not None:
            model_name, version = current_target
            transition_model_version_stage(
                model_name=model_name,
                version=version,
                stage="Archived",
                archive_existing_versions=False,
            )
            rollback_result["registry_result"] = "archived-current-production"
            rollback_result["archived_registry_model_name"] = model_name
            rollback_result["archived_registry_version"] = str(version)
        else:
            rollback_result["registry_result"] = "no-current-production-version"
    elif registry_action == ROLLBACK_REGISTRY_RESTORE_PREVIOUS:
        current_target = current_stable_registry_target()
        previous_target = previous_stable_registry_target()
        if previous_target is not None:
            model_name, version = previous_target
            transition_model_version_stage(
                model_name=model_name,
                version=version,
                stage=PROMOTE_TARGET_STAGE,
                archive_existing_versions=True,
            )
            sync_registry_version_to_stable(
                model_name=model_name,
                version=version,
            )
            restored_model_id = f"{model_name}-v{version}"
            restored_manifest = read_manifest(MODELS_ROOT) or {}
            restored_manifest.update(
                {
                    "restored_at": _utc_now(),
                    "rollback_registry_action": registry_action,
                    "rollback_previous_model_id": stable_manifest.get("model_id"),
                    "rollback_previous_registry_model_name": (
                        current_target[0] if current_target else None
                    ),
                    "rollback_previous_registry_version": (
                        str(current_target[1]) if current_target else None
                    ),
                    "previous_stable_model_id": stable_manifest.get(
                        "previous_stable_model_id"
                    ),
                    "previous_registry_model_name": stable_manifest.get(
                        "previous_registry_model_name"
                    ),
                    "previous_registry_version": stable_manifest.get(
                        "previous_registry_version"
                    ),
                    "previous_registry_stage": stable_manifest.get(
                        "previous_registry_stage"
                    ),
                    "previous_run_id": stable_manifest.get("previous_run_id"),
                    "previous_runtime_bundle_path": stable_manifest.get(
                        "previous_runtime_bundle_path"
                    ),
                }
            )
            _write_json(_manifest_path(MODELS_ROOT), restored_manifest)
            wait_for_model_version(STABLE_URL, restored_model_id)
            rollback_result["registry_result"] = "restored-previous-production"
            rollback_result["restored_registry_model_name"] = model_name
            rollback_result["restored_registry_version"] = str(version)
        else:
            rollback_result["registry_result"] = "no-previous-production-version"

    clear_candidate_slot(stop_service=True)
    state = fetch_rollout_state()
    state["rollback"] = rollback_result
    if candidate_manifest:
        state["rollback"]["candidate_model_id"] = candidate_manifest.get("model_id")
    return state


def fetch_rollout_state() -> dict[str, Any]:
    bootstrap_runtime_layout()
    candidate_percentage = get_canary_percentage()
    return {
        "rollout": {
            "mode": "canary" if candidate_percentage > 0 else "idle",
            "candidate_percentage": candidate_percentage,
        },
        "stable": _slot_payload("stable", MODELS_ROOT),
        "candidate": _slot_payload("candidate", CANDIDATE_DIR),
    }


def fetch_drift_snapshot(stable_url: str = STABLE_URL) -> dict[str, Any]:
    with httpx.Client(timeout=5.0) as client:
        response = client.get(f"{stable_url}/monitoring/drift")
        response.raise_for_status()
        return response.json()


def log_rollout_state() -> None:
    state = fetch_rollout_state()
    logger.info(
        "Rollout state | mode=%s | candidate=%s%% | stable=%s | candidate_model=%s",
        state["rollout"]["mode"],
        state["rollout"]["candidate_percentage"],
        state["stable"].get("model_id"),
        state["candidate"].get("model_id"),
    )


def baseline_payload(index: int) -> dict[str, Any]:
    timestamp = datetime.now(tz=timezone.utc) - timedelta(minutes=index % 180)
    amount = max(50_000, int(random.gauss(400_000, 180_000)))
    return {
        "transaction_id": _random_id("TX"),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": _random_id("CUST"),
        "amount": amount,
        "customer_tier": random.choice(["MASS", "PRIORITY", "PRIVATE", "INSPIRE"]),
        "card_type": random.choice(["VISA", "MASTERCARD"]),
        "card_tier": random.choice(["STANDARD", "GOLD", "PLATINUM"]),
        "currency": "VND",
        "merchant_name": random.choice(["Grab", "Shopee", "Tiki", "CircleK"]),
        "mcc_code": random.choice([4121, 5812, 5411, 5732]),
        "merchant_category": random.choice(
            ["Transport", "Food", "Retail", "Electronics"]
        ),
        "merchant_city": random.choice(["Ha Noi", "Ho Chi Minh", "Da Nang"]),
        "merchant_country": "VN",
        "device_type": random.choice(["Mobile", "Desktop"]),
        "os": random.choice(["iOS", "Android", "Windows"]),
        "ip_country": "VN",
        "distance_from_home_km": round(abs(random.gauss(3.0, 2.0)), 2),
        "cvv_match": "Y",
        "is_3d_secure": random.choice(["Y", "N"]),
        "transaction_status": "APPROVED",
        "tx_count_last_1h": max(0, int(random.gauss(2, 1))),
        "tx_count_last_24h": max(1, int(random.gauss(5, 3))),
        "time_since_last_tx_min": round(abs(random.gauss(90.0, 40.0)), 2),
        "avg_amount_last_30d": max(50_000, int(random.gauss(450_000, 120_000))),
        "amount_ratio_vs_avg": round(max(0.05, random.gauss(0.95, 0.25)), 3),
        "is_new_device": 0,
        "is_new_merchant": 0,
        "card_bin": 411111,
        "account_age_days": max(30, int(random.gauss(700, 180))),
        "is_weekend": 1 if timestamp.weekday() >= 5 else 0,
        "hour_of_day": timestamp.hour,
    }


def drift_payload(index: int) -> dict[str, Any]:
    payload = baseline_payload(index)
    payload.update(
        {
            "amount": max(500_000, int(random.gauss(5_500_000, 1_250_000))),
            "merchant_country": random.choice(["SG", "US", "JP"]),
            "ip_country": random.choice(["SG", "US", "JP"]),
            "device_type": random.choice(["Tablet", "Desktop"]),
            "os": random.choice(["Linux", "HarmonyOS", "Android"]),
            "distance_from_home_km": round(abs(random.gauss(120.0, 35.0)), 2),
            "tx_count_last_1h": random.randint(8, 24),
            "tx_count_last_24h": random.randint(20, 80),
            "time_since_last_tx_min": round(random.uniform(0.0, 8.0), 2),
            "amount_ratio_vs_avg": round(random.uniform(4.5, 12.0), 3),
            "is_new_device": 1,
            "is_new_merchant": 1,
            "hour_of_day": random.choice([0, 1, 2, 3, 4, 23]),
            "merchant_name": random.choice(
                ["UnknownMerchant", "CryptoExpress", "ForeignCasino"]
            ),
            "merchant_category": random.choice(
                ["Gaming", "Crypto", "Travel", "HighRiskRetail"]
            ),
        }
    )
    return payload


def post_retrain_payload(index: int) -> dict[str, Any]:
    payload = baseline_payload(index)
    payload["amount"] = max(100_000, int(random.gauss(750_000, 250_000)))
    payload["merchant_country"] = random.choice(["VN", "SG"])
    payload["ip_country"] = random.choice(["VN", "SG"])
    return payload


def wait_for_http_ready(url: str, *, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = "unknown"
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def wait_for_model_version(
    service_url: str,
    expected_model_id: str,
    *,
    timeout_seconds: float = 60.0,
) -> None:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{service_url}/health")
                response.raise_for_status()
                payload = response.json()
                last_payload = payload
                if payload.get("model_version") == expected_model_id and payload.get("model_loaded"):
                    return
        except Exception as exc:
            last_payload = {"error": str(exc)}
        time.sleep(1)

    raise RuntimeError(
        f"Timed out waiting for {service_url} to load model {expected_model_id}: {last_payload}"
    )


def _send_single_request(
    target_url: str,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None,
) -> bool:
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{target_url}{endpoint}",
                json=payload,
                headers=headers or {},
            )
            response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Request failed for %s%s: %s", target_url, endpoint, exc)
        return False


def run_fixed_rate(
    *,
    name: str,
    payload_builder: Callable[[int], dict[str, Any]],
    duration_seconds: int,
    rps: int,
    target_url: str = LOADBALANCER_URL,
    endpoint: str = "/predict",
    headers: dict[str, str] | None = None,
    log_rollout_every: int = 0,
) -> ScenarioResult:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    if rps <= 0:
        raise ValueError("rps must be > 0")

    total_planned = duration_seconds * rps
    logger.info(
        "Scenario %s starting | target=%s | endpoint=%s | duration=%ss | rps=%s | total_planned=%s",
        name,
        target_url,
        endpoint,
        duration_seconds,
        rps,
        total_planned,
    )
    started_at = time.perf_counter()
    sent = 0
    success = 0
    failed = 0
    max_workers = max(4, min(rps, 64))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for second_index in range(duration_seconds):
            second_started = time.perf_counter()
            futures = []
            for request_index in range(rps):
                current_index = sent + request_index
                futures.append(
                    executor.submit(
                        _send_single_request,
                        target_url,
                        endpoint,
                        payload_builder(current_index),
                        headers,
                    )
                )
            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    failed += 1
            sent += len(futures)
            if log_rollout_every > 0 and sent % log_rollout_every == 0:
                log_rollout_state()
            elapsed = time.perf_counter() - second_started
            if elapsed < 1:
                time.sleep(1 - elapsed)
            logger.info(
                "Scenario %s second=%s/%s | sent=%s | success=%s | failed=%s",
                name,
                second_index + 1,
                duration_seconds,
                sent,
                success,
                failed,
            )

    elapsed_total = time.perf_counter() - started_at
    result = ScenarioResult(
        name=name,
        target_url=target_url,
        total_requests=sent,
        successful_requests=success,
        failed_requests=failed,
        elapsed_seconds=elapsed_total,
    )
    logger.info(
        "Scenario %s completed | total=%s | success=%s | failed=%s | elapsed=%.1fs | achieved_rps=%.2f",
        result.name,
        result.total_requests,
        result.successful_requests,
        result.failed_requests,
        result.elapsed_seconds,
        result.achieved_rps,
    )
    return result


def run_phase_plan(
    *,
    name: str,
    phases: list[tuple[str, int, int, Callable[[int], dict[str, Any]]]],
    target_url: str = LOADBALANCER_URL,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    global_index = 0

    for phase_name, duration_seconds, rps, payload_builder in phases:
        def indexed_builder(local_index: int, *, base_index: int = global_index):
            return payload_builder(base_index + local_index)

        result = run_fixed_rate(
            name=f"{name}:{phase_name}",
            payload_builder=indexed_builder,
            duration_seconds=duration_seconds,
            rps=rps,
            target_url=target_url,
        )
        results.append(result)
        global_index += result.total_requests
    return results


def wait_for_drift_threshold(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_wait_seconds: int = 300,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    deadline = time.time() + max_wait_seconds
    latest_snapshot: dict[str, Any] = {}
    while time.time() < deadline:
        latest_snapshot = fetch_drift_snapshot()
        logger.info(
            "Drift snapshot | ratio=%.6f | overall=%.6f | threshold=%.3f | current_samples=%s",
            latest_snapshot.get("drift_ratio", 0.0),
            latest_snapshot.get("overall_score", 0.0),
            threshold,
            latest_snapshot.get("current_samples"),
        )
        if latest_snapshot.get("ready") and float(
            latest_snapshot.get("drift_ratio", 0.0)
        ) >= threshold:
            return latest_snapshot
        time.sleep(poll_seconds)
    raise RuntimeError(
        f"Drift threshold {threshold} was not reached within {max_wait_seconds}s. "
        f"Latest snapshot: {latest_snapshot}"
    )


def _missing_training_artifacts(processed_dir: Path = PROCESSED_DIR) -> list[Path]:
    return [
        processed_dir / artifact_name
        for artifact_name in TRAINING_ARTIFACTS
        if not (processed_dir / artifact_name).exists()
    ]


def ensure_processed_training_artifacts(
    *,
    processed_dir: Path = PROCESSED_DIR,
    raw_dataset_path: Path = RAW_DATASET_PATH,
    allow_preprocess: bool = True,
) -> None:
    missing_before = _missing_training_artifacts(processed_dir)
    if not missing_before:
        return

    if allow_preprocess and raw_dataset_path.exists():
        logger.info(
            "Processed training artifacts missing. Running preprocessing from %s",
            raw_dataset_path,
        )
        run_preprocessing(
            data_path=str(raw_dataset_path),
            output_dir=str(processed_dir),
        )
        missing_after = _missing_training_artifacts(processed_dir)
        if not missing_after:
            return
        raise FileNotFoundError(
            "Preprocessing completed but required training artifacts are still missing: "
            + ", ".join(str(path) for path in missing_after)
        )

    raise FileNotFoundError(
        "Missing training artifacts: "
        + ", ".join(str(path) for path in missing_before)
        + f". Raw dataset not found at {raw_dataset_path}."
    )


def train_candidate_model(
    *,
    processed_dir: Path = PROCESSED_DIR,
    model_id: str | None = None,
    drift_context: dict[str, Any] | None = None,
) -> tuple[str, Path, dict[str, Any], dict[str, Any]]:
    bootstrap_runtime_layout()
    resolved_model_id = model_id or f"fraud-model-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}"
    version_dir = VERSIONS_DIR / resolved_model_id
    version_dir.mkdir(parents=True, exist_ok=True)

    training_metrics = run_training(
        processed_dir=str(processed_dir),
        models_dir=str(version_dir),
    )
    evaluation = run_evaluation(
        models_dir=str(version_dir),
        processed_dir=str(processed_dir),
        evaluation_dir=str(version_dir / EVALUATION_DIRNAME),
    )
    write_manifest(
        version_dir,
        slot="version",
        model_id=resolved_model_id,
        source_model_dir=version_dir,
        metrics_summary=collect_model_summary(version_dir),
        extra={"trained_for_drift_context": drift_context or {}},
    )
    return resolved_model_id, version_dir, training_metrics, evaluation


def stage_candidate_model(
    *,
    model_id: str,
    source_model_dir: Path,
    source_processed_dir: Path = PROCESSED_DIR,
    initial_percentage: int = 10,
) -> dict[str, Any]:
    bootstrap_runtime_layout()
    copy_runtime_bundle(
        source_model_dir=source_model_dir,
        source_processed_dir=source_processed_dir,
        target_model_dir=CANDIDATE_DIR,
        target_processed_dir=CANDIDATE_PROCESSED_DIR,
    )
    write_manifest(
        CANDIDATE_DIR,
        slot="candidate",
        model_id=model_id,
        source_model_dir=source_model_dir,
        metrics_summary=collect_model_summary(CANDIDATE_DIR),
        extra={
            "processed_dir": str(source_processed_dir.resolve()),
            **registry_manifest_fields(source_model_dir),
        },
    )
    ensure_candidate_service_running()
    wait_for_model_version(CANDIDATE_URL, model_id)
    state = set_canary_percentage(initial_percentage)
    logger.info(
        "Candidate staged | model_id=%s | source=%s | initial_percentage=%s",
        model_id,
        source_model_dir,
        initial_percentage,
    )
    return state


def promote_candidate_to_stable() -> dict[str, Any]:
    bootstrap_runtime_layout()
    candidate_manifest = read_manifest(CANDIDATE_DIR)
    if candidate_manifest is None:
        raise FileNotFoundError(
            f"Candidate manifest not found at {_manifest_path(CANDIDATE_DIR)}"
        )
    previous_stable_manifest = read_manifest(MODELS_ROOT) or {}

    candidate_model_id = str(candidate_manifest.get("model_id"))
    candidate_source_dir = Path(
        candidate_manifest.get("source_model_dir") or CANDIDATE_DIR
    )
    registry_model_name = candidate_manifest.get("registry_model_name")
    registry_version = candidate_manifest.get("registry_version")
    if registry_model_name and registry_version:
        transition_model_version_stage(
            model_name=str(registry_model_name),
            version=int(str(registry_version)),
            stage=PROMOTE_TARGET_STAGE,
            archive_existing_versions=True,
        )

    copy_runtime_bundle(
        source_model_dir=CANDIDATE_DIR,
        source_processed_dir=CANDIDATE_PROCESSED_DIR,
        target_model_dir=MODELS_ROOT,
        target_processed_dir=PROCESSED_DIR,
    )
    write_manifest(
        MODELS_ROOT,
        slot="stable",
        model_id=candidate_model_id,
        source_model_dir=candidate_source_dir,
        metrics_summary=collect_model_summary(MODELS_ROOT),
        extra={
            "promoted_at": _utc_now(),
            "processed_dir": str(PROCESSED_DIR.resolve()),
            "registry_model_name": registry_model_name,
            "registry_version": registry_version,
            "registry_stage": PROMOTE_TARGET_STAGE if registry_model_name else None,
            "run_id": candidate_manifest.get("run_id"),
            "runtime_bundle_path": candidate_manifest.get("runtime_bundle_path"),
            "previous_stable_model_id": previous_stable_manifest.get("model_id"),
            "previous_registry_model_name": previous_stable_manifest.get(
                "registry_model_name"
            ),
            "previous_registry_version": previous_stable_manifest.get(
                "registry_version"
            ),
            "previous_registry_stage": previous_stable_manifest.get(
                "registry_stage"
            ),
            "previous_run_id": previous_stable_manifest.get("run_id"),
            "previous_runtime_bundle_path": previous_stable_manifest.get(
                "runtime_bundle_path"
            ),
        },
    )
    wait_for_model_version(STABLE_URL, candidate_model_id)
    set_canary_percentage(0)
    clear_candidate_slot(stop_service=True)
    state = fetch_rollout_state()
    logger.info("Candidate promoted to stable | model_id=%s", candidate_model_id)
    return state


def manual_stage_candidate_from_stable(
    *,
    initial_percentage: int = 10,
    model_id: str | None = None,
) -> dict[str, Any]:
    bootstrap_runtime_layout()
    if not (MODELS_ROOT / MODEL_FILENAME).exists():
        raise FileNotFoundError(f"Stable model artifact not found at {MODELS_ROOT / MODEL_FILENAME}")

    stable_manifest = read_manifest(MODELS_ROOT) or {}
    stable_model_id = str(stable_manifest.get("model_id") or "stable-bootstrap")
    resolved_model_id = model_id or _random_id(f"{stable_model_id}_canary")
    return stage_candidate_model(
        model_id=resolved_model_id,
        source_model_dir=MODELS_ROOT,
        source_processed_dir=PROCESSED_DIR,
        initial_percentage=initial_percentage,
    )


def manual_train_and_stage_candidate(
    *,
    initial_percentage: int = 10,
    model_id: str | None = None,
    require_training: bool = False,
    allow_preprocess: bool = True,
) -> dict[str, Any]:
    bootstrap_runtime_layout()
    try:
        ensure_processed_training_artifacts(
            processed_dir=PROCESSED_DIR,
            raw_dataset_path=RAW_DATASET_PATH,
            allow_preprocess=allow_preprocess,
        )
    except FileNotFoundError as exc:
        if require_training:
            raise
        logger.warning(
            "Unable to retrain candidate locally: %s Falling back to staging a candidate from stable artifacts.",
            exc,
        )
        return manual_stage_candidate_from_stable(
            initial_percentage=initial_percentage,
            model_id=model_id,
        )

    resolved_model_id, version_dir, _, evaluation = train_candidate_model(
        processed_dir=PROCESSED_DIR,
        model_id=model_id,
    )
    if evaluation.get("overall_status") != "PASS":
        raise RuntimeError(
            f"Candidate {resolved_model_id} failed evaluation: {evaluation.get('overall_status')}"
        )
    return stage_candidate_model(
        model_id=resolved_model_id,
        source_model_dir=version_dir,
        source_processed_dir=PROCESSED_DIR,
        initial_percentage=initial_percentage,
    )


def manual_progressive_rollout(
    *,
    step_percentage: int = 10,
    step_wait_seconds: int = 45,
    auto_promote: bool = True,
) -> dict[str, Any]:
    state = fetch_rollout_state()
    if state["candidate"]["status"] != "ready":
        raise RuntimeError("No candidate model is staged for rollout.")

    current = int(state["rollout"]["candidate_percentage"])
    while current < 100:
        time.sleep(step_wait_seconds)
        current = min(100, current + max(step_percentage, 1))
        set_canary_percentage(current)
        log_rollout_state()

    if auto_promote:
        return promote_candidate_to_stable()
    return fetch_rollout_state()


def rollout_duration_seconds(
    *,
    initial_percentage: int,
    step_percentage: int,
    step_wait_seconds: int,
) -> int:
    remaining = max(0, 100 - initial_percentage)
    if remaining == 0:
        return step_wait_seconds
    steps = math.ceil(remaining / max(step_percentage, 1))
    return steps * step_wait_seconds
