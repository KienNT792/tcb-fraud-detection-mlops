"""
Model loader — singleton lifecycle for FraudDetector plus deployment metadata.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional


_THIS_FILE = Path(__file__).resolve()
_SERVING_DIR = _THIS_FILE.parent.parent
_PROJECT_ROOT = _SERVING_DIR.parent
_ML_PIPELINE_SRC = _PROJECT_ROOT / "ml_pipeline" / "src"

if str(_ML_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE_SRC))

from inference import FraudDetector  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR = str(_PROJECT_ROOT / "models")
_DEFAULT_PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")

MODELS_DIR = os.getenv("MODELS_DIR", _DEFAULT_MODELS_DIR)
PROCESSED_DIR = os.getenv("PROCESSED_DIR", _DEFAULT_PROCESSED_DIR)
MODEL_SLOT = os.getenv("MODEL_SLOT", "default")
ALLOW_EMPTY_MODEL = os.getenv("ALLOW_EMPTY_MODEL", "false").lower() == "true"
API_VERSION = "1.1.0"

_detector: Optional[FraudDetector] = None
_load_error: str | None = None
_last_loaded_at: str | None = None
_loaded_model_id: str | None = None


def _manifest_path(models_dir: str = MODELS_DIR) -> Path:
    return Path(models_dir) / "model_manifest.json"


def _read_manifest(models_dir: str = MODELS_DIR) -> dict[str, Any]:
    manifest_path = _manifest_path(models_dir)
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def _manifest_model_id(models_dir: str = MODELS_DIR) -> str | None:
    return _read_manifest(models_dir).get("model_id")


def _ensure_latest_model(required: bool) -> FraudDetector | None:
    current_model_id = _manifest_model_id()
    if _detector is None:
        detector = load_model(allow_empty=ALLOW_EMPTY_MODEL)
        if detector is None and required:
            raise RuntimeError(
                "FraudDetector not loaded. "
                "Ensure load_model() is called during API startup."
            )
        return detector

    if current_model_id and current_model_id != _loaded_model_id:
        logger.info(
            "Detected model manifest change for slot=%s: %s -> %s. Reloading.",
            MODEL_SLOT,
            _loaded_model_id,
            current_model_id,
        )
        return load_model(force_reload=True, allow_empty=ALLOW_EMPTY_MODEL)

    return _detector


def get_detector(required: bool = True) -> FraudDetector | None:
    return _ensure_latest_model(required=required)


def get_runtime_info() -> dict[str, Any]:
    manifest = _read_manifest()
    return {
        "model_slot": MODEL_SLOT,
        "models_dir": MODELS_DIR,
        "processed_dir": PROCESSED_DIR,
        "model_loaded": _detector is not None,
        "model_version": manifest.get("model_id"),
        "manifest": manifest,
        "allow_empty_model": ALLOW_EMPTY_MODEL,
        "load_error": _load_error,
        "loaded_at": _last_loaded_at,
    }


def load_model(
    models_dir: str = MODELS_DIR,
    processed_dir: str = PROCESSED_DIR,
    *,
    force_reload: bool = False,
    allow_empty: bool | None = None,
) -> FraudDetector | None:
    global _detector, _load_error, _last_loaded_at, _loaded_model_id

    if allow_empty is None:
        allow_empty = ALLOW_EMPTY_MODEL

    if force_reload:
        unload_model()

    if _detector is not None:
        logger.info("FraudDetector already loaded — reusing cached instance.")
        return _detector

    logger.info(
        "Loading FraudDetector — slot=%s | models_dir=%s | processed_dir=%s",
        MODEL_SLOT,
        models_dir,
        processed_dir,
    )
    try:
        _detector = FraudDetector(models_dir, processed_dir)
        health = _detector.health_check()
        _last_loaded_at = str(health.get("loaded_at"))
        _loaded_model_id = _manifest_model_id(models_dir)
        _load_error = None
        logger.info("FraudDetector loaded and cached for slot=%s.", MODEL_SLOT)
        return _detector
    except FileNotFoundError as exc:
        _detector = None
        _last_loaded_at = None
        _loaded_model_id = None
        _load_error = str(exc)
        if allow_empty:
            logger.warning(
                "No model artifacts available for slot=%s. "
                "Service will stay in standby mode: %s",
                MODEL_SLOT,
                exc,
            )
            return None
        raise


def reload_model() -> FraudDetector | None:
    return load_model(force_reload=True)


def unload_model() -> None:
    global _detector, _loaded_model_id
    _detector = None
    _loaded_model_id = None
    logger.info("FraudDetector unloaded for slot=%s.", MODEL_SLOT)
