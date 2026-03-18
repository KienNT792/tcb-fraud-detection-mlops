"""
Model loader — singleton pattern for FraudDetector.

Ensures the FraudDetector is instantiated exactly once at API startup
and reused across all requests. Uses FastAPI lifespan context manager
for clean startup/shutdown lifecycle management.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from infrastructure.pipeline import PIPELINE_CONFIG
from .rollout import CanaryRolloutManager, FraudDetector

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR    = str(PIPELINE_CONFIG.paths.models_dir)
_DEFAULT_PROCESSED_DIR = str(PIPELINE_CONFIG.paths.processed_dir)
_DEFAULT_ROLLOUT_CONFIG_PATH = str(PIPELINE_CONFIG.serving.rollout_config_path)

MODELS_DIR    = os.getenv("MODELS_DIR",    _DEFAULT_MODELS_DIR)
PROCESSED_DIR = os.getenv("PROCESSED_DIR", _DEFAULT_PROCESSED_DIR)
ROLLOUT_CONFIG_PATH = os.getenv("ROLLOUT_CONFIG_PATH", _DEFAULT_ROLLOUT_CONFIG_PATH)

_rollout_manager: Optional[CanaryRolloutManager] = None

API_VERSION = "1.1.0"


def get_detector() -> FraudDetector:
    return get_rollout_manager().get_stable_detector()


def get_rollout_manager() -> CanaryRolloutManager:
    if _rollout_manager is None:
        raise RuntimeError(
            "Serving manager not loaded. "
            "Ensure load_model() is called during API startup."
        )
    return _rollout_manager


def load_model(
    models_dir: str = MODELS_DIR,
    processed_dir: str = PROCESSED_DIR,
    rollout_config_path: str = ROLLOUT_CONFIG_PATH,
) -> CanaryRolloutManager:
    global _rollout_manager

    if _rollout_manager is not None:
        logger.info("Serving manager already loaded — reusing cached instance.")
        return _rollout_manager

    logger.info(
        "Loading serving manager — models_dir=%s | processed_dir=%s | rollout_config=%s",
        models_dir, processed_dir, rollout_config_path,
    )
    _rollout_manager = CanaryRolloutManager(
        stable_models_dir=models_dir,
        stable_processed_dir=processed_dir,
        config_path=rollout_config_path,
    )
    logger.info("Serving manager loaded and cached.")
    return _rollout_manager


def unload_model() -> None:
    """Release the singleton — used during testing or graceful shutdown."""
    global _rollout_manager
    _rollout_manager = None
    logger.info("Serving manager unloaded.")
