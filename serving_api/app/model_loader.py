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


#   tcb-fraud-detection-mlops/
#   ├── data/
#   ├── models/
#   ├── ml_pipeline/src/inference.py
#   └── serving_api/app/model_loader.py

_THIS_FILE   = Path(__file__).resolve()          # .../serving_api/app/model_loader.py
_SERVING_DIR = _THIS_FILE.parent.parent          # .../serving_api/
_PROJECT_ROOT = _SERVING_DIR.parent             # .../tcb-fraud-detection-mlops/
_ML_PIPELINE_SRC = _PROJECT_ROOT / "ml_pipeline" / "src"

if str(_ML_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_PIPELINE_SRC))

from inference import FraudDetector  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR    = str(_PROJECT_ROOT / "models")
_DEFAULT_PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")

MODELS_DIR    = os.getenv("MODELS_DIR",    _DEFAULT_MODELS_DIR)
PROCESSED_DIR = os.getenv("PROCESSED_DIR", _DEFAULT_PROCESSED_DIR)

_detector: Optional[FraudDetector] = None

API_VERSION = "1.0.0"


def get_detector() -> FraudDetector:

    if _detector is None:
        raise RuntimeError(
            "FraudDetector not loaded. "
            "Ensure load_model() is called during API startup."
        )
    return _detector


def load_model(
    models_dir: str = MODELS_DIR,
    processed_dir: str = PROCESSED_DIR,
) -> FraudDetector:
    global _detector

    if _detector is not None:
        logger.info("FraudDetector already loaded — reusing cached instance.")
        return _detector

    logger.info(
        "Loading FraudDetector — models_dir=%s | processed_dir=%s",
        models_dir, processed_dir,
    )
    _detector = FraudDetector(models_dir, processed_dir)
    logger.info("FraudDetector loaded and cached.")
    return _detector


def unload_model() -> None:
    """Release the singleton — used during testing or graceful shutdown."""
    global _detector
    _detector = None
    logger.info("FraudDetector unloaded.")