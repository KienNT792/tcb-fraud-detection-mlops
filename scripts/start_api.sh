#!/usr/bin/env bash
set -euo pipefail

PROCESSED_DIR="${PROCESSED_DIR:-/app/data/processed}"
FEATURES_FILE="${PROCESSED_DIR}/features.json"

if [[ "${BOOTSTRAP_DEMO_ARTIFACTS:-false}" == "true" && ! -f "${FEATURES_FILE}" ]]; then
  python scripts/bootstrap_demo_artifacts.py --if-missing
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
