#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

cmd=(
  python -m monitoring.simulator.run_simulation
  --base-url "${SIMULATOR_BASE_URL:-http://35.222.198.13:8000}"
  --scenario "${SIMULATOR_SCENARIO:-moderate_drift}"
  --warmup-samples "${SIMULATOR_WARMUP_SAMPLES:-120}"
  --requests "${SIMULATOR_REQUESTS:-300}"
  --rps "${SIMULATOR_RPS:-2}"
  --batch-size "${SIMULATOR_BATCH_SIZE:-20}"
  --mode "${SIMULATOR_MODE:-batch}"
  --check-metrics
  --check-drift
  --check-prometheus
)

if [[ -n "${SIMULATOR_PROMETHEUS_URL:-}" ]]; then
  cmd+=(--prometheus-url "${SIMULATOR_PROMETHEUS_URL}")
fi

cmd+=("$@")
"${cmd[@]}"
