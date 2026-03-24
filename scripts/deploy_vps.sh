#!/usr/bin/env bash

set -euo pipefail

DEPLOY_PATH_INPUT="${DEPLOY_PATH:-tcb-fraud-detection-mlops}"
DEPLOY_REF="${DEPLOY_REF:-main}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-$HOME/.tcb_deploy_env}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-tungb12ok}"

if [[ "${DEPLOY_PATH_INPUT}" = /* ]]; then
  DEPLOY_PATH="${DEPLOY_PATH_INPUT}"
else
  DEPLOY_PATH="$HOME/${DEPLOY_PATH_INPUT}"
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command on VPS: $cmd"
    exit 1
  fi
}

for cmd in git docker curl; do
  require_cmd "$cmd"
done

if ! docker compose version >/dev/null 2>&1; then
  echo "Missing required Docker Compose plugin on VPS."
  exit 1
fi

if [[ ! -d "$DEPLOY_PATH" ]]; then
  echo "Deploy path does not exist: $DEPLOY_PATH"
  echo "Clone the repository manually on the VPS first."
  exit 1
fi

cd "$DEPLOY_PATH"

if [[ ! -d .git ]]; then
  echo "No git repository found in $DEPLOY_PATH"
  echo "Clone the repository manually on the VPS first."
  exit 1
fi

echo "Deploy path: $DEPLOY_PATH"
echo "Branch: $DEPLOY_REF"
echo "Image tag: $IMAGE_TAG"

git fetch --all --prune
git checkout "$DEPLOY_REF"
git pull --ff-only origin "$DEPLOY_REF"

if [[ -f "$DEPLOY_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  . "$DEPLOY_ENV_FILE"
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Update secrets before real production use."
fi

if [[ -z "${DOCKERHUB_TOKEN:-}" ]]; then
  echo "Missing DOCKERHUB_TOKEN."
  echo "Set it in the shell or save it in $DEPLOY_ENV_FILE"
  exit 1
fi

set -a
. ./.env
set +a

HEALTH_ENDPOINTS=(
  "http://localhost:${FASTAPI_PORT:-8000}/health"
  "http://localhost:${MLFLOW_PORT:-5000}"
  "http://localhost:${AIRFLOW_PORT:-8080}/health"
  "http://localhost:${GRAFANA_PORT:-3000}/api/health"
)

printf '%s' "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

IMAGE_TAG="$IMAGE_TAG" docker compose pull fastapi-stable fastapi-candidate
IMAGE_TAG="$IMAGE_TAG" docker compose up -d --no-build

echo "Waiting for services to become healthy..."
sleep 20

for endpoint in "${HEALTH_ENDPOINTS[@]}"; do
  echo "Checking ${endpoint}..."
  for attempt in $(seq 1 10); do
    if curl --fail --silent --show-error "$endpoint" >/dev/null 2>&1; then
      echo "  OK ${endpoint}"
      break
    fi
    if [[ "$attempt" -eq 10 ]]; then
      echo "  FAIL ${endpoint}"
      exit 1
    fi
    sleep 10
  done
done

echo "Deploy completed successfully."
