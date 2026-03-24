#!/usr/bin/env bash

set -euo pipefail

DEPLOY_PATH_INPUT="${DEPLOY_PATH:-tcb-fraud-detection-mlops}"
DEPLOY_REF="${DEPLOY_REF:-main}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DEPLOY_IMAGE_TAG="$IMAGE_TAG"
DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-}"
DEPLOY_ENV_B64="${DEPLOY_ENV_B64:-}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-tungb12ok}"
GIT_REMOTE_URL="${GIT_REMOTE_URL:-}"

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

sync_runtime_env_file() {
  if [[ -n "$DEPLOY_ENV_B64" ]]; then
    echo "Writing .env from DEPLOY_ENV_B64"
    printf '%s' "$DEPLOY_ENV_B64" | base64 --decode > .env
    chmod 600 .env || true
    return
  fi

  if [[ -f .env ]]; then
    echo "Using existing .env"
    return
  fi

  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo "Created .env from .env.example. Update secrets before real production use."
    return
  fi

  echo "Missing .env and .env.example in $DEPLOY_PATH"
  exit 1
}

resolve_deploy_env_file() {
  local candidates=()
  local user_home=""

  if [[ -n "${HOME:-}" ]]; then
    candidates+=("$HOME/.tcb_deploy_env")
  fi

  if user_home="$(getent passwd "$(id -un)" | cut -d: -f6 2>/dev/null)"; then
    if [[ -n "$user_home" ]]; then
      candidates+=("$user_home/.tcb_deploy_env")
    fi
  fi

  candidates+=("$DEPLOY_PATH/.tcb_deploy_env")

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

for cmd in base64 git docker curl; do
  require_cmd "$cmd"
done

if ! docker compose version >/dev/null 2>&1; then
  echo "Missing required Docker Compose plugin on VPS."
  exit 1
fi

if [[ ! -d "$DEPLOY_PATH" ]]; then
  if [[ -z "$GIT_REMOTE_URL" ]]; then
    echo "Deploy path does not exist: $DEPLOY_PATH"
    echo "Set GIT_REMOTE_URL to allow automatic bootstrap, or clone the repository manually first."
    exit 1
  fi

  echo "Bootstrapping repository into $DEPLOY_PATH"
  mkdir -p "$(dirname "$DEPLOY_PATH")"
  git clone --branch "$DEPLOY_REF" --single-branch "$GIT_REMOTE_URL" "$DEPLOY_PATH"
fi

cd "$DEPLOY_PATH"

if [[ ! -d .git ]]; then
  if [[ -z "$GIT_REMOTE_URL" ]]; then
    echo "No git repository found in $DEPLOY_PATH"
    echo "Set GIT_REMOTE_URL to allow automatic bootstrap, or clone the repository manually first."
    exit 1
  fi

  if [[ -n "$(find "$DEPLOY_PATH" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Deploy path exists but is not a git repository: $DEPLOY_PATH"
    exit 1
  fi

  echo "Bootstrapping repository into $DEPLOY_PATH"
  (
    cd "$(dirname "$DEPLOY_PATH")"
    git clone --branch "$DEPLOY_REF" --single-branch "$GIT_REMOTE_URL" "$(basename "$DEPLOY_PATH")"
  )
  cd "$DEPLOY_PATH"
fi

echo "Deploy path: $DEPLOY_PATH"
echo "Branch: $DEPLOY_REF"
echo "Image tag: $DEPLOY_IMAGE_TAG"
echo "Git remote: ${GIT_REMOTE_URL:-<existing remote>}"
echo "SSH user: $(id -un)"
echo "HOME: ${HOME:-<unset>}"

git fetch --all --prune
git checkout "$DEPLOY_REF"
git pull --ff-only origin "$DEPLOY_REF"

if [[ -z "$DEPLOY_ENV_FILE" ]]; then
  DEPLOY_ENV_FILE="$(resolve_deploy_env_file || true)"
fi

if [[ -n "$DEPLOY_ENV_FILE" && -f "$DEPLOY_ENV_FILE" ]]; then
  echo "Loading deploy env file: $DEPLOY_ENV_FILE"
  # shellcheck disable=SC1090
  . "$DEPLOY_ENV_FILE"
else
  echo "No deploy env file found."
fi

sync_runtime_env_file

set -a
. ./.env
set +a

# Keep the immutable image tag passed from the workflow.
# Local .env may keep IMAGE_TAG=local for development, but CD must deploy the pushed SHA tag.
IMAGE_TAG="$DEPLOY_IMAGE_TAG"
export IMAGE_TAG

required_env_vars=(
  MINIO_ROOT_USER
  MINIO_ROOT_PASSWORD
  MINIO_BUCKET
  GRAFANA_ADMIN_USER
  GRAFANA_ADMIN_PASSWORD
  AIRFLOW_UID
  FASTAPI_PORT
  MINIO_API_PORT
  MINIO_CONSOLE_PORT
  MLFLOW_PORT
  AIRFLOW_PORT
  PROMETHEUS_PORT
  GRAFANA_PORT
  CADVISOR_PORT
)

missing_env_vars=()

for var_name in "${required_env_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    missing_env_vars+=("$var_name")
  fi
done

if [[ "${#missing_env_vars[@]}" -gt 0 ]]; then
  echo "Missing required values in .env: ${missing_env_vars[*]}"
  exit 1
fi

if [[ -z "${DOCKERHUB_TOKEN:-}" ]]; then
  echo "Missing DOCKERHUB_TOKEN."
  echo "The token exported in an old shell does not persist to a new SSH session."
  echo "Save it in one of these files instead:"
  if [[ -n "${HOME:-}" ]]; then
    echo "  - $HOME/.tcb_deploy_env"
  fi
  echo "  - $DEPLOY_PATH/.tcb_deploy_env"
  if user_home="$(getent passwd "$(id -un)" | cut -d: -f6 2>/dev/null)"; then
    if [[ -n "$user_home" ]]; then
      echo "  - $user_home/.tcb_deploy_env"
    fi
  fi
  echo "Example:"
  echo "  cat > ~/.tcb_deploy_env <<'EOF'"
  echo "  DOCKERHUB_USERNAME=$DOCKERHUB_USERNAME"
  echo "  DOCKERHUB_TOKEN=your_dockerhub_token"
  echo "  EOF"
  echo "  chmod 600 ~/.tcb_deploy_env"
  exit 1
fi

IMAGE_TAG="$IMAGE_TAG" docker compose config >/dev/null

HEALTH_ENDPOINTS=(
  "http://localhost:${FASTAPI_PORT:-8000}/health"
  "http://localhost:${MLFLOW_PORT:-5000}"
  "http://localhost:${AIRFLOW_PORT:-8080}/health"
  "http://localhost:${GRAFANA_PORT:-3000}/api/health"
)

printf '%s' "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

IMAGE_TAG="$IMAGE_TAG" docker compose pull
IMAGE_TAG="$IMAGE_TAG" docker compose up -d --no-build --remove-orphans

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
