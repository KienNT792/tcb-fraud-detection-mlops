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
DEPLOY_REEXEC="${DEPLOY_REEXEC:-0}"
APP_IMAGE_REPOSITORY="${APP_IMAGE_REPOSITORY:-tungb12ok/tcb-detect-credit}"
SYNC_RUNTIME_BUNDLE_FROM_REGISTRY="${SYNC_RUNTIME_BUNDLE_FROM_REGISTRY:-true}"
MLFLOW_REGISTERED_MODEL_NAME="${MLFLOW_REGISTERED_MODEL_NAME:-tcb-fraud-xgboost}"
MLFLOW_DEPLOY_STAGE="${MLFLOW_DEPLOY_STAGE:-Production}"
MLFLOW_RUNTIME_BUNDLE_PATH="${MLFLOW_RUNTIME_BUNDLE_PATH:-runtime_bundle}"
ALLOW_REPO_BOOTSTRAP_ON_EMPTY_REGISTRY="${ALLOW_REPO_BOOTSTRAP_ON_EMPTY_REGISTRY:-true}"
BOOTSTRAP_RUNTIME_BUNDLE_DIR="${BOOTSTRAP_RUNTIME_BUNDLE_DIR:-bootstrap_runtime_bundle}"

if [[ "${DEPLOY_PATH_INPUT}" = /* ]]; then
  DEPLOY_PATH="${DEPLOY_PATH_INPUT}"
else
  DEPLOY_PATH="$HOME/${DEPLOY_PATH_INPUT}"
fi

candidate_manifest_path() {
  printf '%s\n' "$DEPLOY_PATH/models/deployments/candidate/model_manifest.json"
}

candidate_service_enabled() {
  [[ -f "$(candidate_manifest_path)" ]]
}

compose_stack() {
  local -a profile_args=()
  if candidate_service_enabled; then
    profile_args=(--profile candidate)
  fi
  IMAGE_TAG="$IMAGE_TAG" docker compose "${profile_args[@]}" "$@"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command on VPS: $cmd"
    exit 1
  fi
}

show_health_failure_context() {
  local failed_name="$1"
  local failed_endpoint="$2"

  echo "Health check failed for ${failed_name}: ${failed_endpoint}"
  echo "docker compose ps:"
  compose_stack ps || true

  echo "Recent logs: fastapi-stable"
  compose_stack logs --tail=120 fastapi-stable || true

  if candidate_service_enabled; then
    echo "Recent logs: fastapi-candidate"
    compose_stack logs --tail=120 fastapi-candidate || true
  fi

  echo "Recent logs: loadbalancer"
  compose_stack logs --tail=120 loadbalancer || true
}

wait_for_endpoint() {
  local name="$1"
  local endpoint="$2"
  local max_attempts="${3:-18}"
  local sleep_seconds="${4:-5}"

  echo "Waiting for ${name} at ${endpoint}..."
  for attempt in $(seq 1 "$max_attempts"); do
    if curl --fail --silent --show-error "$endpoint" >/dev/null 2>&1; then
      echo "  ${name} is ready."
      return 0
    fi
    sleep "$sleep_seconds"
  done

  echo "Timed out waiting for ${name} at ${endpoint}"
  return 1
}

show_optional_health_warning() {
  local service_name="$1"
  local endpoint="$2"

  echo "Non-critical health check failed for ${service_name}: ${endpoint}"
  echo "Recent logs: ${service_name}"
  compose_stack logs --tail=120 "$service_name" || true
}

bootstrap_runtime_bundle_from_repo() {
  local processed_src="$DEPLOY_PATH/$BOOTSTRAP_RUNTIME_BUNDLE_DIR/processed"
  local required_model_files=(
    "$DEPLOY_PATH/models/xgb_fraud_model.joblib"
    "$DEPLOY_PATH/models/metrics.json"
    "$DEPLOY_PATH/models/feature_importance.csv"
  )
  local required_processed_files=(
    "features.json"
    "customer_stats.parquet"
    "segment_label_map.json"
    "amount_median_train.json"
    "categorical_maps.json"
  )
  local missing_files=()
  local file=""

  for file in "${required_model_files[@]}"; do
    if [[ ! -f "$file" ]]; then
      missing_files+=("$file")
    fi
  done

  for file in "${required_processed_files[@]}"; do
    if [[ ! -f "$processed_src/$file" ]]; then
      missing_files+=("$processed_src/$file")
    fi
  done

  if [[ "${#missing_files[@]}" -gt 0 ]]; then
    echo "Repository bootstrap runtime bundle is incomplete:"
    printf '  %s\n' "${missing_files[@]}"
    return 1
  fi

  docker run --rm \
    -u root \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e BOOTSTRAP_RUNTIME_BUNDLE_DIR="$BOOTSTRAP_RUNTIME_BUNDLE_DIR" \
    -v "$DEPLOY_PATH:/workspace" \
    -w /workspace \
    "${APP_IMAGE_REPOSITORY}:${IMAGE_TAG}" \
    sh -lc '
      set -e
      mkdir -p /workspace/data/processed
      cp "/workspace/${BOOTSTRAP_RUNTIME_BUNDLE_DIR}/processed/"* /workspace/data/processed/
      chown -R "${HOST_UID}:${HOST_GID}" /workspace/data/processed
      chmod -R u+rwX /workspace/data/processed
    '

  echo "Bootstrapped runtime bundle from repository artifacts."
}

bootstrap_mlflow_registry_from_repo() {
  local model_artifact_dir="$DEPLOY_PATH/$BOOTSTRAP_RUNTIME_BUNDLE_DIR/mlflow_model"
  local processed_dir="$DEPLOY_PATH/$BOOTSTRAP_RUNTIME_BUNDLE_DIR/processed"

  if [[ ! -f "$model_artifact_dir/MLmodel" ]]; then
    echo "Missing bootstrap MLflow model directory: $model_artifact_dir"
    return 1
  fi

  if [[ ! -f "$DEPLOY_PATH/models/xgb_fraud_model.joblib" ]]; then
    echo "Missing bootstrap model artifact: $DEPLOY_PATH/models/xgb_fraud_model.joblib"
    return 1
  fi

  echo "Bootstrapping first MLflow Registry version from repository artifacts."
  docker run --rm \
    --network host \
    --user "$(id -u):$(id -g)" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:${MINIO_API_PORT:-9000}" \
    -v "$DEPLOY_PATH:/workspace" \
    -w /workspace \
    ghcr.io/mlflow/mlflow:v2.14.3 \
    python scripts/runtime_bundle_registry.py \
      --tracking-uri "http://127.0.0.1:${MLFLOW_PORT:-5000}" \
      bootstrap-artifacts \
      --model-name "$MLFLOW_REGISTERED_MODEL_NAME" \
      --stage "$MLFLOW_DEPLOY_STAGE" \
      --model-artifact-dir "/workspace/${BOOTSTRAP_RUNTIME_BUNDLE_DIR}/mlflow_model" \
      --models-dir /workspace/models \
      --processed-dir "/workspace/${BOOTSTRAP_RUNTIME_BUNDLE_DIR}/processed"
}

prune_disabled_candidate_service() {
  if candidate_service_enabled; then
    return 0
  fi

  IMAGE_TAG="$IMAGE_TAG" docker compose --profile candidate stop fastapi-candidate >/dev/null 2>&1 || true
  IMAGE_TAG="$IMAGE_TAG" docker compose --profile candidate rm -f fastapi-candidate >/dev/null 2>&1 || true
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

current_rev="$(git rev-parse HEAD)"

git fetch --all --prune
git checkout "$DEPLOY_REF"
git pull --ff-only origin "$DEPLOY_REF"

updated_rev="$(git rev-parse HEAD)"

if [[ "$DEPLOY_REEXEC" != "1" && "$current_rev" != "$updated_rev" ]]; then
  echo "Repository updated from $current_rev to $updated_rev. Re-executing deploy script."
  export DEPLOY_REEXEC=1
  exec bash ./scripts/deploy_vps.sh
fi

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

AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-$MINIO_ROOT_USER}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-$MINIO_ROOT_PASSWORD}"
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

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

compose_stack config >/dev/null

printf '%s' "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

echo "Pulling application image: ${APP_IMAGE_REPOSITORY}:${IMAGE_TAG}"
docker pull "${APP_IMAGE_REPOSITORY}:${IMAGE_TAG}"

compose_stack pull
compose_stack up -d minio minio-init mlflow

wait_for_endpoint "mlflow" "http://127.0.0.1:${MLFLOW_PORT:-5000}" 24 5

if [[ "$SYNC_RUNTIME_BUNDLE_FROM_REGISTRY" == "true" ]]; then
  echo "Syncing runtime bundle from MLflow Registry stage=${MLFLOW_DEPLOY_STAGE}"
  set +e
  runtime_bundle_output="$(
    docker run --rm \
      --network host \
      --user "$(id -u):$(id -g)" \
      -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
      -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
      -e MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:${MINIO_API_PORT:-9000}" \
      -v "$DEPLOY_PATH:/workspace" \
      -w /workspace \
      ghcr.io/mlflow/mlflow:v2.14.3 \
      python scripts/runtime_bundle_registry.py \
        --tracking-uri "http://127.0.0.1:${MLFLOW_PORT:-5000}" \
        download-stage \
        --model-name "$MLFLOW_REGISTERED_MODEL_NAME" \
        --stage "$MLFLOW_DEPLOY_STAGE" \
        --artifact-path "$MLFLOW_RUNTIME_BUNDLE_PATH" \
        --output-root /workspace 2>&1
  )"
  runtime_bundle_status=$?
  set -e
  printf '%s\n' "$runtime_bundle_output"

  if [[ "$runtime_bundle_status" -ne 0 ]]; then
    if [[ "$ALLOW_REPO_BOOTSTRAP_ON_EMPTY_REGISTRY" == "true" ]] && \
      [[ "$runtime_bundle_output" == *"Registered Model with name=${MLFLOW_REGISTERED_MODEL_NAME} not found"* || \
         "$runtime_bundle_output" == *"No model version found for name=${MLFLOW_REGISTERED_MODEL_NAME} stage=${MLFLOW_DEPLOY_STAGE}"* ]]; then
      echo "MLflow Registry is empty for ${MLFLOW_REGISTERED_MODEL_NAME}/${MLFLOW_DEPLOY_STAGE}. Bootstrapping the first version from repository artifacts."
      bootstrap_mlflow_registry_from_repo
      runtime_bundle_output="$(
        docker run --rm \
          --network host \
          --user "$(id -u):$(id -g)" \
          -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
          -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
          -e MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:${MINIO_API_PORT:-9000}" \
          -v "$DEPLOY_PATH:/workspace" \
          -w /workspace \
          ghcr.io/mlflow/mlflow:v2.14.3 \
          python scripts/runtime_bundle_registry.py \
            --tracking-uri "http://127.0.0.1:${MLFLOW_PORT:-5000}" \
            download-stage \
            --model-name "$MLFLOW_REGISTERED_MODEL_NAME" \
            --stage "$MLFLOW_DEPLOY_STAGE" \
            --artifact-path "$MLFLOW_RUNTIME_BUNDLE_PATH" \
            --output-root /workspace 2>&1
      )"
      printf '%s\n' "$runtime_bundle_output"
    else
    echo "Failed to sync runtime bundle from MLflow Registry."
    echo "If the registered model ${MLFLOW_REGISTERED_MODEL_NAME} does not exist yet,"
    echo "bootstrap it once from a machine that already has models/ and data/processed/:"
    echo "  python scripts/runtime_bundle_registry.py --tracking-uri http://127.0.0.1:${MLFLOW_PORT:-5000} bootstrap-run --run-id <training_run_id> --model-name ${MLFLOW_REGISTERED_MODEL_NAME} --stage ${MLFLOW_DEPLOY_STAGE}"
    echo "Or bootstrap the committed repo artifacts with:"
    echo "  python scripts/runtime_bundle_registry.py --tracking-uri http://127.0.0.1:${MLFLOW_PORT:-5000} bootstrap-artifacts --model-name ${MLFLOW_REGISTERED_MODEL_NAME} --stage ${MLFLOW_DEPLOY_STAGE}"
    echo "If the registered model exists but was trained before runtime bundles were published,"
    echo "backfill the latest stage bundle with:"
    echo "  python scripts/runtime_bundle_registry.py --tracking-uri http://127.0.0.1:${MLFLOW_PORT:-5000} publish-stage --model-name ${MLFLOW_REGISTERED_MODEL_NAME} --stage ${MLFLOW_DEPLOY_STAGE}"
    exit 1
    fi
  fi
fi

compose_stack up -d --no-build --remove-orphans
prune_disabled_candidate_service

HEALTH_CHECKS=(
  "fastapi-stable|http://127.0.0.1:${FASTAPI_STABLE_PORT:-8002}/health|critical|10|10"
  "loadbalancer|http://localhost:${FASTAPI_PORT:-8000}/health|critical|10|10"
  "mlflow|http://localhost:${MLFLOW_PORT:-5000}|critical|10|10"
  "airflow|http://localhost:${AIRFLOW_PORT:-8080}/health|optional|18|10"
  "alertmanager|http://localhost:${ALERTMANAGER_PORT:-9093}/-/ready|optional|12|10"
  "grafana|http://localhost:${GRAFANA_PORT:-3000}/api/health|optional|12|10"
)

if candidate_service_enabled; then
  HEALTH_CHECKS+=(
    "fastapi-candidate|http://127.0.0.1:${FASTAPI_CANDIDATE_PORT:-8003}/health|critical|10|10"
  )
fi

echo "Waiting for services to become healthy..."
sleep 20

for check in "${HEALTH_CHECKS[@]}"; do
  check_name="${check%%|*}"
  rest="${check#*|}"
  endpoint="${rest%%|*}"
  rest="${rest#*|}"
  check_mode="${rest%%|*}"
  rest="${rest#*|}"
  max_attempts="${rest%%|*}"
  sleep_seconds="${rest#*|}"

  echo "Checking ${check_name} (${endpoint})..."
  for attempt in $(seq 1 "$max_attempts"); do
    if curl --fail --silent --show-error "$endpoint" >/dev/null 2>&1; then
      echo "  OK ${check_name}"
      break
    fi
    if [[ "$attempt" -eq "$max_attempts" ]]; then
      if [[ "$check_mode" == "critical" ]]; then
        echo "  FAIL ${check_name}"
        show_health_failure_context "$check_name" "$endpoint"
        exit 1
      fi
      echo "  WARN ${check_name}"
      show_optional_health_warning "$check_name" "$endpoint"
      break
    fi
    sleep "$sleep_seconds"
  done
done

echo "Deploy completed successfully."
