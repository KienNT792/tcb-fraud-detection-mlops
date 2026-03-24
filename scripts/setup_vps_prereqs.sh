#!/usr/bin/env bash

set -euo pipefail

TARGET_USER="${1:-${SUDO_USER:-$USER}}"
DEPLOY_PATH_INPUT="${2:-}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this script with sudo."
  echo "Example: sudo bash scripts/setup_vps_prereqs.sh ${TARGET_USER} /opt/tcb-fraud-detection-mlops"
  exit 1
fi

if ! id "${TARGET_USER}" >/dev/null 2>&1; then
  echo "User not found: ${TARGET_USER}"
  exit 1
fi

if [[ -n "${DEPLOY_PATH_INPUT}" ]]; then
  if [[ "${DEPLOY_PATH_INPUT}" = /* ]]; then
    DEPLOY_PATH="${DEPLOY_PATH_INPUT}"
  else
    DEPLOY_PATH="$(getent passwd "${TARGET_USER}" | cut -d: -f6)/${DEPLOY_PATH_INPUT}"
  fi
else
  DEPLOY_PATH="$(getent passwd "${TARGET_USER}" | cut -d: -f6)/tcb-fraud-detection-mlops"
fi

echo "Target user: ${TARGET_USER}"
echo "Deploy path: ${DEPLOY_PATH}"

export DEBIAN_FRONTEND=noninteractive

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot detect Linux distribution: /etc/os-release not found"
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release

case "${ID:-}" in
  debian|ubuntu)
    ;;
  *)
    echo "Unsupported distribution for this script: ${ID:-unknown}"
    echo "Use the official Docker docs for your distribution."
    exit 1
    ;;
esac

apt-get update
apt-get install -y ca-certificates curl git

apt-get remove -y docker.io docker-compose docker-doc podman-docker containerd runc || true

install -m 0755 -d /etc/apt/keyrings
curl -fsSL "https://download.docker.com/linux/${ID}/gpg" -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

if [[ "${ID}" == "ubuntu" ]]; then
  DOCKER_REPO_URL="https://download.docker.com/linux/ubuntu"
else
  DOCKER_REPO_URL="https://download.docker.com/linux/debian"
fi

cat >/etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: ${DOCKER_REPO_URL}
Suites: ${VERSION_CODENAME}
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
usermod -aG docker "${TARGET_USER}"

mkdir -p "${DEPLOY_PATH}"
chown -R "${TARGET_USER}:${TARGET_USER}" "${DEPLOY_PATH}"

echo
echo "Installed prerequisites:"
echo "- git"
echo "- curl"
echo "- docker"
echo "- docker compose plugin"
echo
echo "Verification commands:"
echo "  sudo -u ${TARGET_USER} git --version"
echo "  sudo -u ${TARGET_USER} curl --version"
echo "  docker --version"
echo "  docker compose version"
echo
echo "Next steps:"
echo "1. Log out and log back in for the docker group to apply to ${TARGET_USER}."
echo "2. Make sure ~/.ssh/authorized_keys for ${TARGET_USER} contains the public key matching SSH_DEPLOY_KEY."
echo "3. Rerun the GitHub Actions deploy workflow."
