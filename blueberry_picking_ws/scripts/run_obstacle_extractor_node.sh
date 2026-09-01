#!/usr/bin/env bash
# Depth fusion → /perception/obstacles (non-fruit, non-arm occupancy spheres).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/config/real_robot.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT}/config/real_robot.env"
fi
set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
set -u 2>/dev/null || true

export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
exec python3 "${ROOT}/scripts/obstacle_extractor_node.py" "$@"
