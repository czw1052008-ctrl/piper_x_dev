#!/usr/bin/env bash
# Occupancy + ESDF map from fixed camera depth.
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
exec /usr/bin/python3 "${ROOT}/scripts/occupancy_map_node.py" \
  --ros-args \
  -p end_effector:="${END_EFFECTOR:-agx_gripper_v1}" \
  "$@"
