#!/usr/bin/env bash
# P0b: trajectory_executor (tool_trajectory_4s → IK → joint_cmd).
# With --drive-arm, sends FollowJointTrajectory.
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

ARGS=(
  --end-effector suction_cup_v1
)
# Pass-through: add --drive-arm for live arm.
ARGS+=("$@")

exec python3 "${ROOT}/scripts/trajectory_executor_node.py" "${ARGS[@]}"
