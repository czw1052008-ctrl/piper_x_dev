#!/usr/bin/env bash
# Keyboard teleop when real_robot_bringup.sh is already running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
TELEOP_LINEAR_SPEED=0.03
TELEOP_ANGULAR_SPEED=10.0

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
fi

set +u
# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh" >/dev/null
set -u 2>/dev/null || true

exec ros2 run picking_bringup link6_teleop_node --ros-args \
  -p linear_speed_m_s:=${TELEOP_LINEAR_SPEED} \
  -p angular_speed_deg_s:=${TELEOP_ANGULAR_SPEED}
