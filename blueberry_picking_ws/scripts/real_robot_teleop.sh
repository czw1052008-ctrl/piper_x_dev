#!/usr/bin/env bash
# Keyboard teleop when real_robot_bringup.sh is already running.
# Optional: bash scripts/real_robot_teleop.sh --viz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
TELEOP_LINEAR_SPEED=0.03
TELEOP_ANGULAR_SPEED=10.0
START_VIZ=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --viz) START_VIZ=true; shift ;;
    --raw-viz) START_VIZ=true; RAW_VIZ=true; shift ;;
    -h|--help)
      echo "Usage: bash scripts/real_robot_teleop.sh [--viz] [--raw-viz]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
fi

set +u
# shellcheck disable=SC1091
source "${ROOT}/scripts/setup_env.sh" >/dev/null
export PYTHONNOUSERSITE=1
export RMW_FASTRTPS_USE_SHM=0
set -u 2>/dev/null || true

if [[ "${START_VIZ}" == "true" ]]; then
  VIZ_ARGS=()
  [[ "${RAW_VIZ:-false}" == "true" ]] && VIZ_ARGS+=(--raw)
  bash "${ROOT}/scripts/run_teleop_viz.sh" "${VIZ_ARGS[@]+"${VIZ_ARGS[@]}"}" || true
fi

cat <<EOF
================================================================================
  link6 teleop
  W/S up/down  A/D left/right  Q/E forward/back
  O/K roll  I/J pitch  U/H yaw   R home   Esc quit
  Viz topics: /perception/global|fine/detection_viz , /perception/wrist/depth_viz
================================================================================
EOF

exec ros2 run picking_bringup link6_teleop_node --ros-args \
  -p linear_speed_m_s:=${TELEOP_LINEAR_SPEED} \
  -p angular_speed_deg_s:=${TELEOP_ANGULAR_SPEED}
