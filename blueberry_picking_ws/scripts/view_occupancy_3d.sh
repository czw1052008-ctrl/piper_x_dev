#!/usr/bin/env bash
# Open RViz with 3D occupancy (base_link hemisphere layers).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "${ROOT}/config/real_robot.env" 2>/dev/null || true
set +u
source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"
set -u
export DISPLAY="${DISPLAY:-:1}"
exec rviz2 -d "${ROOT}/config/occupancy_3d.rviz"
