#!/usr/bin/env bash
# RViz: P3 z-slice polygons (berry/branch/rigid/ego), not spheres.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/real_robot.env" 2>/dev/null || true
set +u
source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"
set -u
export DISPLAY="${DISPLAY:-:1}"
exec rviz2 -d "${ROOT}/config/semantic_obstacles.rviz"
