#!/usr/bin/env bash
# Interactive minimal capture for fixed (global mono) camera.
#
# Prereq: fixed camera running
#   bash scripts/real_robot_bringup.sh --fixed-cam --camera-only
#
# Usage:
#   bash scripts/capture_fixed_mono_dataset.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash 2>/dev/null || source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash" 2>/dev/null || true
if [[ -f "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash"
fi
set -u

exec python3 "${ROOT}/scripts/capture_fixed_mono_guide.py" "$@"
