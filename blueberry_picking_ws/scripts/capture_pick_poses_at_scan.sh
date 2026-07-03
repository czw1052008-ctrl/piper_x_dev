#!/usr/bin/env bash
# Capture wrist-camera frames at teleop scan poses (same as pick patrol).
# Prereq: arm + camera bringup (bash scripts/real_robot_bringup.sh --no-wait)
#
# Usage:
#   bash scripts/capture_pick_poses_at_scan.sh [count] [interval_sec]
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COUNT="${1:-50}"
INTERVAL="${2:-2}"
OUT="${ROOT}/datasets/blueberry/images_batch3"

conda deactivate 2>/dev/null || true
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
export RMW_FASTRTPS_USE_SHM=0
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"
if [[ -f "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash"
fi

mkdir -p "${OUT}"

echo "[capture-scan] ${COUNT} frames at teleop poses -> ${OUT} (interval ${INTERVAL}s)"
exec python3 "${ROOT}/scripts/capture_pick_poses_dataset.py" \
  --out-dir "${OUT}" \
  --prefix b3_ \
  --count "${COUNT}" \
  --interval "${INTERVAL}" \
  --startup-timeout 90 \
  --move-timeout 60
