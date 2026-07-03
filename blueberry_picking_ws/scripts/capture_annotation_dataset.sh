#!/usr/bin/env bash
# Capture RGB frames from wrist camera for YOLO annotation (one ROS session).
# Prereq: camera running, e.g. bash scripts/real_robot_bringup.sh --camera-only
#
# Usage:
#   bash scripts/capture_annotation_dataset.sh 50 2
#   bash scripts/capture_annotation_dataset.sh 10 2 datasets/blueberry/images_batch2
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COUNT="${1:-30}"
INTERVAL="${2:-2}"
OUT="${3:-${ROOT}/datasets/blueberry/images}"
if [[ "${OUT}" != /* ]]; then
  OUT="${ROOT}/${OUT}"
fi

mkdir -p "${OUT}"

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

echo "[capture] Saving ${COUNT} frames to ${OUT} (every ${INTERVAL}s)"
echo "[capture] Move arm / plant to vary distance (20-50cm), angle, lighting."

exec python3 "${ROOT}/scripts/capture_annotation_batch.py" \
  --out-dir "${OUT}" \
  --count "${COUNT}" \
  --interval "${INTERVAL}" \
  --startup-timeout 60
