#!/usr/bin/env bash
# Visualize FoundationPose blueberry detections on live DaBai RGB (no HSV boxes).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda deactivate 2>/dev/null || true
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL 2>/dev/null || true

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"
if [[ -f "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash"
fi

SAVE_DIR="${ROOT}/log/real_robot/capture"
mkdir -p "${SAVE_DIR}"

if ! ros2 service list 2>/dev/null | grep -q '/trigger_fine_detection'; then
  echo "[viz] ERROR: /trigger_fine_detection not available." >&2
  echo "  Camera + perception must be running. Either:" >&2
  echo "    bash scripts/real_robot_bringup.sh --camera-only" >&2
  echo "  or (camera already up):" >&2
  echo "    bash scripts/run_perception_only.sh" >&2
  exit 1
fi

echo "[viz] FoundationPose overlay — green LOCK = berry, magenta square = planned cup contact"
echo "[viz] Topic: /camera_wrist/color/detection_viz"
echo "[viz] View:  ros2 run rqt_image_view rqt_image_view /camera_wrist/color/detection_viz"
echo "[viz] Save:  ${SAVE_DIR}/latest_detection_viz.png"
echo "[viz] Ctrl+C to stop"

exec /usr/bin/python3 "${ROOT}/scripts/visualize_blueberry_detection.py" \
  --save-dir "${SAVE_DIR}" \
  "$@"
