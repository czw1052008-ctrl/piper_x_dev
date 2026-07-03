#!/usr/bin/env bash
# Real-robot only: verify DaBai RGB-D capture + FoundationPose fine detection.
# NO simulation. Requires arm + camera + TF + fine_detector_node.
#
# Usage:
#   # Terminal 1 — start stack (or already running):
#   bash scripts/real_robot_bringup.sh --perception
#
#   # Terminal 2 — capture + detect + print 3D poses:
#   bash scripts/run_real_fine_detection_test.sh
#
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log/real_robot"
CAPTURE_DIR="${LOG_DIR}/capture"
mkdir -p "${CAPTURE_DIR}"

conda deactivate 2>/dev/null || true
# Avoid conda NumPy breaking ROS cv_bridge / system python.
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL 2>/dev/null || true
PYTHON=/usr/bin/python3
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"
if [[ -f "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash"
fi

COLOR_TOPIC="/camera_wrist/color/image_raw"
DEPTH_TOPIC="/camera_wrist/depth/image_raw"

echo "[real_fp] Checking DaBai RGB-D topics ..."
if ! timeout 12 ros2 topic echo "${COLOR_TOPIC}" --once >/dev/null 2>&1; then
  echo "[real_fp] ERROR: no ${COLOR_TOPIC}" >&2
  echo "  Start: bash scripts/real_robot_bringup.sh --perception" >&2
  exit 1
fi
if ! timeout 12 ros2 topic echo "${DEPTH_TOPIC}" --once >/dev/null 2>&1; then
  echo "[real_fp] ERROR: no ${DEPTH_TOPIC} (need depth_registration + both USB attach)" >&2
  exit 1
fi
echo "[real_fp] RGB-D topics OK"

echo "[real_fp] Capturing one frame to ${CAPTURE_DIR} ..."
"${PYTHON}" "${ROOT}/scripts/capture_camera_frame.py" --out-dir "${CAPTURE_DIR}" || exit 1

if ! ros2 service list 2>/dev/null | grep -q '/trigger_fine_detection'; then
  echo "[real_fp] Starting fine_detector_node (FoundationPose) ..."
  bash "${ROOT}/scripts/run_fine_detector_node.sh" >"${LOG_DIR}/fine_detector.log" 2>&1 &
  FP_PID=$!
  for i in $(seq 1 90); do
    if ros2 service list 2>/dev/null | grep -q '/trigger_fine_detection'; then
      echo "[real_fp] fine_detector ready (${i}s)"
      break
    fi
    sleep 1
    if [[ "${i}" -eq 90 ]]; then
      echo "[real_fp] ERROR: fine_detector timeout — see ${LOG_DIR}/fine_detector.log" >&2
      tail -30 "${LOG_DIR}/fine_detector.log" >&2 || true
      kill "${FP_PID}" 2>/dev/null || true
      exit 1
    fi
  done
fi

echo "[real_fp] Checking TF base_link -> camera_wrist_color_optical_frame ..."
if ! timeout 5 ros2 run tf2_ros tf2_echo base_link camera_wrist_color_optical_frame 2>/dev/null | head -5; then
  echo "[real_fp] WARN: TF chain incomplete — detection may return TF failed" >&2
  echo "  Need: arm + joint_states relay + link6->camera_wrist_link static TF" >&2
fi

echo "[real_fp] Calling trigger_fine_detection (real camera data) ..."
"${PYTHON}" "${ROOT}/scripts/verify_fine_detection.py" --skip-wrapper --ros-service --service-timeout 180
RC=$?

LATEST_RGB="$(ls -t "${CAPTURE_DIR}"/color_*.png 2>/dev/null | head -1 || true)"
if [[ -n "${LATEST_RGB}" ]]; then
  echo "[real_fp] Latest capture: ${LATEST_RGB}"
  echo "[real_fp] Open with: xdg-open ${LATEST_RGB}  (or copy to Windows)"
fi
exit "${RC}"
