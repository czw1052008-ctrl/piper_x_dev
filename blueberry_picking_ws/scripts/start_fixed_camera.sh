#!/usr/bin/env bash
# Hot-add fixed DaBai + TF without full stack restart.
# Usage: bash scripts/start_fixed_camera.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "${ROOT}/config/real_robot.env"
LOG_DIR="${ROOT}/log/real_robot"
mkdir -p "${LOG_DIR}"

set +u
source /opt/ros/humble/setup.bash
source "${AGX_ARM_WS:-${ROOT}/../agx_arm_ros}/install/setup.bash"
source "${ORBBEC_WS_FIXED}/install/setup.bash"
set -u

_fixed_orb_install="${ORBBEC_WS_FIXED}/install/orbbec_camera/lib"
_fixed_orb_sdk="${ORBBEC_WS_FIXED}/orbbec_camera/SDK/lib/x64"
_fixed_ld=""
[[ -d "${_fixed_orb_install}" ]] && _fixed_ld="${_fixed_orb_install}:${_fixed_ld}"
[[ -d "${_fixed_orb_sdk}" ]] && _fixed_ld="${_fixed_orb_sdk}:${_fixed_orb_sdk}/extensions/depthengine:${_fixed_ld}"
export LD_LIBRARY_PATH="${_fixed_ld}${LD_LIBRARY_PATH:-}"
export RMW_FASTRTPS_USE_SHM=0

pkill -f 'orbbec_camera dabai.launch.py' 2>/dev/null || true
pkill -f "child-frame-id ${FIXED_CAMERA_FRAME}" 2>/dev/null || true
sleep 1

fixed_cam_args="camera_name:=${FIXED_CAMERA_NAME} depth_registration:=true publish_tf:=false"
[[ -n "${FIXED_CAM_USB_PORT:-}" ]] && fixed_cam_args+=" usb_port:=${FIXED_CAM_USB_PORT}"
[[ -n "${FIXED_CAM_SERIAL:-}" ]] && fixed_cam_args+=" serial_number:=${FIXED_CAM_SERIAL}"

echo "[fixed_cam] launching dabai ${fixed_cam_args}"
# shellcheck disable=SC2086
nohup ros2 launch orbbec_camera dabai.launch.py ${fixed_cam_args} \
  > "${LOG_DIR}/fixed_cam.log" 2>&1 &
echo $! > "${LOG_DIR}/fixed_cam.pid"

echo "[fixed_cam] waiting for color ..."
ok=0
for i in $(seq 1 60); do
  if timeout 2 ros2 topic echo "/${FIXED_CAMERA_NAME}/color/image_raw" --once >/dev/null 2>&1; then
    echo "[fixed_cam] color OK (${i}s)"
    ok=1
    break
  fi
  sleep 1
done
if [[ "${ok}" != "1" ]]; then
  echo "[fixed_cam] ERROR: color timeout — see ${LOG_DIR}/fixed_cam.log" >&2
  exit 1
fi

echo "[fixed_cam] publishing TF base_link → ${FIXED_CAMERA_FRAME}"
nohup ros2 run tf2_ros static_transform_publisher \
  --x "${FIXED_CAM_TX}" --y "${FIXED_CAM_TY}" --z "${FIXED_CAM_TZ}" \
  --qx "${FIXED_CAM_QX}" --qy "${FIXED_CAM_QY}" --qz "${FIXED_CAM_QZ}" --qw "${FIXED_CAM_QW}" \
  --frame-id base_link --child-frame-id "${FIXED_CAMERA_FRAME}" \
  > "${LOG_DIR}/fixed_cam_tf.log" 2>&1 &
echo $! > "${LOG_DIR}/fixed_cam_tf.pid"

echo "[fixed_cam] ready"
