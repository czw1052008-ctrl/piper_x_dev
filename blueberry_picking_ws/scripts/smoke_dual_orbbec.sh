#!/usr/bin/env bash
# Dual-camera topic smoke: Gemini 305 wrist + DaBai fixed (no arm).
# Usage: bash scripts/smoke_dual_orbbec.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_X_DEV="$(cd "${ROOT}/.." && pwd)"
# shellcheck disable=SC1090
[[ -f "${ROOT}/config/real_robot.env" ]] && source "${ROOT}/config/real_robot.env"
ORBBEC_WS="${ORBBEC_WS:-${PIPER_X_DEV}/OrbbecSDK_ROS2}"
ORBBEC_WS_FIXED="${ORBBEC_WS_FIXED:-${PIPER_X_DEV}/OrbbecSDK_ROS2_main}"
LOG="${ROOT}/log/real_robot"
mkdir -p "${LOG}"

pkill -f 'gemini305.launch|dabai.launch.py|camera_wrist/camera|camera_fixed/camera' 2>/dev/null || true
sleep 1

set +u
source /opt/ros/humble/setup.bash
set -u

_WRIST_LAUNCH="${ORBBEC_WS}/install/orbbec_camera/share/orbbec_camera/launch/gemini305.launch.py"
_FIXED_LAUNCH="${ORBBEC_WS_FIXED}/install/orbbec_camera/share/orbbec_camera/launch/dabai.launch.py"
[[ -f "${_WRIST_LAUNCH}" ]] || { echo "missing ${_WRIST_LAUNCH}"; exit 1; }
[[ -f "${_FIXED_LAUNCH}" ]] || { echo "missing ${_FIXED_LAUNCH}"; exit 1; }

_UVC="${ORBBEC_UVC_BACKEND:-v4l2}"
_WRIST_LD="${ORBBEC_WS}/install/orbbec_camera/lib:${ORBBEC_WS}/install/orbbec_camera/lib/extensions"
_FIXED_LD="${ORBBEC_WS_FIXED}/install/orbbec_camera/lib:${ORBBEC_WS_FIXED}/orbbec_camera/SDK/lib/x64"

echo "[smoke] starting fixed DaBai (${FIXED_CAM_SERIAL:-auto}) first …"
(
  set +u
  source /opt/ros/humble/setup.bash
  source "${ORBBEC_WS_FIXED}/install/setup.bash"
  set -u
  export RMW_FASTRTPS_USE_SHM=0
  export LD_LIBRARY_PATH="${_FIXED_LD}:${LD_LIBRARY_PATH:-}"
  exec ros2 launch "${_FIXED_LAUNCH}" \
    camera_name:=camera_fixed depth_registration:=true publish_tf:=false \
    ${FIXED_CAM_SERIAL:+serial_number:=${FIXED_CAM_SERIAL}} \
    ${FIXED_CAM_USB_PORT:+usb_port:=${FIXED_CAM_USB_PORT}}
) >"${LOG}/smoke_fixed.log" 2>&1 &
FIXED_PID=$!
sleep 8

echo "[smoke] starting wrist Gemini (${ORBBEC_SERIAL:-auto}, port=${ORBBEC_USB_PORT:-2-6}, uvc=${_UVC}) …"
(
  set +u
  source /opt/ros/humble/setup.bash
  source "${ORBBEC_WS}/install/setup.bash"
  set -u
  _filtered_ament=""
  IFS=':' read -ra _ap <<< "${AMENT_PREFIX_PATH:-}"
  for _p in "${_ap[@]}"; do
    [[ -z "${_p}" ]] && continue
    [[ "${_p}" == *OrbbecSDK_ROS2_main/install/orbbec_camera* ]] && continue
    _filtered_ament+="${_p}:"
  done
  export AMENT_PREFIX_PATH="${ORBBEC_WS}/install:${_filtered_ament%:}"
  export RMW_FASTRTPS_USE_SHM=0
  export LD_LIBRARY_PATH="${_WRIST_LD}:${LD_LIBRARY_PATH:-}"
  exec ros2 launch "${_WRIST_LAUNCH}" \
    camera_name:=camera_wrist depth_registration:=true publish_tf:=false \
    uvc_backend:="${_UVC}" \
    ${ORBBEC_SERIAL:+serial_number:=${ORBBEC_SERIAL}} \
    ${ORBBEC_USB_PORT:+usb_port:=${ORBBEC_USB_PORT}}
) >"${LOG}/smoke_wrist.log" 2>&1 &
WRIST_PID=$!

echo "[smoke] PIDs wrist=${WRIST_PID} fixed=${FIXED_PID}; waiting 15s …"
sleep 15

set +u
source /opt/ros/humble/setup.bash
set -u
ok=0
for t in \
  /camera_wrist/color/image_raw \
  /camera_wrist/depth/image_raw \
  /camera_fixed/color/image_raw \
  /camera_fixed/depth/image_raw
do
  if timeout 8 ros2 topic echo "${t}" --once >/tmp/echo_$(basename ${t}).txt 2>&1; then
    echo "[smoke] OK ${t}"
    ok=$((ok + 1))
  else
    echo "[smoke] FAIL ${t} (see /tmp/echo_$(basename ${t}).txt)"
  fi
done

echo "[smoke] ${ok}/4 topics publishing"
echo "[smoke] logs: ${LOG}/smoke_wrist.log ${LOG}/smoke_fixed.log"
echo "[smoke] note: Gemini on USB2.x often fails depth; use USB3 + optional sudo uvc unbind for libuvc"
exit $(( ok < 4 ? 1 : 0 ))
