#!/usr/bin/env bash
# Launch Gemini with lower res (USB2-friendly) for diagnosis.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_X_DEV="$(cd "${ROOT}/.." && pwd)"
# shellcheck disable=SC1090
[[ -f "${ROOT}/config/real_robot.env" ]] && source "${ROOT}/config/real_robot.env"
ORB="${ORBBEC_WS:-${PIPER_X_DEV}/OrbbecSDK_ROS2}"
PORT="${ORBBEC_USB_PORT:-2-6}"
SERIAL="${ORBBEC_SERIAL:-CV2L761000W1}"
set +u
source /opt/ros/humble/setup.bash
source "${ORB}/install/setup.bash"
set -u
# Wrist Gemini must use OrbbecSDK_ROS2 — strip main overlay or wrong lib loads.
_filtered_ament=""
IFS=':' read -ra _ap <<< "${AMENT_PREFIX_PATH:-}"
for _p in "${_ap[@]}"; do
  [[ -z "${_p}" ]] && continue
  [[ "${_p}" == *OrbbecSDK_ROS2_main/install/orbbec_camera* ]] && continue
  _filtered_ament+="${_p}:"
done
export AMENT_PREFIX_PATH="${ORB}/install:${_filtered_ament%:}"
export RMW_FASTRTPS_USE_SHM=0
export LD_LIBRARY_PATH="/opt/ros/humble/lib:\
${ORB}/install/orbbec_camera_msgs/lib:\
${ORB}/install/orbbec_camera/lib:\
${ORB}/install/orbbec_camera/lib/extensions:\
${ORB}/install/orbbec_camera/lib/extensions/depthengine:\
${LD_LIBRARY_PATH:-}"
exec ros2 launch "${ORB}/install/orbbec_camera/share/orbbec_camera/launch/gemini305.launch.py" \
  camera_name:=camera_wrist depth_registration:=true publish_tf:=false \
  uvc_backend:=v4l2 serial_number:="${SERIAL}" usb_port:="${PORT}" \
  depth_width:=640 depth_height:=480 depth_fps:=15 \
  color_width:=640 color_height:=480 color_fps:=15 \
  enable_point_cloud:=false enable_left_ir:=false enable_right_ir:=false
