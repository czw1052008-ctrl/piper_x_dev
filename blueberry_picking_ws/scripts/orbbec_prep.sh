#!/usr/bin/env bash
# Prep Orbbec Dabai before bringup (LD path + optional UVC unbind).
# Dabai color is often claimed by kernel uvcvideo; Orbbec libuvc then fails
# with device_online=false. Prefer uvc_backend:=v4l2 (set in dabai.launch.py).
set -euo pipefail
PIPER_X_DEV="${PIPER_X_DEV:-/home/user/codes/piper_x_dev}"
ORBBEC_WS="${ORBBEC_WS:-${PIPER_X_DEV}/OrbbecSDK_ROS2_main}"
SDK_LIB="${ORBBEC_WS}/orbbec_camera/SDK/lib/x64"
export LD_LIBRARY_PATH="${SDK_LIB}:${SDK_LIB}/extensions/depthengine:${LD_LIBRARY_PATH:-}"

if [[ "${1:-}" == "--unbind-uvc" ]]; then
  # Only if using libuvc backend
  for iface in 1-7.2:1.0 1-7.2:1.1; do
    if [[ -e "/sys/bus/usb/drivers/uvcvideo/${iface}" ]]; then
      echo "${iface}" | sudo tee /sys/bus/usb/drivers/uvcvideo/unbind >/dev/null
      echo "unbound ${iface}"
    fi
  done
fi

echo "LD_LIBRARY_PATH includes Orbbec SDK: ${SDK_LIB}"
lsusb -d 2bc5: || echo "WARN: no Orbbec USB device"
