#!/usr/bin/env bash
# Phase 0 hardware + env smoke (native Ubuntu). Needs ROS Humble installed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
[[ -f "${CONFIG}" ]] && source "${CONFIG}"

echo "== OS =="
source /etc/os-release
echo "${PRETTY_NAME}"

echo "== ROS =="
if [[ -f /opt/ros/humble/setup.bash ]]; then
  echo "OK: /opt/ros/humble"
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
else
  echo "MISSING: install with: bash scripts/install_deps.sh  (requires sudo password)"
fi

echo "== USB =="
lsusb | rg -i 'orbbec|1d50:606f|2bdf' || true
echo "video nodes:"
for d in /sys/class/video4linux/video*; do
  echo "  $(basename "$d") -> $(cat "$d/name" 2>/dev/null)"
done

echo "== CAN =="
if ip link show can0 &>/dev/null; then
  ip -details link show can0 | head -5
  state="$(ip -o link show can0 | awk '{print $9}')"
  if [[ "${state}" != "UP" ]]; then
    echo "can0 is DOWN — run: sudo ip link set can0 up type can bitrate ${CAN_BITRATE:-1000000}"
  else
    echo "OK: can0 UP"
  fi
else
  echo "MISSING can0"
fi

echo "== Workspaces =="
for ws in "${AGX_ARM_WS:-$ROOT/../agx_arm_ros}" "${ORBBEC_WS:-$ROOT/../OrbbecSDK_ROS2_main}" "${ROOT}"; do
  if [[ -f "${ws}/install/setup.bash" ]]; then
    echo "built: ${ws}"
  else
    echo "NOT built: ${ws}"
  fi
done

echo "== Fixed camera device =="
echo "FIXED_CAMERA_DEVICE=${FIXED_CAMERA_DEVICE:-/dev/video0}"
[[ -e "${FIXED_CAMERA_DEVICE:-/dev/video0}" ]] && echo "OK exists" || echo "MISSING"

echo "Done. Full bringup after ROS+colcon: bash scripts/run_real_reach.sh"
