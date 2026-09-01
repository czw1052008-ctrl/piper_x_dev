#!/usr/bin/env bash
# Open/close Piper X AgileX gripper via /control/joint_states and print feedback.
# Requires bringup with ARM_EFFECTOR=agx_gripper (see config/real_robot.env).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
WIDTH_OPEN="${GRIPPER_OPEN_WIDTH:-0.08}"
WIDTH_HALF="${GRIPPER_HALF_WIDTH:-0.04}"
FORCE="${GRIPPER_FORCE:-1.0}"
WAIT_S="${GRIPPER_SETTLE_S:-2.5}"

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
fi

set +u
source /opt/ros/humble/setup.bash 2>/dev/null || source /opt/ros/jazzy/setup.bash
source "${AGX_ARM_WS:-${ROOT}/../agx_arm_ros}/install/setup.bash"
set -u 2>/dev/null || true

_pub_gripper() {
  local width="$1"
  local force="$2"
  ros2 topic pub --once /control/joint_states sensor_msgs/msg/JointState \
    "{name: ['gripper'], position: [${width}], velocity: [], effort: [${force}]}" >/dev/null
}

_read_status() {
  timeout 3 ros2 topic echo /feedback/gripper_status --once 2>/dev/null || true
}

echo "[gripper_smoke] Checking /feedback/gripper_status ..."
if ! timeout 3 ros2 topic info /feedback/gripper_status 2>/dev/null | rg -q 'Publisher count: [1-9]'; then
  cat >&2 <<EOF
[gripper_smoke] ERROR: no gripper feedback publisher.

Bring up with agx_gripper, e.g.:
  1. Set ARM_EFFECTOR=agx_gripper in config/real_robot.env
  2. bash scripts/real_robot_shutdown.sh --no-disable
  3. bash scripts/real_robot_bringup.sh

Wiring (PIPER-X manual §2.4.2): XT30 J6 <-> gripper; same CAN bus as arm.
EOF
  exit 1
fi

echo "[gripper_smoke] Initial status:"
_read_status

echo "[gripper_smoke] OPEN width=${WIDTH_OPEN} m force=${FORCE} N ..."
_pub_gripper "${WIDTH_OPEN}" "${FORCE}"
sleep "${WAIT_S}"
echo "[gripper_smoke] After OPEN:"
_read_status

echo "[gripper_smoke] HALF width=${WIDTH_HALF} m ..."
_pub_gripper "${WIDTH_HALF}" "${FORCE}"
sleep "${WAIT_S}"
echo "[gripper_smoke] After HALF:"
_read_status

echo "[gripper_smoke] CLOSE width=0 ..."
_pub_gripper 0.0 "${FORCE}"
sleep "${WAIT_S}"
echo "[gripper_smoke] After CLOSE:"
_read_status

echo "[gripper_smoke] Done. Gripper should have visibly opened then closed."
