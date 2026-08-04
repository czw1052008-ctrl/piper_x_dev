#!/usr/bin/env bash
# Staged smoke for detect → range → plan → FSM.
# Usage:
#   bash scripts/reach_stage_smoke.sh [0|1|2|3|4|all]
# Screenshots: log/real_robot/stage_N/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
STAGE="${1:-all}"
LOG_ROOT="${ROOT}/log/real_robot"

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
fi

set +u
# shellcheck disable=SC1090
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null 2>&1 || true
[[ -f "${ROOT}/install/setup.bash" ]] && source "${ROOT}/install/setup.bash"
set -u 2>/dev/null || true

_ok() { echo "[OK] $*"; }
_fail() { echo "[FAIL] $*" >&2; return 1; }
_info() { echo "[..] $*"; }

_save_image() {
  local topic="$1" out="$2" timeout_s="${3:-8}"
  mkdir -p "$(dirname "${out}")"
  /usr/bin/python3 - <<PY
import sys
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

rclpy.init()
node = rclpy.create_node('stage_snap')
got = {'msg': None}

def cb(msg):
    got['msg'] = msg

sub = node.create_subscription(Image, '${topic}', cb, qos_profile_sensor_data)
deadline = node.get_clock().now().nanoseconds + int(${timeout_s} * 1e9)
while got['msg'] is None and node.get_clock().now().nanoseconds < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)
msg = got['msg']
node.destroy_node()
rclpy.shutdown()
if msg is None:
    print('no image on ${topic}', file=sys.stderr)
    sys.exit(2)
# write ppm via numpy without cv2 if needed
import numpy as np
enc = msg.encoding.lower()
h, w = msg.height, msg.width
if enc == 'rgb8':
    rgb = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
elif enc == 'bgr8':
    bgr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
    rgb = bgr[:, :, ::-1]
else:
    print(f'unsupported encoding {msg.encoding}', file=sys.stderr)
    sys.exit(3)
try:
    import cv2
    cv2.imwrite('${out}', rgb[:, :, ::-1])
except Exception:
    # fallback PPM
    path = '${out}'.rsplit('.', 1)[0] + '.ppm'
    with open(path, 'wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode())
        f.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    print(path)
    sys.exit(0)
print('${out}')
PY
}

stage0() {
  local d="${LOG_ROOT}/stage_0"
  mkdir -p "${d}"
  {
    echo "=== USB ==="
    lsusb | grep -iE '2bdf|2bc5|1d50|orbbec|General|OpenMoko' || true
    echo "=== video ==="
    ls -l /dev/video* 2>&1 || true
    echo "=== can0 ==="
    ip link show can0 2>&1 || true
  } | tee "${d}/hw.txt"

  [[ -e "${FIXED_CAMERA_DEVICE:-/dev/video0}" ]] || _fail "missing FIXED_CAMERA_DEVICE=${FIXED_CAMERA_DEVICE:-}"
  _ok "fixed camera device ${FIXED_CAMERA_DEVICE}"
  ip link show can0 2>/dev/null | grep -q 'UP' || _fail "can0 not UP"
  _ok "can0 UP"
  if lsusb | grep -q '2bc5:0657'; then
    _ok "Orbbec depth USB present"
  else
    _fail "Orbbec depth (2bc5:0657) not in lsusb — plug DaBai dual USB"
  fi
}

stage1() {
  local d="${LOG_ROOT}/stage_1"
  mkdir -p "${d}"
  _info "waiting /perception/global/berries"
  timeout 10 ros2 topic echo /perception/global/berries --once > "${d}/berries.txt" \
    || _fail "no global berries"
  grep -q 'position:' "${d}/berries.txt" || _fail "empty global berries"
  _ok "global berries received"
  _save_image /perception/global/detection_viz "${d}/global_viz.png" 10 \
    || _fail "no global detection_viz"
  _ok "saved ${d}/global_viz.png"
}

stage2() {
  local d="${LOG_ROOT}/stage_2"
  mkdir -p "${d}"
  timeout 12 ros2 topic echo /perception/fine/berries --once > "${d}/berries.txt" \
    || _fail "no fine berries"
  if grep -q 'confidence:\|berries:' "${d}/berries.txt"; then
    _ok "fine berries (may be empty array if wrist FOV empty)"
  else
    _fail "fine topic malformed"
  fi
  _save_image /perception/fine/detection_viz "${d}/fine_viz.png" 12 \
    || _info "WARN: no fine viz yet (wrist may not see fruit)"
  _save_image /camera_wrist/color/image_raw "${d}/wrist_color.png" 8 \
    || _fail "no wrist color"
  _ok "wrist color ok"
}

stage3() {
  local d="${LOG_ROOT}/stage_3"
  mkdir -p "${d}"
  _info "unit: cup_axis_plan"
  PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}" /usr/bin/python3 -m pytest \
    "${ROOT}/test/test_cup_axis_plan.py" -q | tee "${d}/pytest_plan.txt"
  _ok "cup_axis_plan unit tests"
}

stage4() {
  local d="${LOG_ROOT}/stage_4"
  mkdir -p "${d}"
  _info "FSM dry-run: publish start, wait WAIT_CONFIRM"
  timeout 3 ros2 topic echo /reach/status --once > "${d}/status_before.txt" || true
  # Stream status (not --once): /reach/status is not latched.
  timeout 40 ros2 topic echo /reach/status > "${d}/status_stream.txt" &
  local echo_pid=$!
  sleep 0.5
  ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: start}" >/dev/null
  local i=0
  while (( i < 40 )); do
    if grep -q 'WAIT_CONFIRM' "${d}/status_stream.txt" 2>/dev/null; then
      kill "${echo_pid}" 2>/dev/null || true
      wait "${echo_pid}" 2>/dev/null || true
      _ok "reached WAIT_CONFIRM"
      timeout 5 ros2 topic echo /reach/plan --once > "${d}/plan.txt" || true
      ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: confirm_reset}" >/dev/null
      return 0
    fi
    if grep -q 'ERROR' "${d}/status_stream.txt" 2>/dev/null; then
      kill "${echo_pid}" 2>/dev/null || true
      _fail "FSM ERROR — see ${d}/status_stream.txt"
    fi
    sleep 1
    i=$((i + 1))
  done
  kill "${echo_pid}" 2>/dev/null || true
  _fail "timeout waiting WAIT_CONFIRM"
}

case "${STAGE}" in
  0) stage0 ;;
  1) stage1 ;;
  2) stage2 ;;
  3) stage3 ;;
  4) stage4 ;;
  all)
    stage0
    stage3
    stage1 || true
    stage2 || true
    stage4 || true
    ;;
  *)
    echo "Usage: $0 [0|1|2|3|4|all]" >&2
    exit 2
    ;;
esac
