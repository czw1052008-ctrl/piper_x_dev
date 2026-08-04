#!/usr/bin/env bash
# Open teleop camera windows: fixed HSV overlay, wrist YOLO overlay, wrist depth.
# Prereq: arm stack + fixed cam + Orbbec + global/fine detectors running.
# Docs: docs/REACH_PIPELINE.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log/real_robot"
mkdir -p "${LOG_DIR}"

SHOW_RAW=false
NO_RQT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw) SHOW_RAW=true; shift ;;
    --no-rqt) NO_RQT=true; shift ;;
    -h|--help)
      cat <<EOF
Usage: bash scripts/run_teleop_viz.sh [--raw] [--no-rqt]

Windows (rqt_image_view):
  /perception/global/detection_viz   fixed mono + HSV
  /perception/fine/detection_viz     wrist RGB + YOLO boxes
  /perception/wrist/depth_viz        wrist depth (jet)

Optional --raw also opens raw color streams.
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
export PYTHONNOUSERSITE=1
export RMW_FASTRTPS_USE_SHM=0
set -u 2>/dev/null || true

_wait_topic() {
  local topic="$1"
  local timeout="${2:-30}"
  local t=0
  while (( t < timeout )); do
    if ros2 topic list 2>/dev/null | grep -qx "${topic}"; then
      # Prefer list presence; echo can be flaky under FastDDS.
      if timeout 1.5 ros2 topic hz "${topic}" 2>&1 | head -3 | grep -q 'average rate\|does not appear'; then
        return 0
      fi
      return 0
    fi
    sleep 1
    t=$((t + 1))
  done
  echo "[teleop-viz] WARN: topic not listed yet: ${topic}" >&2
  return 1
}

echo "[teleop-viz] Starting wrist depth colorizer ..."
if ! pgrep -f 'wrist_depth_viz_node.py' >/dev/null 2>&1; then
  nohup python3 "${ROOT}/scripts/wrist_depth_viz_node.py" \
    >"${LOG_DIR}/wrist_depth_viz.log" 2>&1 &
  echo $! >> "${LOG_DIR}/stack.pids"
fi

echo "[teleop-viz] Waiting for detection overlays (restart detectors if missing) ..."
_wait_topic /perception/global/detection_viz 60 || true
_wait_topic /perception/fine/detection_viz 60 || true
_wait_topic /perception/wrist/depth_viz 30 || true

if [[ "${NO_RQT}" == "true" ]]; then
  echo "[teleop-viz] --no-rqt: topics only. View with:"
  echo "  ros2 run rqt_image_view rqt_image_view /perception/fine/detection_viz"
  exit 0
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[teleop-viz] ERROR: DISPLAY unset — cannot open rqt windows." >&2
  exit 1
fi

_open_view() {
  local topic="$1"
  local title="$2"
  echo "[teleop-viz] window: ${title}  (${topic})"
  nohup ros2 run rqt_image_view rqt_image_view "${topic}" \
    >"${LOG_DIR}/rqt_${title}.log" 2>&1 &
  echo $! >> "${LOG_DIR}/stack.pids"
  sleep 0.4
}

_open_view /perception/global/detection_viz fixed_hsv
_open_view /perception/fine/detection_viz wrist_yolo
_open_view /perception/wrist/depth_viz wrist_depth

if [[ "${SHOW_RAW}" == "true" ]]; then
  _open_view /camera_fixed/image_raw fixed_raw
  _open_view /camera_wrist/color/image_raw wrist_raw
fi

cat <<EOF

[teleop-viz] Opened rqt windows. In another terminal run teleop:
  bash scripts/real_robot_teleop.sh

Stop views: bash scripts/real_robot_shutdown.sh
  (or pkill -f rqt_image_view ; pkill -f wrist_depth_viz_node.py)
EOF
