#!/usr/bin/env bash
# DEPRECATED for real-robot main path.
# Prefer the topic-driven reach pipeline:
#   bash scripts/run_real_reach.sh
#   see docs/REACH_PIPELINE.md
#
# Legacy: suction pick loop (service-based fine detection).
# Logs: log/real_robot/*.log
set -eo pipefail
echo "[DEPRECATED] Prefer: bash scripts/run_real_reach.sh  (docs/REACH_PIPELINE.md)" >&2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log/real_robot"
CONFIG="${ROOT}/config/real_robot.env"
mkdir -p "${LOG_DIR}"

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
fi
PICK_EE_LINK="${PICK_EE_LINK:-tcp_link}"
PICK_CUP_CONTACT_OFFSET="${PICK_CUP_CONTACT_OFFSET:-0.04}"
PICK_CAMERA_MOUNT_Y="${PICK_CAMERA_MOUNT_Y:--0.08}"
PICK_TCP_MOUNT_Z="${PICK_TCP_MOUNT_Z:-0.05}"
PICK_FLANGE_FRAME="${PICK_FLANGE_FRAME:-link6}"

DO_BRINGUP=true
DO_VIZ=true
SHUTDOWN_ON_EXIT=false
SHUTDOWN_ON_INT=true
PROBE_ONLY=false
BRINGUP_ARGS=(--perception --no-wait)
PICK_EXTRA=()
COMMON_PICK_FLAGS=(
  --ee-link "${PICK_EE_LINK}"
  --cup-contact-offset "${PICK_CUP_CONTACT_OFFSET}"
  --camera-mount-y "${PICK_CAMERA_MOUNT_Y}"
  --tcp-mount-z "${PICK_TCP_MOUNT_Z}"
  --flange-frame "${PICK_FLANGE_FRAME}"
)

usage() {
  sed -n '2,14p' "$0"
  echo ""
  echo "Options:"
  echo "  --no-bringup         Skip arm/camera/perception restart (planner+viz+pick only)"
  echo "  --no-viz             Skip detection overlay window"
  echo "  --shutdown-on-exit   Stop arm/camera after pick loop exits"
  echo "  --keep-stack         Ctrl+C: leave arm/camera running (default: full cleanup)"
  echo "  --attach-usb         usbipd attach CAN (Windows)"
  echo "  --probe-only         观测位探针：锁定目标后打印距离/路径，不抓取"
  echo "  -h, --help           Show help"
  echo ""
  echo "Extra args -> real_suction_pick_loop.py"
  echo "  default: --move --once --auto-retreat"
  echo "  --probe-only: --probe-only --once (no grasp)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bringup) DO_BRINGUP=false; shift ;;
    --no-viz) DO_VIZ=false; shift ;;
    --shutdown-on-exit) SHUTDOWN_ON_EXIT=true; shift ;;
    --keep-stack) SHUTDOWN_ON_INT=false; shift ;;
    --attach-usb) BRINGUP_ARGS+=(--attach-usb); shift ;;
    --probe-only) PROBE_ONLY=true; PICK_EXTRA+=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) PICK_EXTRA+=("$1"); shift ;;
  esac
done

if [[ ${#PICK_EXTRA[@]} -eq 0 ]]; then
  if [[ "${PROBE_ONLY}" == true ]]; then
    PICK_ARGS=(--probe-only --once "${COMMON_PICK_FLAGS[@]}")
  else
    PICK_ARGS=(--move --once --auto-retreat "${COMMON_PICK_FLAGS[@]}")
  fi
else
  PICK_ARGS=("${PICK_EXTRA[@]}" "${COMMON_PICK_FLAGS[@]}")
fi

conda deactivate 2>/dev/null || true
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"

_log() { echo "[pick-all] $*"; }
_die() { echo "[pick-all] ERROR: $*" >&2; exit 1; }

_stop_aux() {
  pkill -f 'grasp_planner_node' 2>/dev/null || true
  pkill -f 'visualize_blueberry_detection.py' 2>/dev/null || true
  rm -f "${LOG_DIR}/grasp_planner.pid"
}

_cleanup_all() {
  _log "Cleaning stale processes (arm, camera, perception, planner, viz) ..."
  bash "${ROOT}/scripts/real_robot_shutdown.sh" --quiet --no-disable 2>/dev/null || true
  _stop_aux
  sleep 1
}

_shutdown_stack() {
  _cleanup_all
  [[ "${SHUTDOWN_ON_INT}" == true || "${SHUTDOWN_ON_EXIT}" == true ]] \
    && _log "Stack stopped."
}

_on_interrupt() {
  if [[ "${SHUTDOWN_ON_INT}" == true ]]; then
    _shutdown_stack
  else
    _log "Interrupted — arm/camera left running"
    _stop_aux
  fi
  exit 130
}

trap _on_interrupt INT TERM

_stack_check() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    return 0
  fi
  echo "${label}"
  return 1
}

_stack_ready() {
  local missing=""
  ros2 action list 2>/dev/null | grep -q '/move_action' \
    || missing="${missing} move_action"
  ros2 service list 2>/dev/null | grep -q '/trigger_fine_detection' \
    || missing="${missing} trigger_fine_detection"
  _stack_check joint_states timeout 4 ros2 topic echo /joint_states --once \
    || missing="${missing} /joint_states(no data)"
  _stack_check feedback timeout 4 ros2 topic echo /feedback/joint_states --once \
    || missing="${missing} /feedback/joint_states(no data)"
  _stack_check camera timeout 4 ros2 topic echo /camera_wrist/color/image_raw --once \
    || missing="${missing} /camera_wrist/color/image_raw"
  if [[ -n "${missing}" ]]; then
    _STACK_MISSING="${missing# }"
    return 1
  fi
  _STACK_MISSING=""
  return 0
}

_wait_ready() {
  local i=0
  local timeout_sec=45
  _log "Verifying stack ..."
  while (( i < timeout_sec )); do
    if _stack_ready; then
      _log "Stack OK."
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  _die "Stack not ready. See ${LOG_DIR}/arm.log"
}

if [[ "${DO_BRINGUP}" == true ]]; then
  _cleanup_all
  _log "Starting fresh arm + camera + perception ..."
  bash "${ROOT}/scripts/real_robot_bringup.sh" "${BRINGUP_ARGS[@]}"
  _wait_ready
else
  _stop_aux
  if ! _stack_ready; then
    _die "Stack not up (missing:${_STACK_MISSING:- unknown}).
  Arm feedback dead usually means CAN disconnected — check: candump can0
  Full restart: bash scripts/run_real_suction_pick.sh"
  fi
fi

if ! timeout 8 ros2 run tf2_ros tf2_echo link6 camera_wrist_color_optical_frame 2>/dev/null | head -3 | grep -q Translation; then
  _log "WARN: TF link6 -> camera_wrist_color_optical_frame not ready — check camera_tf.log"
fi

bash "${ROOT}/scripts/run_grasp_planner.sh"

if [[ "${DO_VIZ}" == true ]]; then
  _log "Starting detection viz (background) ..."
  nohup bash "${ROOT}/scripts/run_detection_viz.sh" --show \
    >"${LOG_DIR}/detection_viz.log" 2>&1 &
  sleep 2
fi

_log "Pick loop: ${PICK_ARGS[*]}"
_log "Keys: R/H stop+home | G continue | Q quit | focus this terminal"
echo ""

/usr/bin/python3 "${ROOT}/scripts/real_suction_pick_loop.py" "${PICK_ARGS[@]}"
rc=$?

if [[ "${SHUTDOWN_ON_EXIT}" == true ]]; then
  _shutdown_stack
fi
exit "${rc}"
