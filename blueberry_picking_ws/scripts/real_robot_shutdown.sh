#!/usr/bin/env bash
# Stop real-robot stack started by real_robot_bringup.sh (disable arm + kill children).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
PIDFILE="${ROOT}/log/real_robot/stack.pids"
QUIET=false
NO_DISABLE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quiet) QUIET=true; shift ;;
    --no-disable) NO_DISABLE=true; shift ;;
    -h|--help)
      echo "Usage: bash scripts/real_robot_shutdown.sh [--quiet] [--no-disable]"
      echo "  --no-disable   Kill ROS nodes only; do NOT call /enable_agx_arm false."
      echo "                 Use for restart (avoids arm dropping during bringup recycle)."
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
fi

PICKING_WS="${PICKING_WS:-${ROOT}}"
AGX_ARM_WS="${AGX_ARM_WS:-${HOME}/piper_x_dev/agx_arm_ros}"

_disable_arm() {
  if [[ "${NO_DISABLE}" == true ]]; then
    return 0
  fi
  if [[ "${QUIET}" == true && ! -f "${PIDFILE}" ]]; then
    return 0
  fi
  if [[ ! -f "${AGX_ARM_WS}/install/setup.bash" ]]; then
    return 0
  fi
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash 2>/dev/null || true
  # shellcheck disable=SC1090
  source "${AGX_ARM_WS}/install/setup.bash" 2>/dev/null || true
  set -u 2>/dev/null || true
  if command -v ros2 >/dev/null 2>&1; then
    [[ "${QUIET}" != true ]] && echo "[shutdown] Disabling arm (/enable_agx_arm false) ..."
    timeout 5 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: false}" \
      >/dev/null 2>&1 || [[ "${QUIET}" == true ]] || \
      echo "[shutdown] WARN: enable_agx_arm call failed or timed out"
  fi
}

_kill_pidfile() {
  if [[ ! -f "${PIDFILE}" ]]; then
    return 0
  fi
  while read -r name pid; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[shutdown] Stopping ${name} (pid ${pid}) ..."
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
      sleep 0.5
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  done < "${PIDFILE}"
  rm -f "${PIDFILE}"
}

_kill_patterns() {
  local patterns=(
    'start_single_agx_arm_moveit.launch.py'
    'agx_arm_ctrl_single'
    'topic_tools relay /feedback/joint_states /joint_states'
    'orbbec_camera.*launch'
    'component_container.*camera'
    'static_transform_publisher.*camera_wrist_link'
    'perception.launch.py'
    'fine_detector_node'
    'run_fine_detector_node'
    'link6_teleop_node'
    'grasp_planner.launch.py'
    'grasp_planner_node'
    'visualize_blueberry_detection.py'
    'real_suction_pick_loop.py'
  )
  local pat pid
  for pat in "${patterns[@]}"; do
    while read -r pid; do
      [[ -z "${pid}" ]] && continue
      kill -TERM "${pid}" 2>/dev/null || true
    done < <(pgrep -f "${pat}" 2>/dev/null || true)
  done
  sleep 0.5
  rm -f "${ROOT}/log/real_robot/grasp_planner.pid"
}

_disable_arm
_kill_pidfile
_kill_patterns

if command -v ros2 >/dev/null 2>&1; then
  ros2 daemon stop >/dev/null 2>&1 || true
  ros2 daemon start >/dev/null 2>&1 || true
fi

[[ "${QUIET}" != true ]] && echo "[shutdown] Real-robot stack stopped."
