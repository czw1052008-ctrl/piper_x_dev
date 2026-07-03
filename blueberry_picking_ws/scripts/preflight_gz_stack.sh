#!/usr/bin/env bash
# Verify Gazebo vibration stack is ready before sending a pick goal.
set -euo pipefail

TIMEOUT_SEC="${1:-120}"
DEADLINE=$((SECONDS + TIMEOUT_SEC))
ROS_ARGS=(--ros-args -p use_sim_time:=true)

export RMW_FASTRTPS_USE_SHM="${RMW_FASTRTPS_USE_SHM:-0}"

log() { echo "[preflight_gz] $*"; }
fail() { log "ERROR: $*" >&2; exit 1; }

wait_for() {
  local desc="$1"
  shift
  log "waiting for ${desc} (timeout=${TIMEOUT_SEC}s) ..."
  while (( SECONDS < DEADLINE )); do
    if "$@" >/dev/null 2>&1; then
      log "OK: ${desc}"
      return 0
    fi
    sleep 2
  done
  fail "timeout waiting for ${desc}"
}

has_node() {
  ros2 node list 2>/dev/null | grep -qF "$1"
}

has_service() {
  ros2 service list 2>/dev/null | grep -qF "$1"
}

controller_active() {
  local ctrl="$1"
  ros2 control list_controllers -c /controller_manager "${ROS_ARGS[@]}" 2>/dev/null \
    | grep -qE "^${ctrl}[[:space:]]+.*[[:space:]]+active"
}

has_clock() {
  timeout 5 ros2 topic echo /clock --once 2>/dev/null | grep -qE 'sec:|nanosec:'
}

wait_for "/clock topic" has_clock
wait_for "grasp_planner_node in graph" has_node "/grasp_planner_node"
wait_for "plan_vibration service" has_service "/plan_vibration"
wait_for "pick_action_server in graph" has_node "/pick_action_server"
wait_for "arm_controller active" controller_active arm_controller

if [[ "${2:-}" == vibration ]]; then
  wait_for "vibration_motor_controller active" controller_active vibration_motor_controller
fi

log "stack ready"
