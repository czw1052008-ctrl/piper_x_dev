#!/usr/bin/env bash
# Sequential gz_ros2_control spawner: avoids controller_manager lock contention.
set -euo pipefail

MODE="${1:-vibration}"
CM="/controller_manager"
unset ROS_LOCALHOST_ONLY
unset ROS_AUTOMATIC_DISCOVERY_RANGE
export RMW_FASTRTPS_USE_SHM="${RMW_FASTRTPS_USE_SHM:-0}"
unset FASTRTPS_DEFAULT_PROFILES_FILE
ROS_ARGS=(--ros-args -p use_sim_time:=true)
COMMON=(
  --controller-manager-timeout 120
  --switch-timeout 60
  --service-call-timeout 60
)

controller_manager_responsive() {
  ros2 control list_controllers -c "${CM}" "${ROS_ARGS[@]}" >/dev/null 2>&1
}

wait_for_clock() {
  local deadline=$((SECONDS + 60))
  echo "[spawn_gz_controllers] waiting for /clock (max 60s) ..."
  while (( SECONDS < deadline )); do
    if timeout 5 ros2 topic echo /clock --once 2>/dev/null \
      | grep -qE 'sec:|nanosec:'; then
      echo "[spawn_gz_controllers] /clock ready (${SECONDS}s)"
      return 0
    fi
    sleep 1
  done
  echo "[spawn_gz_controllers] WARN: /clock not seen, continuing anyway" >&2
  return 0
}

wait_for_controller_manager() {
  local deadline=$((SECONDS + 120))
  echo "[spawn_gz_controllers] waiting for ${CM} (max 120s) ..."
  while (( SECONDS < deadline )); do
    if controller_manager_responsive; then
      echo "[spawn_gz_controllers] controller_manager ready (${SECONDS}s)"
      return 0
    fi
    sleep 2
  done
  echo "[spawn_gz_controllers] ERROR: controller_manager not ready in 120s" >&2
  return 1
}

controller_active() {
  local ctrl="$1"
  ros2 control list_controllers -c "${CM}" "${ROS_ARGS[@]}" 2>/dev/null \
    | grep -qE "^${ctrl}[[:space:]]+.*[[:space:]]+active"
}

spawn_controller() {
  local ctrl="$1"
  shift
  if controller_active "${ctrl}"; then
    echo "[spawn_gz_controllers] ${ctrl} already active, skip"
    return 0
  fi
  echo "[spawn_gz_controllers] spawning ${ctrl} ..."
  if ros2 run controller_manager spawner "${ctrl}" "$@" \
    -c "${CM}" "${COMMON[@]}" "${ROS_ARGS[@]}"; then
    return 0
  fi
  if controller_active "${ctrl}"; then
    echo "[spawn_gz_controllers] ${ctrl} active after spawner error (race), continue"
    return 0
  fi
  echo "[spawn_gz_controllers] ERROR: failed to activate ${ctrl}" >&2
  return 1
}

echo "[spawn_gz_controllers] mode=${MODE}"
wait_for_clock
wait_for_controller_manager
spawn_controller joint_state_broadcaster

ARM=(arm_controller)
if [[ "${MODE}" == vibration ]]; then
  ARM+=(vibration_motor_controller)
fi

TO_SPAWN=()
for ctrl in "${ARM[@]}"; do
  if controller_active "${ctrl}"; then
    echo "[spawn_gz_controllers] ${ctrl} already active, skip"
  else
    TO_SPAWN+=("${ctrl}")
  fi
done

if [[ ${#TO_SPAWN[@]} -gt 0 ]]; then
  echo "[spawn_gz_controllers] spawning group: ${TO_SPAWN[*]}"
  if ! ros2 run controller_manager spawner "${TO_SPAWN[@]}" \
    -c "${CM}" --activate-as-group "${COMMON[@]}" "${ROS_ARGS[@]}"; then
    missing=0
    for ctrl in "${TO_SPAWN[@]}"; do
      if ! controller_active "${ctrl}"; then
        echo "[spawn_gz_controllers] ERROR: ${ctrl} not active" >&2
        missing=1
      fi
    done
    [[ "${missing}" -eq 0 ]] || exit 1
    echo "[spawn_gz_controllers] group spawner error but all targets active (race), continue"
  fi
fi

for ctrl in "${ARM[@]}"; do
  if ! controller_active "${ctrl}"; then
    echo "[spawn_gz_controllers] ERROR: ${ctrl} not active at end" >&2
    exit 1
  fi
done

echo "[spawn_gz_controllers] all controllers active"
