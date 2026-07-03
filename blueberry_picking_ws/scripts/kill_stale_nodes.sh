#!/usr/bin/env bash
# Stop leftover ROS nodes from a previous launch (duplicate action/service servers break picking).
set -euo pipefail

# Kill previous launch trees first (safe: new launch has not started yet).
launch_patterns=(
  'sim_gz_vibration.launch.py'
  'sim_gz_suction.launch.py'
  'ros2 launch picking_bringup'
  'gz sim -r'
  'gz sim server'
  'gz sim gui'
  'ign gazebo'
  'ruby.*gz sim'
)

node_patterns=(
  'lib/robot_state_publisher/robot_state_publisher'
  'lib/picking_bringup/robot_description_publisher'
  'lib/picking_task/pick_action_server'
  'lib/moveit_ros_move_group/move_group'
  'move_group'
  'lib/controller_manager/spawner'
  'controller_manager/spawner'
  'lib/ros_gz_bridge/parameter_bridge'
  'ros_gz_bridge/parameter_bridge'
  'fake_perception_node'
  'static_fake_perception_node'
  'send_pick_goal'
  'grasp_planner_node'
)

killed=0
for pat in "${launch_patterns[@]}" "${node_patterns[@]}"; do
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    # Do not kill processes started in the last 20s (avoids suicide during launch).
    age="$(ps -o etimes= -p "${pid}" 2>/dev/null | tr -d ' ')"
    if [[ -n "${age}" && "${age}" -lt 20 ]]; then
      continue
    fi
    kill -9 "${pid}" 2>/dev/null || true
    killed=1
    sleep 0.2
  done < <(pgrep -f "${pat}" 2>/dev/null || true)
done

if command -v ros2 >/dev/null 2>&1; then
  ros2 daemon stop >/dev/null 2>&1 || true
  sleep 0.5
  ros2 daemon start >/dev/null 2>&1 || true
fi

# Clear stale FastDDS SHM segments (reduces discovery / port lock failures).
rm -rf /dev/shm/fastrtps_* /dev/shm/fastdds_* 2>/dev/null || true

if [[ "${killed}" -eq 1 ]]; then
  sleep 1.0
  echo "[kill_stale_nodes] Cleared stale launch / pick processes."
else
  echo "[kill_stale_nodes] No stale launch / pick processes found."
fi

remaining="$(pgrep -af 'lib/picking_task/pick_action_server' 2>/dev/null || true)"
if [[ -n "${remaining}" ]]; then
  echo "[kill_stale_nodes] WARN: pick_action_server still running:" >&2
  echo "${remaining}" >&2
  exit 1
fi

remaining_spawner="$(pgrep -af 'lib/controller_manager/spawner' 2>/dev/null || true)"
if [[ -n "${remaining_spawner}" ]]; then
  echo "[kill_stale_nodes] WARN: controller spawner still running:" >&2
  echo "${remaining_spawner}" >&2
  exit 1
fi
