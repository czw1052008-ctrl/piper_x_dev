#!/usr/bin/env bash
# Clean start for Gazebo vibration sim with stable DDS settings.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WS}"

# DDS: disable SHM on WSL2; use default SUBNET discovery (LOCALHOST breaks WSL2).
unset ROS_LOCALHOST_ONLY
unset ROS_AUTOMATIC_DISCOVERY_RANGE
export RMW_FASTRTPS_USE_SHM=0
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset GZ_SIM_RESOURCE_PATH
unset GZ_SIM_SYSTEM_PLUGIN_PATH

AUTO_PICK="${AUTO_PICK:-false}"
FINE_PERCEPTION="${FINE_PERCEPTION:-gz}"

bash scripts/kill_stale_nodes.sh

set +u
source install/setup.bash
set -u

echo "[run_gz_vibration] launching fine_perception:=${FINE_PERCEPTION} auto_pick:=${AUTO_PICK}"
exec ros2 launch picking_bringup sim_gz_vibration.launch.py \
  fine_perception:="${FINE_PERCEPTION}" \
  auto_pick:="${AUTO_PICK}"
