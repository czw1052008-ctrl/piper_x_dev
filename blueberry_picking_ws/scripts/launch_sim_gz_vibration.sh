#!/usr/bin/env bash
# Kill stale nodes, source env, then launch Gazebo vibration sim (safe entry point).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "WARN: conda is active (${CONDA_PREFIX}). Run: conda deactivate" >&2
fi

bash scripts/kill_stale_nodes.sh
# shellcheck disable=SC1091
source scripts/setup_env.sh
colcon build --symlink-install --packages-select picking_task picking_grasp picking_moveit_config picking_bringup
# shellcheck disable=SC1091
source install/setup.bash
echo "Starting ros2 launch (Ctrl+C to stop) ..."
exec ros2 launch picking_bringup sim_gz_vibration.launch.py "$@"
