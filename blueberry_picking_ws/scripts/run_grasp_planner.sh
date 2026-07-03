#!/usr/bin/env bash
# Start grasp_planner_node for real-robot suction (nearest-berry selection).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log/real_robot"
PIDFILE="${LOG_DIR}/grasp_planner.pid"

mkdir -p "${LOG_DIR}"

if ros2 node list 2>/dev/null | grep -q '/grasp_planner_node'; then
  echo "[grasp] grasp_planner_node already running."
  exit 0
fi

conda deactivate 2>/dev/null || true
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"

echo "[grasp] Starting grasp_planner_node (suction, nearest berry) ..."
nohup ros2 launch picking_grasp grasp_planner.launch.py \
  end_effector_mode:=suction stream_topics:=false \
  >"${LOG_DIR}/grasp_planner.log" 2>&1 &
echo $! >"${PIDFILE}"

for _ in $(seq 1 20); do
  if ros2 service list 2>/dev/null | grep -q '/plan_suction' \
      && ros2 node list 2>/dev/null | grep -q '/grasp_planner_node'; then
    echo "[grasp] Ready: /plan_suction"
    exit 0
  fi
  sleep 1
done

echo "[grasp] WARN: plan_suction not ready — see ${LOG_DIR}/grasp_planner.log" >&2
exit 1
