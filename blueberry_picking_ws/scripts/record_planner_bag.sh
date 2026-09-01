#!/usr/bin/env bash
# Record the authoritative planner input topic (+ joints / viz) as rosbag2.
#
# Prerequisite: planner_input_assembler already publishing /planning/planner_input
#
# Usage:
#   bash scripts/record_planner_bag.sh
#   bash scripts/record_planner_bag.sh --qa-dir log/real_robot/planner_qa --extra /planning/viz/wrist
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QA_DIR="${ROOT}/log/real_robot/planner_qa"
EXTRA_TOPICS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qa-dir) QA_DIR="$2"; shift 2 ;;
    --extra) EXTRA_TOPICS+=("$2"); shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SESSION="$(date +%Y%m%d_%H%M%S)"
OUT="${QA_DIR}/${SESSION}"
mkdir -p "${QA_DIR}"

TOPICS=(
  # 统一感知（clusters + berries，带稳定 track id）
  /perception/scene_graph
  /perception/occupancy_esdf
  /perception/occupancy_local
  # 原始感知（debug 对照：感知是否出错）
  /perception/global/berries
  /perception/fine/berries
  # 规划输入（训练真源）
  /planning/planner_input
  /planning/task_context
  # 模型输出 + IK/执行（debug 对照）
  /planning/tool_trajectory_4s
  /planning/ik_joint_trajectory
  /planning/joint_cmd
  /planning/executor_status
  # 真机关节反馈
  /feedback/joint_states
  /joint_states
  # 编排状态
  /pick/status
  /reach/status
  # 可视化
  /planning/viz/fixed
  /planning/viz/wrist
  /planning/viz/hud
  /planning/viz/status
  /planning/viz/occupancy_slice
  /planning/viz/occupancy_overlay
)
TOPICS+=("${EXTRA_TOPICS[@]}")

echo "[record_planner_bag] → ${OUT}"
echo "[record_planner_bag] topics: ${TOPICS[*]}"
echo "[record_planner_bag] verify live first:  ros2 topic echo /planning/planner_input --once"
exec ros2 bag record -o "${OUT}/bag" "${TOPICS[@]}"
