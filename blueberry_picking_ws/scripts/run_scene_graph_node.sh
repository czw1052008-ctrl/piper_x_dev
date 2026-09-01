#!/usr/bin/env bash
# Fuse /perception/global|fine/berries → /perception/scene_graph
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/config/real_robot.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT}/config/real_robot.env"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi
set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
set -u 2>/dev/null || true

ARGS=(--ros-args -p publish_hz:=10.0 -p stabilize_berry_ids:=true)
ARGS+=("$@")
exec ros2 run picking_perception scene_graph_node "${ARGS[@]}"
