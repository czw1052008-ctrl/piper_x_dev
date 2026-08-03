#!/usr/bin/env bash
# Start fine_detector_node (YOLO+depth topic stream, no FoundationPose by default).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/config/real_robot.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT}/config/real_robot.env"
fi

# Leave conda — system/ROS Python for Humble
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi

set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
set -u 2>/dev/null || true

FP_FLAG="${ENABLE_FOUNDATION_POSE:-false}"
ARGS=(--ros-args
  -p enable_foundation_pose:="${FP_FLAG}"
  -p publish_hz:=3.0
  -p mask_source:=yolo
)

YOLO_CFG="${ROOT}/src/picking_perception/config/foundation_pose.yaml"
if [[ -f "${YOLO_CFG}" ]]; then
  ARGS+=(--params-file "${YOLO_CFG}")
fi
ARGS+=("$@")

exec ros2 run picking_perception fine_detector_node "${ARGS[@]}"
