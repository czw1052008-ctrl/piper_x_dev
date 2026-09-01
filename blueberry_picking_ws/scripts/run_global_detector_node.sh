#!/usr/bin/env bash
# Start global_detector_node (fixed DaBai RGB-D YOLO → base_link).
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

# shellcheck disable=SC1090
source "${ROOT}/scripts/_yolo_pythonpath.sh"

# DaBai global: color + registered depth (no mono diameter pose).
ARGS=(--ros-args
  -p publish_hz:=10.0
  -p image_topic:=/camera_fixed/color/image_raw
  -p depth_topic:=/camera_fixed/depth/image_raw
  -p info_topic:=/camera_fixed/color/camera_info
  -p camera_frame:=camera_fixed_color_optical_frame
  -p depth_pose_source:=depth
  -p mask_source:=yolo
)
YOLO_CFG="${ROOT}/src/picking_perception/config/foundation_pose.yaml"
if [[ -f "${YOLO_CFG}" ]]; then
  ARGS+=(--params-file "${YOLO_CFG}")
fi
ARGS+=("$@")

exec ros2 run picking_perception global_detector_node "${ARGS[@]}"
