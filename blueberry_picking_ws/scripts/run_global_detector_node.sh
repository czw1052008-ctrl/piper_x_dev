#!/usr/bin/env bash
# Start global_detector_node (fixed-camera topic stream).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
set -u 2>/dev/null || true
exec ros2 run picking_perception global_detector_node --ros-args \
  -p publish_hz:=3.0 \
  -p image_topic:=/camera_fixed/image_raw \
  "$@"
