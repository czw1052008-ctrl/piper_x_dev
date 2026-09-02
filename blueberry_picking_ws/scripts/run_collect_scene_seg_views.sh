#!/usr/bin/env bash
# Collect multi-view RGB-D for scene_seg (moves wrist, then restores).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/config/real_robot.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT}/config/real_robot.env"
fi
set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
source "${ROOT}/scripts/_yolo_pythonpath.sh"
set +u
export PYTHONNOUSERSITE=1
exec /usr/bin/python3 "${ROOT}/scripts/collect_scene_seg_views.py" "$@"
