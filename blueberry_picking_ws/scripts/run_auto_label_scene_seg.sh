#!/usr/bin/env bash
# Auto-label scene_seg color frames (review overlays afterwards).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi
export PATH="/usr/bin:/bin:${PATH}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_yolo_pythonpath.sh"
export PYTHONNOUSERSITE=1
exec /usr/bin/python3 "${ROOT}/scripts/auto_label_scene_seg.py" "$@"
