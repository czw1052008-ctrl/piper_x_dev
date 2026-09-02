#!/usr/bin/env bash
# Fast scene_seg labeling: brush + SAM click + YOLO berries (no polygons).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi
export PATH="/usr/bin:/bin:${HOME}/.local/bin:${PATH}"
export DISPLAY="${DISPLAY:-:1}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_yolo_pythonpath.sh"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-/usr/lib/x86_64-linux-gnu/qt5/plugins}"
PY="${ANNOTATE_PYTHON:-/usr/bin/python3}"
exec "${PY}" "${ROOT}/scripts/annotate_scene_seg.py" "$@"
