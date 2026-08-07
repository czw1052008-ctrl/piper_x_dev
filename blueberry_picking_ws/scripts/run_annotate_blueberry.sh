#!/usr/bin/env bash
# Local bbox annotation (needs GUI / WSLg).
# Use system Python — conda base often lacks apt OpenCV (cv2).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/datasets/blueberry/images" "${ROOT}/datasets/blueberry/labels"

# Prefer /usr/bin/python3 (python3-opencv). Drop conda from PATH for this process.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi
export PATH="/usr/bin:/bin:${PATH}"
PY="${ANNOTATE_PYTHON:-/usr/bin/python3}"
if ! "${PY}" -c 'import cv2' 2>/dev/null; then
  echo "[annotate] ERROR: ${PY} has no cv2. Install: sudo apt install python3-opencv" >&2
  exit 1
fi
exec "${PY}" "${ROOT}/scripts/annotate_blueberry.py" "$@"
