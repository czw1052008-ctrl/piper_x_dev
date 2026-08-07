#!/usr/bin/env bash
# Auto YOLO bbox pre-label — then review in annotate GUI.
# Uses system python + ~/.local ultralytics (same as train_blueberry_yolo_v5.sh).
# Optional: FOUNDATIONPOSE_CONDA_ENV / PRELABEL_CONDA_ENV to force a conda env.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/datasets/blueberry/images" "${ROOT}/datasets/blueberry/labels"

ENV_NAME="${PRELABEL_CONDA_ENV:-${FOUNDATIONPOSE_CONDA_ENV:-}}"

if [[ -n "${ENV_NAME}" ]]; then
  set +u
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
  set -u
  export PYTHONPATH="${ROOT}/src/picking_perception${PYTHONPATH:+:${PYTHONPATH}}"
  exec python3 "${ROOT}/scripts/prelabel_blueberry.py" "$@"
fi

# Default: system Python (avoid conda base NumPy / missing cv2).
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi
export PATH="/usr/bin:/bin:${PATH}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_yolo_pythonpath.sh"
export PYTHONPATH="${ROOT}/src/picking_perception${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PRELABEL_PYTHON:-/usr/bin/python3}"
if ! "${PY}" -c 'from ultralytics import YOLO' 2>/dev/null; then
  echo "[prelabel] ERROR: ${PY} cannot import ultralytics." >&2
  echo "  Fix: source scripts/_yolo_pythonpath.sh and ensure ~/.local has ultralytics" >&2
  exit 1
fi
exec "${PY}" "${ROOT}/scripts/prelabel_blueberry.py" "$@"
