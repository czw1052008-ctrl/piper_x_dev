#!/usr/bin/env bash
# Train YOLO11 on merged dataset → runs/detect/blueberry-5/weights/best.pt
# Uses system python + ~/.local ultralytics (see _yolo_pythonpath.sh).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-blueberry-5}"
DATA="${2:-${ROOT}/datasets/blueberry/data.yaml}"

if [[ ! -f "${DATA}" ]]; then
  echo "[train] ERROR: ${DATA} not found." >&2
  echo "  Run: python3 scripts/prepare_yolo_dataset.py --val-count 20" >&2
  echo "  Or:  python3 scripts/prepare_yolo_unified_dataset.py" >&2
  exit 1
fi

# Avoid conda base (wrong NumPy / no system cv2). Match annotate launcher.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi
export PATH="/usr/bin:/bin:${PATH}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_yolo_pythonpath.sh"
PY="${TRAIN_PYTHON:-/usr/bin/python3}"

cd "${ROOT}"
"${PY}" - <<PY
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='${DATA}',
    epochs=80,
    imgsz=640,
    batch=8,
    patience=15,
    project='${ROOT}/runs/detect',
    name='${NAME}',
    device=0,
    exist_ok=True,
)
print('[train] done')
PY

echo ""
echo "[train] Weights: ${ROOT}/runs/detect/${NAME}/weights/best.pt"
echo "[train] Update config:"
echo "  python3 scripts/update_foundation_pose_yolo.py --model ${ROOT}/runs/detect/${NAME}/weights/best.pt --conf 0.35"
