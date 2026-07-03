#!/usr/bin/env bash
# Train YOLO11 on merged dataset; output runs/detect/blueberry-4/weights/best.pt
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/datasets/blueberry/data.yaml"
ENV_NAME="${FOUNDATIONPOSE_CONDA_ENV:-foundationpose}"
WEIGHTS="${ROOT}/runs/detect/blueberry-4/weights/best.pt"

if [[ ! -f "${DATA}" ]]; then
  echo "[train] ERROR: ${DATA} not found." >&2
  echo "  Run: python3 scripts/prepare_yolo_dataset.py --val-count 15" >&2
  exit 1
fi

set +u
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

export PYTHONPATH=""
yolo detect train \
  model=yolo11n.pt \
  data="${DATA}" \
  epochs=80 \
  imgsz=640 \
  batch=8 \
  patience=15 \
  project="${ROOT}/runs/detect" \
  name=blueberry-4 \
  device=0

echo ""
echo "[train] Weights: ${WEIGHTS}"
echo "[train] Update config:"
echo "  python3 scripts/update_foundation_pose_yolo.py --model ${WEIGHTS} --conf 0.38"
