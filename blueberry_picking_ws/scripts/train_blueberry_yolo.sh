#!/usr/bin/env bash
# Train YOLO11 detect on local blueberry bbox dataset (conda foundationpose).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/datasets/blueberry/data.yaml"
ENV_NAME="${FOUNDATIONPOSE_CONDA_ENV:-foundationpose}"
WEIGHTS="${ROOT}/runs/detect/blueberry/weights/best.pt"

if [[ ! -f "${DATA}" ]]; then
  echo "[train] ERROR: ${DATA} not found." >&2
  echo "  Run: python3 scripts/prepare_yolo_dataset.py" >&2
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
  name=blueberry \
  device=0

echo ""
echo "[train] Weights: ${WEIGHTS}"
echo "[train] Edit src/picking_perception/config/foundation_pose.yaml:"
echo "  yolo_model: ${WEIGHTS}"
echo "  yolo_open_vocab: false"
echo "  yolo_conf_threshold: 0.35"
echo "[train] Then: pkill -f fine_detector_node; bash scripts/run_perception_only.sh"
