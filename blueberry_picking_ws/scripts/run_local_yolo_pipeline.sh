#!/usr/bin/env bash
# Local blueberry YOLO pipeline: capture -> annotate -> prepare -> train
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${ROOT}/datasets/blueberry/images"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_local_yolo_pipeline.sh capture [N] [interval_sec]
  bash scripts/run_local_yolo_pipeline.sh prelabel [--overwrite]
  bash scripts/run_local_yolo_pipeline.sh annotate
  bash scripts/run_local_yolo_pipeline.sh prepare [val_count]
  bash scripts/run_local_yolo_pipeline.sh train
  bash scripts/run_local_yolo_pipeline.sh all [N]

Steps:
  1. capture   - save wrist-camera frames to datasets/blueberry/images/
  2. prelabel  - YOLOE auto bbox (then review in annotate)
  3. annotate  - confirm / fix boxes (GUI)
  3. prepare   - split train/val + data.yaml
  4. train     - yolo11n detect fine-tune
EOF
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  capture)
    N="${1:-40}"; INT="${2:-2}"
    mkdir -p "${IMG}"
    bash "${ROOT}/scripts/capture_annotation_dataset.sh" "${N}" "${INT}"
    echo "[pipeline] images in ${IMG}"
    ;;
  annotate)
    bash "${ROOT}/scripts/run_annotate_blueberry.sh"
    ;;
  prelabel)
    bash "${ROOT}/scripts/run_prelabel_blueberry.sh" "$@"
    ;;
  import-web)
    IMPORT_DIR="${1:?Usage: ... import-web <unzipped_yolo_export_dir> [prefix]}"
    PREFIX="${2:-web_}"
    bash "${ROOT}/scripts/run_import_external_yolo.sh" "${IMPORT_DIR}" --prefix "${PREFIX}"
    python3 "${ROOT}/scripts/prepare_yolo_dataset.py"
    echo "[pipeline] Merged dataset ready. Train: bash scripts/run_local_yolo_pipeline.sh train"
    ;;
  prepare)
    VAL_COUNT="${1:-10}"
    python3 "${ROOT}/scripts/prepare_yolo_dataset.py" --val-count "${VAL_COUNT}"
    ;;
  train)
    bash "${ROOT}/scripts/train_blueberry_yolo.sh"
    ;;
  all)
    N="${1:-40}"
    bash "$0" capture "${N}" 2
    echo "[pipeline] Auto pre-label, then review in GUI:"
    bash "$0" prelabel --overwrite
    bash "$0" annotate
  bash "$0" prepare
  bash "$0" train
    ;;
  -h|--help|"") usage ;;
  *) echo "Unknown: ${cmd}"; usage; exit 1 ;;
esac
