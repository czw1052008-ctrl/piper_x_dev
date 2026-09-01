#!/usr/bin/env bash
# Fixed mono: capture → VLM cluster prelabel → manual review → unified dataset prep
#
# Usage:
#   bash scripts/run_fixed_mono_label_pipeline.sh capture
#   bash scripts/run_fixed_mono_label_pipeline.sh prelabel          # Ollama local
#   bash scripts/run_fixed_mono_label_pipeline.sh prelabel-api        # OpenAI-compatible API
#   bash scripts/run_fixed_mono_label_pipeline.sh annotate
#   bash scripts/run_fixed_mono_label_pipeline.sh merge
#   bash scripts/run_fixed_mono_label_pipeline.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

IMG="${ROOT}/datasets/fixed_mono/images"
LAB="${ROOT}/datasets/fixed_mono/labels"
PREV="${ROOT}/datasets/fixed_mono/previews"

CMD="${1:-help}"

case "${CMD}" in
  capture)
    bash "${ROOT}/scripts/capture_fixed_mono_dataset.sh"
    ;;
  prelabel)
    python3 "${ROOT}/scripts/calibrate_vlm_cluster_prompt.py"
    echo "[pipeline] VLM prelabel via Ollama (qwen2.5vl:7b)"
    echo "  Tip: cluster count is NOT fixed — review previews before annotate"
    python3 "${ROOT}/scripts/teacher_prelabel_vlm.py" \
      --images "${IMG}" --labels "${LAB}" --preview-dir "${PREV}" \
      --provider ollama --model qwen2.5vl:7b --overwrite
    ;;
  prelabel-api)
    echo "[pipeline] VLM prelabel via API (set OPENAI_API_KEY or DASHSCOPE_API_KEY)"
    python3 "${ROOT}/scripts/teacher_prelabel_vlm.py" \
      --images "${IMG}" --labels "${LAB}" --preview-dir "${PREV}" \
      --provider openai \
      --base-url "${VLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
      --model "${VLM_MODEL:-qwen-vl-max}" \
      --api-key-env "${VLM_API_KEY_ENV:-DASHSCOPE_API_KEY}" \
      --overwrite
    ;;
  annotate)
    python3 "${ROOT}/scripts/annotate_blueberry.py" \
      --images "${IMG}" --labels "${LAB}"
    ;;
  calibrate)
    python3 "${ROOT}/scripts/calibrate_vlm_cluster_prompt.py"
    ;;
  merge)
    python3 "${ROOT}/scripts/calibrate_vlm_cluster_prompt.py"
    python3 "${ROOT}/scripts/prepare_yolo_unified_dataset.py"
    ;;
  all)
    bash "$0" capture
    bash "$0" prelabel
    echo ""
    echo "Review labels in datasets/fixed_mono/previews/ then:"
    echo "  bash scripts/run_fixed_mono_label_pipeline.sh annotate"
    echo "  bash scripts/run_fixed_mono_label_pipeline.sh merge"
    ;;
  help|*)
    cat <<EOF
Usage: bash scripts/run_fixed_mono_label_pipeline.sh <step>

  capture      Interactive minimal fixed-camera capture (~18 shots)
  prelabel     Qwen2.5-VL via Ollama (local 7B)
  prelabel-api Qwen-VL via OpenAI-compatible API
  calibrate    Rebuild vlm_calibration.json from human labels
  annotate     OpenCV GUI review / fix bboxes
  merge        calibrate + merge wrist+fixed → datasets/blueberry_unified
  all          capture + prelabel (then annotate manually)

Ollama one-time setup:
  ollama pull qwen2.5vl:7b
EOF
    ;;
esac
