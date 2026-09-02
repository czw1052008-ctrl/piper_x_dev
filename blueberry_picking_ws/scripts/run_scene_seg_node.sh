#!/usr/bin/env bash
# DINOv3 scene segmentation on fixed + wrist RGB (same checkpoint).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/config/real_robot.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT}/config/real_robot.env"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi
set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
set -u 2>/dev/null || true
# ROS dist-packages ship Pillow 9.0 without Image.Resampling; transformers DINOv3 needs newer.
export PYTHONPATH="${HOME}/.local/lib/python3.10/site-packages:${ROOT}/src/picking_perception:${PYTHONPATH:-}"
unset PYTHONNOUSERSITE
CKPT="${SCENE_SEG_CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  for cand in \
      "${ROOT}/runs/seg/dinov3-yolo-hm/best.pt" \
      "${ROOT}/runs/seg/dinov3-overfit-10/best.pt"; do
    if [[ -f "${cand}" ]]; then
      CKPT="${cand}"
      break
    fi
  done
fi
CKPT="${CKPT:-${ROOT}/runs/seg/dinov3-overfit-10/best.pt}"
ARGS=(--ros-args
  -p publish_hz:=2.0
  -p image_topic:=/camera_fixed/color/image_raw
  -p camera_frame:=camera_fixed_color_optical_frame
  -p wrist_image_topic:=/camera_wrist/color/image_raw
  -p wrist_camera_frame:=camera_wrist_color_optical_frame
  -p input_size:=448
  -p checkpoint:="${CKPT}"
)
ARGS+=("$@")
exec ros2 run picking_perception scene_seg_node "${ARGS[@]}"
