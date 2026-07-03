#!/usr/bin/env bash
# Batch3 YOLO pipeline: collect at pick poses -> prelabel -> annotate -> merge -> train -> yaml
#
# Usage:
#   bash scripts/yolo_batch3_pipeline.sh collect [count] [interval_sec]
#   bash scripts/yolo_batch3_pipeline.sh annotate
#   bash scripts/yolo_batch3_pipeline.sh finish [val_count]
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG_B3="${ROOT}/datasets/blueberry/images_batch3"
LAB_B3="${ROOT}/datasets/blueberry/labels_batch3"
PREVIEW_B3="${ROOT}/datasets/blueberry/previews_batch3"
MODEL_B3="${ROOT}/runs/detect/blueberry-3/weights/best.pt"
WEIGHTS_V4="${ROOT}/runs/detect/blueberry-4/weights/best.pt"

usage() {
  cat <<EOF
Usage:
  bash scripts/yolo_batch3_pipeline.sh collect [count] [interval_sec]
  bash scripts/yolo_batch3_pipeline.sh annotate
  bash scripts/yolo_batch3_pipeline.sh finish [val_count]

Steps:
  collect   - bringup arm+camera, move to teleop scan poses, save 50 frames, prelabel
  annotate  - review/fix bboxes in GUI (batch3 dirs)
  finish    - merge batch2+batch3, prepare train/val, train blueberry-4, update yaml

After collect:
  bash scripts/run_annotate_blueberry.sh --images ${IMG_B3} --labels ${LAB_B3}

After finish:
  pkill -f fine_detector_node; bash scripts/run_perception_only.sh
EOF
}

_wait_topic() {
  local topic="$1"
  local timeout="${2:-90}"
  local t0
  t0=$(date +%s)
  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "${topic}"; then
      return 0
    fi
    if (( $(date +%s) - t0 >= timeout )); then
      echo "[batch3] ERROR: topic ${topic} not up after ${timeout}s" >&2
      return 1
    fi
    sleep 2
  done
}

_wait_action() {
  local action="$1"
  local timeout="${2:-90}"
  local t0
  t0=$(date +%s)
  while true; do
    if ros2 action list 2>/dev/null | grep -qx "${action}"; then
      return 0
    fi
    if (( $(date +%s) - t0 >= timeout )); then
      echo "[batch3] ERROR: action ${action} not up after ${timeout}s" >&2
      return 1
    fi
    sleep 2
  done
}

_ros_env() {
  conda deactivate 2>/dev/null || true
  export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
  export RMW_FASTRTPS_USE_SHM=0
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  # shellcheck disable=SC1090
  source "${ROOT}/install/setup.bash"
  if [[ -f "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash"
  fi
}

cmd_collect() {
  local count="${1:-50}"
  local interval="${2:-2}"

  mkdir -p "${IMG_B3}" "${LAB_B3}" "${PREVIEW_B3}"

  if [[ ! -f "${MODEL_B3}" ]]; then
    echo "[batch3] ERROR: prelabel model missing: ${MODEL_B3}" >&2
    exit 1
  fi

  echo "[batch3] Stopping old stack ..."
  bash "${ROOT}/scripts/real_robot_shutdown.sh" --quiet --no-disable || true
  sleep 2

  echo "[batch3] Starting arm + camera (no perception) ..."
  bash "${ROOT}/scripts/real_robot_bringup.sh" --no-wait

  _ros_env
  echo "[batch3] Waiting for camera and arm ..."
  _wait_topic /camera_wrist/color/image_raw 120
  _wait_topic /feedback/joint_states 120
  _wait_action /move_action 120

  bash "${ROOT}/scripts/capture_pick_poses_at_scan.sh" "${count}" "${interval}"

  echo "[batch3] Prelabel with ${MODEL_B3} ..."
  bash "${ROOT}/scripts/run_prelabel_blueberry.sh" \
    --images "${IMG_B3}" \
    --labels "${LAB_B3}" \
    --custom-model \
    --model "${MODEL_B3}" \
    --conf 0.20 \
    --overwrite \
    --preview-dir "${PREVIEW_B3}"

  echo ""
  echo "[batch3] Collect done."
  echo "  Images:  ${IMG_B3}"
  echo "  Labels:  ${LAB_B3}"
  echo "  Preview: ${PREVIEW_B3}"
  echo ""
  echo "Review annotations:"
  echo "  bash scripts/yolo_batch3_pipeline.sh annotate"
  echo "Then merge + train:"
  echo "  bash scripts/yolo_batch3_pipeline.sh finish"
}

cmd_annotate() {
  mkdir -p "${IMG_B3}" "${LAB_B3}"
  bash "${ROOT}/scripts/run_annotate_blueberry.sh" \
    --images "${IMG_B3}" \
    --labels "${LAB_B3}"
}

cmd_finish() {
  local val_count="${1:-15}"

  if [[ -d "${ROOT}/datasets/blueberry/images_batch2" ]] \
     && [[ -n "$(ls -A "${ROOT}/datasets/blueberry/images_batch2" 2>/dev/null)" ]]; then
    echo "[batch3] Merging batch1 + batch2 ..."
    bash "${ROOT}/scripts/merge_batch2_dataset.sh"
  fi

  echo "[batch3] Merging batch3 ..."
  bash "${ROOT}/scripts/merge_batch3_dataset.sh"

  echo "[batch3] Preparing train/val split (val=${val_count}) ..."
  python3 "${ROOT}/scripts/prepare_yolo_dataset.py" --val-count "${val_count}"

  echo "[batch3] Training blueberry-4 ..."
  bash "${ROOT}/scripts/train_blueberry_yolo_v4.sh"

  if [[ ! -f "${WEIGHTS_V4}" ]]; then
    echo "[batch3] ERROR: training did not produce ${WEIGHTS_V4}" >&2
    exit 1
  fi

  python3 "${ROOT}/scripts/update_foundation_pose_yolo.py" \
    --model "${WEIGHTS_V4}" \
    --conf 0.38

  echo ""
  echo "[batch3] Finish complete."
  echo "  Weights: ${WEIGHTS_V4}"
  echo "  Restart perception:"
  echo "    pkill -f fine_detector_node; bash scripts/run_perception_only.sh"
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  collect) cmd_collect "$@" ;;
  annotate) cmd_annotate ;;
  finish) cmd_finish "$@" ;;
  -h|--help|"") usage ;;
  *) echo "Unknown: ${cmd}"; usage; exit 1 ;;
esac
