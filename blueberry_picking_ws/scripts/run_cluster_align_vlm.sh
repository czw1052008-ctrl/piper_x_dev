#!/usr/bin/env bash
# Initial pose → cluster align pose via fixed-mono cluster lock + local VLM ALIGN.
#
# Flow:
#   1. Bringup arm + dual Orbbec + global YOLO cluster detector + fine detector
#   2. reach_fsm: LOCK (file pick cluster) → ALIGN (Ollama VLM joint trajectories)
#   3. coarse_ok → save refine_entry_pose.json → WAIT_CONFIRM (hold align pose)
#
# Usage:
#   bash scripts/run_cluster_align_vlm.sh
#   bash scripts/run_cluster_align_vlm.sh --cluster-index 1
#   bash scripts/run_cluster_align_vlm.sh --dry-run   # stack only, no start
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
CONFIG="${ROOT}/config/real_robot.env"
QA_DIR="${ROOT}/log/real_robot/qa"
POSE="${ROOT}/log/real_robot/refine_entry_pose.json"
LOG="${ROOT}/log/real_robot/cluster_align_vlm.log"

CLUSTER_INDEX=""
AUTO_LOCK=true
DRY_RUN=false
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster-index) CLUSTER_INDEX="$2"; AUTO_LOCK=false; shift 2 ;;
    --auto) AUTO_LOCK=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

mkdir -p "${QA_DIR}" "$(dirname "${LOG}")"
: > "${LOG}"
log() { echo "$*" | tee -a "${LOG}"; }

log "=== cluster align VLM $(date -Iseconds) ==="

# --- Stack ---
bash "${ROOT}/scripts/real_robot_bringup.sh" \
  --config "${CONFIG}" --fixed-cam --reach-perception --no-wait \
  "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee -a "${LOG}"

set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
if [[ -f "${AGX_ARM_WS:-}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${AGX_ARM_WS}/install/setup.bash"
fi
set -u 2>/dev/null || true

# Global cluster detector (YOLO class=cluster)
if ! pgrep -f 'global_detector_node' >/dev/null; then
  log "[stack] starting global_detector_node"
  nohup bash "${ROOT}/scripts/run_global_detector_node.sh" \
    >> "${ROOT}/log/real_robot/global_detector.log" 2>&1 &
  sleep 4
fi

# Fine detector (wrist, for ALIGN obs / coarse_ok visibility)
if ! pgrep -f 'fine_detector_node' >/dev/null; then
  log "[stack] starting fine_detector_node"
  nohup bash "${ROOT}/scripts/run_fine_detector_node.sh" \
    -p depth_pose_source:=depth \
    -p lock_bbox_file:=/dev/null \
    >> "${ROOT}/log/real_robot/fine_detector.log" 2>&1 &
  sleep 4
fi

# Wrist static TF
_republish_wrist_tf() {
  set +u
  # shellcheck disable=SC1091
  source "${CONFIG}" 2>/dev/null || true
  set -u
  kill $(ps -eo pid,args | awk '/static_transform_publisher.*camera_wrist_color_optical_frame/ {print $1}') 2>/dev/null || true
  sleep 0.3
  IFS=',' read -r _roll _pitch _yaw <<< "${CAMERA_MOUNT_RPY:-0,0,0}"
  nohup ros2 run tf2_ros static_transform_publisher \
    --x "${CAMERA_MOUNT_TX:-0}" --y "${CAMERA_MOUNT_TY:-0}" --z "${CAMERA_MOUNT_TZ:-0}" \
    --roll "${_roll:-0}" --pitch "${_pitch:-0}" --yaw "${_yaw:-0}" \
    --frame-id link6 --child-frame-id camera_wrist_color_optical_frame \
    >> "${ROOT}/log/real_robot/camera_wrist_tf.log" 2>&1 &
  sleep 0.8
}
if ! pgrep -f 'static_transform_publisher.*camera_wrist_color_optical_frame' >/dev/null; then
  _republish_wrist_tf
fi

# Ollama check
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  log "WARN: Ollama not reachable — start: ollama serve && ollama pull qwen2.5vl:7b"
fi

# reach_fsm (VLM align + file cluster lock, align-only stops before PBVS)
if pgrep -f 'python3 scripts/reach_fsm_node.py' >/dev/null; then
  pkill -f 'python3 scripts/reach_fsm_node.py' || true
  sleep 2
fi

export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0 PYTHONNOUSERSITE=1

log "[stack] starting reach_fsm (vlm align + file lock, align-only)"
nohup /usr/bin/python3 "${ROOT}/scripts/reach_fsm_node.py" \
  --ee-link link6 \
  --align-judge-mode vlm \
  --align-only \
  --align-traj-s 4.0 \
  --align-max-steps 12 \
  --align-settle-s 1.2 \
  --align-timeout-s 20.0 \
  --qa-dir "${QA_DIR}" \
  >> "${ROOT}/log/real_robot/reach_fsm.log" 2>&1 &
sleep 3

timeout 12 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "${LOG}" | tail -1 || true
timeout 8 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "${LOG}" | tail -1 || true

if [[ "${DRY_RUN}" == true ]]; then
  cat <<EOF | tee -a "${LOG}"

[DRY-RUN] Stack up. Manual steps:
  ros2 topic pub --once /reach/cmd std_msgs/String "{data: start}"
  python3 scripts/wait_and_lock_cluster.py --qa-dir ${QA_DIR} --wait-start --auto
  # or: --index N
  ros2 topic echo /reach/status
  python3 scripts/refine_entry_pose.py check

EOF
  exit 0
fi

log "=== START: home → LOCK cluster → VLM ALIGN ==="
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1 || true
sleep 1
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: start}" 2>&1 | tee -a "${LOG}"

LOCK_ARGS=(--qa-dir "${QA_DIR}" --wait-start --timeout-s 90)
if [[ "${AUTO_LOCK}" == true ]]; then
  LOCK_ARGS+=(--auto)
else
  LOCK_ARGS+=(--index "${CLUSTER_INDEX}")
fi
python3 "${ROOT}/scripts/wait_and_lock_cluster.py" "${LOCK_ARGS[@]}" 2>&1 | tee -a "${LOG}"

log "=== waiting VLM ALIGN → WAIT_CONFIRM ==="
for i in $(seq 1 120); do
  if grep -aE 'state → WAIT_CONFIRM|coarse_ok|ALIGNING → fine' "${ROOT}/log/real_robot/reach_fsm.log" 2>/dev/null | tail -1 | grep -q .; then
    log "ALIGN done (tick $i)"
    break
  fi
  if grep -aE 'state → ERROR' "${ROOT}/log/real_robot/reach_fsm.log" 2>/dev/null | tail -3 | grep -q 'state → ERROR'; then
    log "FAIL: reach_fsm ERROR — see log/real_robot/reach_fsm.log"
    tail -30 "${ROOT}/log/real_robot/reach_fsm.log" | tee -a "${LOG}"
    exit 2
  fi
  sleep 2
done

if [[ ! -f "${POSE}" ]]; then
  log "WARN: ${POSE} not found (coarse_ok may not have saved yet)"
else
  log "Saved align pose: $(python3 -c "import json; d=json.load(open('${POSE}')); print(d.get('joints_deg'))")"
  python3 "${ROOT}/scripts/refine_entry_pose.py" check --max-deg 8 2>&1 | tee -a "${LOG}" || true
fi

cat <<EOF | tee -a "${LOG}"

================================================================================
  Cluster align VLM complete (arm holding align pose, state=WAIT_CONFIRM).
  Next: pick cycle  →  bash scripts/run_pick_system.sh
  Or abort/home     →  ros2 topic pub --once /reach/cmd std_msgs/String "{data: confirm_reset}"
  Shutdown          →  bash scripts/_safe_poweroff.sh
================================================================================

EOF
