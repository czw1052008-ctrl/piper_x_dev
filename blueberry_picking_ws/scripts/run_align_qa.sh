#!/usr/bin/env bash
# Small-step ALIGNING QA (yaw-face): snap wrist+fixed before/after; align-only (no approach).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 HOME=/home/user
export ROS_LOG_DIR="${ROOT}/log/real_robot/ros_logs"
mkdir -p "${ROS_LOG_DIR}"
source /opt/ros/humble/setup.bash
source install/setup.bash

BLEND="${1:-0.45}"
ORIENT_FRAC="${2:-0.55}"
TRAJ_S="${3:-4.0}"
JUDGE_MODE="${ALIGN_JUDGE_MODE:-file}"
VLM_PROVIDER="${ALIGN_VLM_PROVIDER:-agent}"
QA_DIR="${ROOT}/log/real_robot/qa"

pkill -f 'scripts/reach_fsm_node.py' 2>/dev/null || true
sleep 1
mkdir -p "${QA_DIR}"
: > log/real_robot/reach_fsm.log
if [[ "${JUDGE_MODE}" == "file" ]]; then
  pkill -f 'scripts/align_vlm_decider.py' 2>/dev/null || true
  echo "[run_align_qa] file judge provider=${VLM_PROVIDER}"
  if [[ "${VLM_PROVIDER}" == "heuristic" || "${VLM_PROVIDER}" == "agent" ]]; then
    /usr/bin/python3 scripts/align_vlm_decider.py \
      --qa-dir "${QA_DIR}" \
      --provider "${VLM_PROVIDER}" \
      > log/real_robot/align_vlm_decider.log 2>&1 &
  fi
fi
/usr/bin/python3 scripts/reach_fsm_node.py \
  --ee-link link6 \
  --align-only \
  --align-judge-mode "${JUDGE_MODE}" \
  --align-traj-s "${TRAJ_S}" \
  --align-standoff-m 0.28 \
  --align-eye-height-m 0.10 \
  --align-fine-visible-conf 0.55 \
  --fine-assoc-max-m 0.22 \
  --fine-max-age-s 0.5 \
  --align-max-steps 10 \
  --qa-dir "${QA_DIR}" \
  >> log/real_robot/reach_fsm.log 2>&1 &
FSM=$!
echo "FSM pid=${FSM}"
sleep 3
bash scripts/snap_reach_qa.sh qa_before
ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: start}"
for i in $(seq 1 90); do
  sleep 1
  st=$(timeout 2 ros2 topic echo /reach/status --once 2>/dev/null | awk '/data:/{print $2; exit}' || true)
  echo "t=${i} status=${st}"
  case "${st}" in WAIT_CONFIRM|ERROR*) break;; esac
done
sleep 1
bash scripts/snap_reach_qa.sh qa_after
echo '--- fsm log ---'
tail -40 log/real_robot/reach_fsm.log
# leave FSM up so user can confirm_reset; or kill
# kill "${FSM}" 2>/dev/null || true
echo "FSM still running pid=${FSM} — publish confirm_reset to home, or abort"
ls -1dt log/real_robot/qa/*/ | head -6
