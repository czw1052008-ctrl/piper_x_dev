#!/usr/bin/env bash
# LOCK → ALIGN (fixed-mono judge) → save refine_entry_pose → abort (hold pose, no REFINING).
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0 PYTHONNOUSERSITE=1 HOME=/home/user
export LD_LIBRARY_PATH=/home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/orbbec_camera/lib:${LD_LIBRARY_PATH:-}
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
set +u

POSE=log/real_robot/refine_entry_pose.json
LOG=log/real_robot/_align_calibrate_only.txt
: > "$LOG"
log() { echo "$*" | tee -a "$LOG"; }
fsm_has() { grep -aEq "$1" log/real_robot/reach_fsm.log 2>/dev/null; }

align_phase() {
  local from_log
  from_log=$(grep -aE 'waiting for decision file phase=' log/real_robot/reach_fsm.log 2>/dev/null \
    | tail -1 | sed -n 's/.*phase=\([a-z]*\).*/\1/p')
  [[ -n "$from_log" ]] && { echo "$from_log"; return; }
  [[ -z "${SESS:-}" ]] && return
  python3 - <<'PY'
import json, glob, os, sys
from pathlib import Path
sess = Path(os.environ.get('SESS', ''))
if not sess.is_dir():
    sys.exit(0)
if (sess / 'align_decision.json').is_file():
    sys.exit(0)
files = sorted(
    list(sess.glob('align_*_judge.json'))
    + list(sess.glob('align_*_after_*.json'))
    + list(sess.glob('align_*_request.json')),
    key=lambda p: p.stat().st_mtime,
)
if not files:
    sys.exit(0)
p = files[-1]
if p.name.endswith('_judge.json') or p.name.endswith('_after_'):
    print('judge')
elif p.name.endswith('_request.json'):
    with open(p) as f:
        ph = json.load(f).get('phase', 'command')
    print(ph if ph in ('command', 'judge') else 'command')
PY
}

align_step() {
  python3 - <<PY
import json,glob,os
sess=os.environ.get('SESS','')
files=sorted(
    glob.glob(sess+'align_*_judge.json')
    +glob.glob(sess+'align_*_after_*.json')
    +glob.glob(sess+'align_*_request.json'),
    key=os.path.getmtime)
print(int(json.load(open(files[-1])).get('step_idx',0)) if files else 0)
PY
}

align_decision_pending() {
  local phase="$1"
  local f="${SESS}align_decision.json"
  [[ -f "$f" ]] || return 0
  local existing_phase
  existing_phase=$(python3 -c "import json; print(json.load(open('${f}')).get('phase',''))" 2>/dev/null)
  if [[ "$existing_phase" == "$phase" ]]; then
    return 1
  fi
  rm -f "$f"
  return 0
}

bash scripts/_retest_fsm.sh
sleep 3

timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -1
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -1

log '=== ALIGN CALIBRATE: abort → start (LOCK+ALIGN) ==='
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1
sleep 1
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: start}" 2>&1 | tee -a "$LOG" | tail -1

SESS=""
for i in $(seq 1 50); do
  for d in $(ls -1dt log/real_robot/qa/*/ 2>/dev/null | head -8); do
    [[ -f "${d}lock_region_request.json" ]] || continue
    age=$(( $(date +%s) - $(stat -c %Y "${d}lock_region_request.json") ))
    if (( age < 120 )); then SESS="$d"; break; fi
  done
  [[ -n "$SESS" ]] && break
  sleep 1
done
[[ -z "$SESS" ]] && { log FAIL_NO_LOCK; exit 1; }
log "SESS=$SESS"
python3 scripts/write_lock_region_decision.py --session-dir "$SESS" --index 0 \
  --reason 'align calibrate fixed-mono' | tee -a "$LOG"

export SESS
for round in $(seq 1 30); do
  fsm_has 'state → REFINING' && { log ENTER_REFINING; break; }
  fsm_has 'state → ERROR' && { log ALIGN_ERROR; tail -20 log/real_robot/reach_fsm.log | tee -a "$LOG"; exit 2; }
  PHASE=$(align_phase)
  [[ -z "$PHASE" ]] && { sleep 1; continue; }
  align_decision_pending "$PHASE" || { sleep 1; continue; }
  STEP=$(align_step)
  log "round=$round phase=$PHASE step=$STEP"
  python3 scripts/apply_align_decision_from_request.py \
    --session-dir "$SESS" --step-idx "$STEP" --phase "$PHASE" | tee -a "$LOG"
  for _ in $(seq 1 35); do
    fsm_has 'state → REFINING|hold after|coarse_ok' && break
    [[ -f "${SESS}align_decision.json" ]] || break
    sleep 1
  done
  sleep 2
done

if ! fsm_has 'state → REFINING'; then
  log FAIL_NO_ALIGN_DONE
  grep -aE 'ALIGN|ERROR|waiting|coarse|fixed' log/real_robot/reach_fsm.log | tail -25 | tee -a "$LOG"
  exit 3
fi

[[ -f "$POSE" ]] || { log FAIL_NO_POSE; exit 4; }
log "ALIGN_OK pose=$(python3 -c "import json; print(json.load(open('$POSE'))['joints_deg'])")"
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" 2>&1 | tee -a "$LOG" | tail -1
sleep 2
python3 scripts/refine_entry_pose.py check --max-deg 8 | tee -a "$LOG"
log DONE
exit 0
