#!/usr/bin/env bash
# Automated REFINING debug: calibrate ALIGN entry pose if missing, restore, start_refine.
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
LOG=log/real_robot/_refine_debug_auto.txt
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
if p.name.endswith('_judge.json') or '_after_' in p.name:
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

# Return 0 when we should write a new align_decision for the FSM-waiting phase.
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

# Ensure FSM with latest REFINING logic
bash scripts/_retest_fsm.sh
sleep 3

go_home() {
  log '=== HOME (move_home) ==='
  timeout 15 ros2 service call /move_home std_srvs/srv/Empty "{}" 2>&1 | tee -a "$LOG" | tail -2
  for i in $(seq 1 25); do
    j2=$(timeout 2 ros2 topic echo /feedback/joint_states --once 2>/dev/null \
      | awk '/position:/{getline; print $2; exit}')
    python3 -c "import sys; v=float('$j2'); sys.exit(0 if abs(v)<0.25 else 1)" 2>/dev/null && return 0
    sleep 1
  done
  log 'WARN: home may not have completed'
}

timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -1
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -1

NEED_CALIB=0
if [[ "${FORCE_ALIGN_CALIB:-0}" == 1 ]]; then
  log 'FORCE_ALIGN_CALIB=1'
  NEED_CALIB=1
elif [[ ! -f "$POSE" ]]; then
  log "NO_CALIB: missing $POSE"
  NEED_CALIB=1
elif ! python3 -c "
import json,sys
d=json.load(open('$POSE'))
src=str(d.get('source',''))
sys.exit(0 if 'fixed-mono' in src or 'coarse facing' in src.lower() else 1)
" 2>/dev/null; then
  log 'NO_CALIB: saved pose not from fixed-mono coarse_ok — recalibrate ALIGN'
  NEED_CALIB=1
elif ! python3 scripts/refine_entry_pose.py check --max-deg 12; then
  log "NO_CALIB: arm far from saved refine entry pose"
  NEED_CALIB=1
fi

if (( NEED_CALIB )); then
  log '=== CALIBRATE: home → LOCK → ALIGN → save entry pose ==='
  timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1
  sleep 1
  go_home
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
  log "CALIB_SESS=$SESS"
  python3 scripts/write_lock_region_decision.py --session-dir "$SESS" --index 0 \
    --reason 'refine debug auto calib' | tee -a "$LOG"

  export SESS
  for round in $(seq 1 20); do
    fsm_has 'state → REFINING' && { log CALIB_ENTER_REFINING; break; }
    fsm_has 'state → ERROR' && { log CALIB_ERROR; tail -15 log/real_robot/reach_fsm.log | tee -a "$LOG"; exit 2; }
    PHASE=$(align_phase)
    [[ -z "$PHASE" ]] && { sleep 1; continue; }
    align_decision_pending "$PHASE" || { sleep 1; continue; }
    STEP=$(align_step)
    log "calib round=$round phase=$PHASE step=$STEP"
    python3 scripts/apply_align_decision_from_request.py \
      --session-dir "$SESS" --step-idx "$STEP" --phase "$PHASE" | tee -a "$LOG"
    for _ in $(seq 1 30); do
      fsm_has 'state → REFINING|hold after|coarse_ok|phase=judge' && break
      [[ -f "${SESS}align_decision.json" ]] || break
      sleep 1
    done
    sleep 2
  done
  fsm_has 'state → REFINING' || { log FAIL_NO_REFINE_AFTER_CALIB; exit 3; }
  [[ -f "$POSE" ]] || { log FAIL_NO_POSE_FILE; exit 4; }
  log "CALIB_OK pose=$(cat "$POSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('joints_deg'))")"
  # Pause REFINING motion — restore+restart from saved pose for clean debug
  timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1
  sleep 2
fi

log '=== RESTORE refine entry pose ==='
python3 scripts/refine_entry_pose.py restore --traj-s 4.0 --settle-s 1.5 | tee -a "$LOG" || exit 5

log '=== START_REFINE ==='
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: start_refine}" 2>&1 | tee -a "$LOG" | tail -2

log '=== MONITOR REFINING ==='
for i in $(seq 1 200); do
  st=$(timeout 2 ros2 topic echo /reach/status --once 2>/dev/null | awk -F"'" '/data:/{print $2; exit}')
  [[ -n "$st" ]] && log "t=$i status=$st"
  [[ "$st" == "REFINING_WAIT_NEAR" ]] && {
    log 'WAIT_NEAR: inspect lock viz — confirm_near manually when ready'
    log "  viz: $LOCK_VIZ"
    log '  ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: confirm_near}"'
  }
  [[ "$st" == "WAIT_CONFIRM" ]] && { log SUCCESS; exit 0; }
  [[ "$st" == ERROR* ]] && {
    log ERROR
    timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1
    sleep 1
    python3 scripts/refine_entry_pose.py restore --traj-s 5.0 --settle-s 2.0 | tee -a "$LOG"
    break
  }
  sleep 1
done

grep -aE 'REFINING|ERROR|WAIT|fruit lock|saved entry' log/real_robot/reach_fsm.log | tail -40 | tee -a "$LOG"
timeout 3 ros2 topic echo /feedback/joint_states --once 2>&1 | sed -n '/^position:/,/^velocity:/p' | tee -a "$LOG"
log DONE
exit 1
