#!/usr/bin/env bash
# Unattended reach: LOCK → ALIGN (observation-based heuristic) → monitor REFINING.
# ALIGN uses apply_align_decision_from_request.py (align_judge.decide_action) — no fixed joints.
# For human/agent-in-the-loop, use align_vlm_decider provider=agent instead.
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0 PYTHONNOUSERSITE=1 HOME=/home/user
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
set +u

LOG=log/real_robot/_reach_until_contact.txt
: > "$LOG"
log() { echo "$*" | tee -a "$LOG"; }
fsm_has() { grep -aEq "$1" log/real_robot/reach_fsm.log 2>/dev/null; }
fsm_tail() { grep -aE "$1" log/real_robot/reach_fsm.log 2>/dev/null | tail -"${2:-1}"; }
log_mark() { wc -c < log/real_robot/reach_fsm.log; }
wait_new() {
  local pat="$1" mark="$2" n="${3:-60}"
  for i in $(seq 1 "$n"); do
    if (( $(wc -c < log/real_robot/reach_fsm.log) > mark )) && fsm_has "$pat"; then
      local hit
      hit=$(grep -aEn "$pat" log/real_robot/reach_fsm.log | tail -1 | cut -d: -f1)
      if [[ -n "$hit" ]]; then
        local bytes
        bytes=$(head -n "$hit" log/real_robot/reach_fsm.log | wc -c)
        if (( bytes > mark )); then return 0; fi
      fi
    fi
    sleep 1
  done
  return 1
}

align_phase() {
  fsm_tail 'waiting for decision file phase=' 1 | sed -n 's/.*phase=\([a-z]*\).*/\1/p'
}

align_step() {
  local sess="$1"
  python3 - <<PY
import json, glob, os
sess = os.environ['SESS']
files = sorted(
    glob.glob(sess + 'align_*_judge.json')
    + glob.glob(sess + 'align_*_after_*.json')
    + glob.glob(sess + 'align_*_request.json'),
    key=os.path.getmtime,
)
if not files:
    print(0)
else:
    print(int(json.load(open(files[-1])).get('step_idx', 0)))
PY
}

write_align_from_obs() {
  local sess="$1" step="$2" phase="$3"
  SESS="$sess" python3 scripts/apply_align_decision_from_request.py \
    --session-dir "$sess" --step-idx "$step" --phase "$phase" \
    | tee -a "$LOG"
}

timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1
sleep 1
timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -1
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -1

MARK=$(log_mark)
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: start}" 2>&1 | tee -a "$LOG" | tail -2

SESS=""
for i in $(seq 1 50); do
  for d in $(ls -1dt log/real_robot/qa/*/ 2>/dev/null | head -6); do
    [[ -f "${d}lock_region_request.json" ]] || continue
    age=$(( $(date +%s) - $(stat -c %Y "${d}lock_region_request.json") ))
    if (( age < 90 )); then SESS="$d"; break; fi
  done
  [[ -n "$SESS" ]] && break
  sleep 1
done
log "SESS=$SESS"
[[ -z "$SESS" ]] && { log FAIL_NO_LOCK; exit 1; }
python3 scripts/write_lock_region_decision.py --session-dir "$SESS" --index 0 \
  --reason 'plant in workspace' | tee -a "$LOG"

export SESS
for round in $(seq 1 12); do
  if fsm_has 'state → REFINING'; then log ENTER_REFINE; break; fi
  if fsm_has 'state → ERROR'; then log ALIGN_ERROR; fsm_tail 'ERROR' 10 | tee -a "$LOG"; exit 2; fi

  PHASE=$(align_phase)
  [[ -z "$PHASE" ]] && { sleep 1; continue; }
  if [[ -f "${SESS}align_decision.json" ]]; then
    EXISTING_PHASE=$(python3 -c "import json; print(json.load(open('${SESS}align_decision.json')).get('phase',''))" 2>/dev/null)
    if [[ "$EXISTING_PHASE" == "$PHASE" ]]; then
      sleep 1
      continue
    fi
    rm -f "${SESS}align_decision.json"
  fi

  STEP=$(align_step "$SESS")
  log "round=$round phase=$PHASE step=$STEP (observation-based decide_action)"
  MARK=$(log_mark)
  write_align_from_obs "$SESS" "$STEP" "$PHASE" || { log FAIL_ALIGN_DECISION; exit 2; }
  wait_new 'hold after|phase=judge|ERROR|REFINING' "$MARK" 55 || true
done

if ! fsm_has 'state → REFINING'; then
  log FAIL_NO_REFINE
  fsm_tail 'LOCK|ALIGN|ERROR|waiting|REFINING' 40 | tee -a "$LOG"
  exit 3
fi

log '=== REFINING (reach-first) ==='
OK=0
for i in $(seq 1 180); do
  if fsm_has 'WAIT_CONFIRM'; then log "SUCCESS_CONTACT t=$i"; OK=1; break; fi
  if fsm_has 'state → ERROR'; then log "ERROR_t=$i"; break; fi
  if (( i % 3 == 0 )); then
    fsm_tail 'servo\[|contact|anchors|ERROR|WAIT|waiting for fine' 1 | tee -a "$LOG"
  fi
  sleep 1
done
log '=== final ==='
fsm_tail 'anchors|servo\[|contact|WAIT_CONFIRM|ERROR|fb_j=' 50 | tee -a "$LOG"
timeout 2 ros2 topic echo /feedback/joint_states --once 2>&1 | sed -n '/^position:/,/^velocity:/p' | head -10 | tee -a "$LOG"
log "DONE ok=$OK"
exit $(( 1 - OK ))
