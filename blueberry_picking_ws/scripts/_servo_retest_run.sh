#!/usr/bin/env bash
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0 PYTHONNOUSERSITE=1 HOME=/home/user
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
set +u

LOG=log/real_robot/_servo_retest.txt
: > "$LOG"
# Isolate this run's FSM log from prior ERROR lines.
: > log/real_robot/reach_fsm.log
# give FSM a moment to reprint ready (node still running)
sleep 1

timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -2
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -2
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: start}" 2>&1 | tee -a "$LOG" | tail -3

# Wait lock request
SESS=""
for i in $(seq 1 30); do
  cand=$(ls -1dt log/real_robot/qa/*/ 2>/dev/null | head -1)
  if [[ -f "${cand}lock_region_request.json" ]]; then
    # prefer newest session modified after start
    SESS="$cand"
    echo "LOCK_REQ SESS=$SESS t=$i" | tee -a "$LOG"
    break
  fi
  sleep 1
done
if [[ -z "$SESS" ]]; then
  echo 'FAIL no lock request' | tee -a "$LOG"
  tail -30 log/real_robot/reach_fsm.log | tee -a "$LOG"
  exit 1
fi

python3 scripts/write_lock_region_decision.py --session-dir "$SESS" --index 0 \
  --reason 'servo retest region' | tee -a "$LOG"

# Wait ALIGNING request
for i in $(seq 1 20); do
  if ls "${SESS}"align_*_request.json >/dev/null 2>&1; then
    echo "ALIGN_REQ t=$i" | tee -a "$LOG"
    break
  fi
  sleep 1
done
REQ=$(ls -1 "${SESS}"align_*_request.json 2>/dev/null | sort | tail -1)
[[ -z "$REQ" ]] && { echo 'FAIL no align request' | tee -a "$LOG"; exit 1; }
STEP=$(python3 -c "import json; print(json.load(open('$REQ'))['step_idx'])")
PHASE=$(python3 -c "import json; print(json.load(open('$REQ')).get('phase','command'))")
echo "REQ=$REQ STEP=$STEP PHASE=$PHASE" | tee -a "$LOG"

python3 scripts/apply_align_decision_from_request.py \
  --session-dir "$SESS" --step-idx "$STEP" --phase "$PHASE" | tee -a "$LOG"

# If command phase rejects coarse_ok, handle in monitor
for i in $(seq 1 120); do
  if grep -q 'state → REFINING' log/real_robot/reach_fsm.log; then
    echo "ENTER_REFINE t=$i" | tee -a "$LOG"
    break
  fi
  if grep -q 'unsupported align action\|state → ERROR' log/real_robot/reach_fsm.log; then
    # maybe need set_joints then judge — if still in command waiting, skip
    if grep -q 'phase=judge' log/real_robot/reach_fsm.log; then
      python3 scripts/write_align_decision.py --session-dir "$SESS" --step-idx "$STEP" \
        --phase judge --action coarse_ok --reason 'judge ok for servo retest' | tee -a "$LOG"
    fi
  fi
  # If waiting command after rejecting coarse_ok:
  if grep -q "unsupported align action: 'coarse_ok'" log/real_robot/reach_fsm.log 2>/dev/null; then
    :
  fi
  sleep 1
done

# If still ALIGNING command, use observation-based set_joints then judge
if ! grep -q 'state → REFINING' log/real_robot/reach_fsm.log; then
  echo 'still aligning — decide_action from request then judge' | tee -a "$LOG"
  python3 scripts/apply_align_decision_from_request.py \
    --session-dir "$SESS" --step-idx "$STEP" --phase command | tee -a "$LOG"
  for i in $(seq 1 40); do
    if grep -q 'phase=judge\|hold after' log/real_robot/reach_fsm.log; then
      python3 scripts/apply_align_decision_from_request.py \
        --session-dir "$SESS" --step-idx "$STEP" --phase judge | tee -a "$LOG"
      break
    fi
    sleep 1
  done
fi

# Monitor servo
for i in $(seq 1 120); do
  if grep -qE 'WAIT_CONFIRM|state → ERROR' log/real_robot/reach_fsm.log; then
    echo "END t=$i" | tee -a "$LOG"
    break
  fi
  if (( i % 5 == 0 )); then
    tail -1 log/real_robot/reach_fsm.log | tee -a "$LOG"
  fi
  sleep 1
done

echo '=== fsm tail ===' | tee -a "$LOG"
rg -n 'LOCK_REGION|ALIGNING|REFINING|servo|contact|WAIT_CONFIRM|ERROR|orient_frac' log/real_robot/reach_fsm.log | tee -a "$LOG" | tail -60

echo '=== joints ===' | tee -a "$LOG"
timeout 3 ros2 topic echo /feedback/joint_states --once 2>&1 | rg -A8 '^position:' | tee -a "$LOG"
echo DONE | tee -a "$LOG"
