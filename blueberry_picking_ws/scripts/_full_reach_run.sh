#!/usr/bin/env bash
# Full LOCK → ALIGN (agent file) → REFINING monitor. Run outside sandbox.
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0 PYTHONNOUSERSITE=1 HOME=/home/user
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
set +u

LOG=log/real_robot/_full_reach_run.txt
: > "$LOG"
: > log/real_robot/reach_fsm.log
# FSM may need restart to clear ERROR and pick up empty log — keep process, just abort+start
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1
sleep 1

timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -2
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -2

timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: start}" 2>&1 | tee -a "$LOG" | tail -3

SESS=""
for i in $(seq 1 40); do
  cand=$(ls -1dt log/real_robot/qa/*/ 2>/dev/null | head -1)
  if [[ -f "${cand}lock_region_request.json" ]]; then
    # Prefer newest session created in last minute
    SESS="$cand"
    echo "LOCK_REQ $SESS t=$i" | tee -a "$LOG"
    break
  fi
  sleep 1
done
if [[ -z "$SESS" ]]; then
  echo 'FAIL no lock request' | tee -a "$LOG"
  strings log/real_robot/reach_fsm.log | tail -40 | tee -a "$LOG"
  exit 1
fi

python3 scripts/write_lock_region_decision.py --session-dir "$SESS" --index 0 \
  --reason 'desktop plant cluster in fixed mono FOV' | tee -a "$LOG"

# Wait ALIGN request
for i in $(seq 1 30); do
  if ls "${SESS}"align_*_request.json >/dev/null 2>&1; then
    echo "ALIGN_REQ t=$i" | tee -a "$LOG"
    break
  fi
  sleep 1
done

# ALIGN loop: up to 6 steps of set_joints from observation, then coarse_ok when fine_visible
for step_try in $(seq 0 7); do
  sleep 1
  if strings log/real_robot/reach_fsm.log | grep -q 'state → REFINING'; then
    echo 'ENTERED_REFINING' | tee -a "$LOG"
    break
  fi
  if strings log/real_robot/reach_fsm.log | grep -q 'state → ERROR'; then
    echo 'ERROR during align' | tee -a "$LOG"
    strings log/real_robot/reach_fsm.log | tail -30 | tee -a "$LOG"
    exit 1
  fi

  # Find pending phase from latest request / fsm log
  REQ=$(ls -1 "${SESS}"align_*_request.json 2>/dev/null | sort | tail -1)
  JUDGE=$(ls -1 "${SESS}"align_*_judge.json 2>/dev/null | sort | tail -1)
  if strings log/real_robot/reach_fsm.log | grep -q 'phase=judge'; then
    STEP=$(python3 -c "import json,glob; p=sorted(glob.glob('${SESS}align_*_judge.json')+glob.glob('${SESS}align_*_request.json'))[-1]; print(json.load(open(p)).get('step_idx',0))")
    [[ -f "${SESS}align_decision.json" ]] && continue
    python3 scripts/apply_align_decision_from_request.py \
      --session-dir "$SESS" --step-idx "$STEP" --phase judge | tee -a "$LOG"
    for j in $(seq 1 20); do
      strings log/real_robot/reach_fsm.log | grep -q 'state → REFINING\|phase=command' && break
      sleep 1
    done
    if strings log/real_robot/reach_fsm.log | grep -q 'state → REFINING'; then
      break
    fi
  fi

  if strings log/real_robot/reach_fsm.log | grep -q 'phase=command'; then
    REQ=$(ls -1 "${SESS}"align_*_request.json 2>/dev/null | sort | tail -1)
    [[ -z "$REQ" ]] && continue
    [[ -f "${SESS}align_decision.json" ]] && continue
    STEP=$(python3 -c "import json; print(json.load(open('$REQ')).get('step_idx',0))")
    python3 scripts/apply_align_decision_from_request.py \
      --session-dir "$SESS" --step-idx "$STEP" --phase command | tee -a "$LOG"
    # wait hold/judge
    for j in $(seq 1 50); do
      strings log/real_robot/reach_fsm.log | grep -q 'phase=judge\|hold after\|ERROR\|REFINING' && break
      sleep 1
    done
  fi
done

echo '=== monitor REFINING ===' | tee -a "$LOG"
for i in $(seq 1 150); do
  if strings log/real_robot/reach_fsm.log | grep -qE 'WAIT_CONFIRM|state → ERROR'; then
    echo "END t=$i" | tee -a "$LOG"
    break
  fi
  if (( i % 5 == 0 )); then
    strings log/real_robot/reach_fsm.log | tail -1 | tee -a "$LOG"
  fi
  sleep 1
done

echo '=== summary ===' | tee -a "$LOG"
strings log/real_robot/reach_fsm.log | grep -E 'LOCK_REGION|ALIGNING|REFINING|servo\[|contact|WAIT_CONFIRM|ERROR|fb_j=' | tee -a "$LOG" | tail -80
echo '=== joints ===' | tee -a "$LOG"
timeout 3 ros2 topic echo /feedback/joint_states --once 2>&1 | sed -n '/^position:/,/^velocity:/p' | head -10 | tee -a "$LOG"
echo DONE | tee -a "$LOG"
