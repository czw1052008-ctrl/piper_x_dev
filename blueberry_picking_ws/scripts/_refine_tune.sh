#!/usr/bin/env bash
# REFINING tune: restore entry → start_refine → monitor.
# - FAIL: diagnose first; auto-restore only when root cause is clear; else ask user.
# - REFINING_WAIT_NEAR: pause for confirm_near (no auto-confirm)
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
LOG=log/real_robot/_refine_tune.txt
LOCK_VIZ=log/real_robot/refine_tune_viz/latest_lock_sidebyside.png
FAIL_DIR=log/real_robot/refine_tune_fail
mkdir -p "$FAIL_DIR"
: > "$LOG"
log() { echo "$*" | tee -a "$LOG"; }

restore_entry() {
  log '=== RESTORE refine entry pose ==='
  timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tail -1 | tee -a "$LOG"
  timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tail -1 | tee -a "$LOG"
  python3 scripts/refine_entry_pose.py restore --traj-s 5.0 --settle-s 2.0 | tee -a "$LOG"
}

# Returns 0 if failure cause is clear enough to auto-restore; writes report to $1
classify_failure() {
  local report="$1" reason="$2"
  local verdict=unclear summary=''

  if echo "$reason" | grep -qE 'refine/servo timeout|no fruit lock at REFINING|servo step timeout|servo traj rejected|refine: no fruit lock'; then
    verdict=clear
    summary='REFINING 超时或腕部检测/轨迹失败 — 可从入口位姿重试'
  elif echo "$reason" | grep -qE 'align loop|LOCKING|ALIGNING'; then
    verdict=clear
    summary='非 REFINING 阶段错误 — 建议 restore 后重跑'
  elif grep -aq 'joint delta tiny' log/real_robot/reach_fsm.log 2>/dev/null \
      && ! echo "$reason" | grep -q ERROR; then
    verdict=unclear
    summary='伺服步进被 joint-deadband 卡住（未进 ERROR）— 需看图确认是否到位/限位'
  else
    verdict=unclear
    summary='根因未明 — 请结合 viz + 日志判断后再决定是否 restore'
  fi

  {
    echo "=== REFINING failure report $(date -Iseconds) ==="
    echo "status/reason: $reason"
    echo "verdict: $verdict"
    echo "summary: $summary"
    echo ""
    echo "--- last REFINING/ERROR log ---"
    grep -aE 'REFINING|ERROR|servo|fruit lock|WAIT_NEAR' log/real_robot/reach_fsm.log | tail -40
    echo ""
    echo "--- joints ---"
    timeout 3 ros2 topic echo /feedback/joint_states --once 2>/dev/null | sed -n '/^position:/,/^velocity:/p'
    echo ""
    echo "--- lock viz ---"
    echo "latest: $LOCK_VIZ"
    ls -1dt log/real_robot/qa/*/refine_fruit_lock_sidebyside.png 2>/dev/null | head -3
    ls -1dt log/real_robot/qa/*/near_handoff_pending_sidebyside.png 2>/dev/null | head -2
    echo ""
    echo "--- saved entry pose ---"
    cat "$POSE" 2>/dev/null
  } > "$report"
  echo "$verdict|$summary"
}

handle_failure() {
  local reason="$1"
  local tag
  tag=$(date +%Y%m%d_%H%M%S)
  local report="$FAIL_DIR/fail_${tag}.txt"
  local cls
  cls=$(classify_failure "$report" "$reason")
  local verdict="${cls%%|*}"
  local summary="${cls#*|}"

  log ''
  log "======== FAIL: $reason ========"
  log "诊断报告: $report"
  log "结论: $summary"
  [[ -f "$LOCK_VIZ" ]] && log "锁定目标 viz: $LOCK_VIZ"

  timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1
  sleep 1

  if [[ "$verdict" == clear ]]; then
    log "根因明确 → 自动 restore 入口位姿"
    restore_entry
    exit 1
  fi

  log ''
  log '根因未完全明确 — 臂保持当前位姿，请查看报告和 viz 后决定:'
  log "  cat $report"
  if [[ -t 0 ]]; then
    read -r -p '是否 restore 到 ALIGN 入口位姿? [y/N] ' ans
    if [[ "$ans" == [yY] ]]; then
      restore_entry
    else
      log '跳过 restore — 臂保持失败时位姿供检查'
    fi
  else
    log '非交互终端: 不自动 restore。确认后手动: python3 scripts/refine_entry_pose.py restore'
  fi
  exit 2
}

[[ -f "$POSE" ]] || { log "missing $POSE — run _align_calibrate_only.sh first"; exit 1; }

# Restart FSM if missing or stale (no lock-viz support in old proc)
if ! pgrep -f 'python3 scripts/reach_fsm_node.py' >/dev/null; then
  log 'FSM not running — starting _retest_fsm.sh'
  bash scripts/_retest_fsm.sh
  sleep 3
else
  log 'restarting FSM for latest REFINING snap/viz code'
  bash scripts/_retest_fsm.sh
  sleep 3
fi

if ! pgrep -f 'fine_detector_node' >/dev/null; then
  log 'starting fine_detector_node'
  nohup ros2 run picking_perception fine_detector_node --ros-args \
    -p enable_foundation_pose:=false -p publish_hz:=10.0 -p mask_source:=yolo \
    --params-file src/picking_perception/config/foundation_pose.yaml \
    >log/real_robot/fine_detector.log 2>&1 &
  sleep 3
fi

/usr/bin/python3 - <<'PY' >/dev/null 2>&1
import rclpy, time
from std_msgs.msg import String
rclpy.init()
from rclpy.node import Node
n = Node('abort_retry')
pub = n.create_publisher(String, '/reach/cmd', 10)
time.sleep(0.4)
for _ in range(4):
    pub.publish(String(data='abort'))
    rclpy.spin_once(n, timeout_sec=0.05)
    time.sleep(0.15)
n.destroy_node()
rclpy.shutdown()
PY
sleep 1
restore_entry || exit 5

log '=== START_REFINE ==='
LOG_FS=log/real_robot/reach_fsm.log
LOG_START=$(wc -l < "$LOG_FS" 2>/dev/null || echo 0)
/usr/bin/python3 - <<'PY' 2>&1 | tee -a "$LOG"
import rclpy, time
from std_msgs.msg import String
rclpy.init()
from rclpy.node import Node
n = Node('start_refine_retry')
pub = n.create_publisher(String, '/reach/cmd', 10)
time.sleep(0.6)
for i in range(5):
    pub.publish(String(data='start_refine'))
    print(f'published start_refine #{i + 1}')
    rclpy.spin_once(n, timeout_sec=0.05)
    time.sleep(0.2)
n.destroy_node()
rclpy.shutdown()
PY

NEAR_PAUSED=0
STUCK_REFINING=0
for i in $(seq 1 240); do
  st=$(timeout 2 ros2 topic echo /reach/status --once 2>/dev/null | awk -F"'" '/data:/{print $2; exit}')
  [[ -n "$st" ]] && log "t=$i status=$st"

  if [[ -f "$LOCK_VIZ" ]]; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOCK_VIZ") ))
    (( age < 4 )) && log "lock viz → $LOCK_VIZ"
  fi

  # Stuck in REFINING without motion (joint delta tiny spam)
  if [[ "$st" == REFINING ]]; then
    tiny=$(grep -ac 'joint delta tiny' log/real_robot/reach_fsm.log 2>/dev/null || echo 0)
    if (( tiny > 30 && STUCK_REFINING == 0 )); then
      recent=$(grep -ac 'joint delta tiny' <(tail -40 log/real_robot/reach_fsm.log) 2>/dev/null || echo 0)
      if (( recent > 15 )); then
        STUCK_REFINING=1
        handle_failure "REFINING stuck: repeated 'joint delta tiny' (servo deadband/limits?)"
      fi
    fi
  fi

  if [[ "$st" == REFINING_WAIT_NEAR && "$NEAR_PAUSED" == 0 ]]; then
    NEAR_PAUSED=1
    sleep 1
    log ''
    log '======== PAUSED: REFINING_WAIT_NEAR ========'
    log "锁定目标可视化: $LOCK_VIZ"
    ls -1dt log/real_robot/qa/*/near_handoff_pending_sidebyside.png 2>/dev/null | head -1 | tee -a "$LOG"
    log '确认 OK 后: ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: confirm_near}"'
    log '=========================================='
  fi

  [[ "$st" == WAIT_CONFIRM ]] && { log SUCCESS; exit 0; }

  if [[ "$st" == ERROR* ]]; then
    handle_failure "$st"
  fi
  # ros2 topic echo can miss fast ERROR transitions — watch new FSM log lines only
  if tail -n +$((LOG_START + 1)) "$LOG_FS" 2>/dev/null | grep -aqE \
      'ERROR|refine/servo timeout|refine: no fruit lock'; then
    err=$(tail -n +$((LOG_START + 1)) "$LOG_FS" 2>/dev/null \
      | grep -aE 'ERROR|refine/servo timeout|refine: no fruit lock' | tail -1 \
      | sed 's/.*reach_fsm_node]: //')
    handle_failure "${err:-ERROR from reach_fsm.log}"
  fi
  sleep 1
done

handle_failure "TIMEOUT after 240s (no WAIT_CONFIRM)"
