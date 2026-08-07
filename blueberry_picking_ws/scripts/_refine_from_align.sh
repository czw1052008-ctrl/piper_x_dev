#!/usr/bin/env bash
# Debug REFINING only — arm stays at ALIGN coarse_ok pose (no home reset).
# Prereq: FSM running (_retest_fsm.sh), fine_detector, arm enabled.
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0 PYTHONNOUSERSITE=1 HOME=/home/user
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
set +u

LOG=log/real_robot/_refine_from_align.txt
POSE=log/real_robot/refine_entry_pose.json
LOCK_VIZ=log/real_robot/refine_tune_viz/latest_lock_sidebyside.png
: > "$LOG"
log() { echo "$*" | tee -a "$LOG"; }

restore_entry() {
  python3 scripts/refine_entry_pose.py restore --traj-s 5.0 --settle-s 2.0 | tee -a "$LOG"
}

[[ -f "$POSE" ]] || { log "missing $POSE"; exit 1; }

timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -1
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tee -a "$LOG" | tail -1

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

log "=== joints before start_refine ==="
timeout 3 ros2 topic echo /feedback/joint_states --once 2>&1 | sed -n '/^position:/,/^velocity:/p' | tee -a "$LOG"

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

log '=== monitor REFINING (confirm_near pauses before near handoff) ==='
for i in $(seq 1 180); do
  st=$(timeout 2 ros2 topic echo /reach/status --once 2>/dev/null | awk '/data:/{print $2; exit}')
  [[ -n "$st" ]] && log "t=$i status=$st"
  echo "$st" | grep -q 'REFINING_WAIT_NEAR' && {
    log 'PAUSED at REFINING_WAIT_NEAR'
    log "锁定目标可视化: $LOCK_VIZ"
    log '确认后执行: 重复发布 confirm_near（不要只发一次）'
    break
  }
  echo "$st" | grep -q 'WAIT_CONFIRM' && { log SUCCESS; exit 0; }
  echo "$st" | grep -q 'ERROR' && {
    log ERROR
    tail -20 log/real_robot/reach_fsm.log | tee -a "$LOG"
    timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" >/dev/null 2>&1
    restore_entry
    exit 1
  }
  sleep 1
done

tail -30 log/real_robot/reach_fsm.log | tee -a "$LOG"
log DONE
