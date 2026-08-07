#!/usr/bin/env bash
# Full REFINING: mono_probe → center (+live-replan) → confirm_near → oneshot to berry (offset=0).
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HOME=/home/user
export LD_LIBRARY_PATH=/home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/orbbec_camera/lib:${LD_LIBRARY_PATH:-}
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
source /home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash
set +u

LOG=log/real_robot/_refine_contact.txt
: > "$LOG"
log() { echo "$*" | tee -a "$LOG"; }

if ! pgrep -f 'fine_detector_node' >/dev/null; then
  log 'starting fine_detector_node'
  nohup /usr/bin/python3 -u install/picking_perception/lib/picking_perception/fine_detector_node --ros-args \
    -p enable_foundation_pose:=false -p publish_hz:=10.0 -p mask_source:=yolo \
    --params-file src/picking_perception/config/foundation_pose.yaml \
    >>log/real_robot/fine_detector.log 2>&1 &
  sleep 5
fi

log '=== restart FSM (full contact path) ==='
bash scripts/_retest_fsm_contact.sh 2>&1 | tee -a "$LOG"
sleep 3

# Compact bag: joints/TCP/status + camera_info (images already in QA snaps).
BAG_STAMP=$(date +%Y%m%d_%H%M%S)
BAG_DIR="log/real_robot/qa_bags/${BAG_STAMP}"
mkdir -p log/real_robot/qa_bags
log "=== ros2 bag record → $BAG_DIR ==="
nohup ros2 bag record -o "$BAG_DIR" --compression-mode file --compression-format zstd \
  /feedback/joint_states /feedback/tcp_pose /reach/status /reach/cmd \
  /camera_wrist/color/camera_info \
  /tf /tf_static \
  >>log/real_robot/_refine_contact_bag.txt 2>&1 &
BAG_PID=$!
log "BAG_PID=$BAG_PID"
trap 'kill $BAG_PID 2>/dev/null; wait $BAG_PID 2>/dev/null; true' EXIT

log '=== restore entry + enable ==='
/usr/bin/python3 scripts/refine_entry_pose.py restore 2>&1 | tee -a "$LOG"
timeout 5 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tail -1 | tee -a "$LOG"
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tail -1 | tee -a "$LOG"

log '=== start_refine ==='
/usr/bin/python3 - <<'PY'
import rclpy, time
from std_msgs.msg import String
rclpy.init()
from rclpy.node import Node
n = Node('cmd_pub_retry')
pub = n.create_publisher(String, '/reach/cmd', 10)
time.sleep(0.8)
for cmd in ('confirm_reset', 'start_refine'):
    for i in range(5):
        pub.publish(String(data=cmd))
        print(f'published {cmd} #{i + 1}')
        rclpy.spin_once(n, timeout_sec=0.05)
        time.sleep(0.2)
time.sleep(0.5)
n.destroy_node()
rclpy.shutdown()
PY

NEAR_DONE=0
OK=0
for i in $(seq 1 180); do
  if grep -q 'state → WAIT_CONFIRM' log/real_robot/reach_fsm.log 2>/dev/null; then
    log "SUCCESS t=$i"
    OK=1
    break
  fi
  if grep -q 'state → REFINING_WAIT_NEAR' log/real_robot/reach_fsm.log 2>/dev/null \
      && [[ "$NEAR_DONE" == 0 ]]; then
    NEAR_DONE=1
    log "t=$i WAIT_NEAR → confirm_near"
    # Short-lived publishers often miss DDS match; hold publisher + spin longer.
    /usr/bin/python3 - <<'PY' 2>&1 | tee -a "$LOG"
import rclpy, time
from std_msgs.msg import String
rclpy.init()
from rclpy.node import Node
n = Node('confirm_near_retry')
pub = n.create_publisher(String, '/reach/cmd', 10)
for _ in range(30):
    rclpy.spin_once(n, timeout_sec=0.05)
time.sleep(1.0)
for i in range(12):
    pub.publish(String(data='confirm_near'))
    print(f'published confirm_near #{i + 1}')
    rclpy.spin_once(n, timeout_sec=0.05)
    time.sleep(0.25)
time.sleep(0.5)
n.destroy_node()
rclpy.shutdown()
PY
  fi
  if grep -q 'state → ERROR' log/real_robot/reach_fsm.log 2>/dev/null; then
    log "ERROR t=$i"
    break
  fi
  sleep 1
done

log '=== KEY LOG ==='
grep -E 'climb\[|center arrive|center openloop|mono chord|contact from probe|WAIT_NEAR|near oneshot|contact \(cup|WAIT_CONFIRM|ERROR \(|arrive viz' \
  log/real_robot/reach_fsm.log | grep -v 'waiting for fine' | tail -50 | tee -a "$LOG"

kill $BAG_PID 2>/dev/null
wait $BAG_PID 2>/dev/null
trap - EXIT

SESS=$(ls -1dt log/real_robot/qa/*/ | head -1)
log "SESS=$SESS ok=$OK BAG=$BAG_DIR"
# Dump climb JSON summary if present.
for f in "${SESS}"center_climb_*.json; do
  [[ -f "$f" ]] || continue
  log "--- $(basename "$f") ---"
  /usr/bin/python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k:d.get(k) for k in ['phase','planned_dz_mm','tip_above_berry_z_mm','ee_err_mm','z_aim_m','center_mono_z_scale','berry_src','dist_cup_m']})" "$f" | tee -a "$LOG"
done
/usr/bin/python3 scripts/refine_replay.py --qa-dir "$SESS" 2>&1 | tee -a "$LOG" | tail -5
/usr/bin/python3 scripts/refine_entry_pose.py restore 2>&1 | tail -2 | tee -a "$LOG"
# Point bag at session for later analysis.
if [[ -d "$BAG_DIR" ]]; then
  echo "$BAG_DIR" > "${SESS}bag_path.txt"
  log "bag linked → ${SESS}bag_path.txt"
fi
exit $((1 - OK))
