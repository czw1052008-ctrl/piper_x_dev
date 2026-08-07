#!/usr/bin/env bash
# Half REFINING: center → depth QA stop (YOLO or probe-tri reproject).
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

LOG=log/real_robot/_depth_qa_half.txt
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

log '=== restart FSM ==='
bash scripts/_retest_fsm.sh 2>&1 | tee -a "$LOG"
sleep 3

log '=== restore entry + enable ==='
/usr/bin/python3 scripts/refine_entry_pose.py restore 2>&1 | tee -a "$LOG"
timeout 5 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tail -1 | tee -a "$LOG"
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tail -1 | tee -a "$LOG"

log '=== fine berries hz ==='
timeout 4 ros2 topic hz /perception/fine/berries 2>&1 | head -3 | tee -a "$LOG"

log '=== reset + start_refine ==='
/usr/bin/python3 - <<'PY'
import rclpy
from std_msgs.msg import String
import time
rclpy.init()
from rclpy.node import Node
n = Node('cmd_pub_once')
pub = n.create_publisher(String, '/reach/cmd', 10)
time.sleep(0.8)
pub.publish(String(data='confirm_reset'))
time.sleep(1.0)
pub.publish(String(data='start_refine'))
print('published confirm_reset + start_refine')
time.sleep(0.5)
n.destroy_node()
rclpy.shutdown()
PY

for i in $(seq 1 60); do
  if grep -qE 'depth QA stop|state → WAIT_CONFIRM|state → ERROR' log/real_robot/reach_fsm.log; then
    log "done_i=$i"
    break
  fi
  sleep 2
done

log '=== KEY LOG ==='
grep -E 'fruit lock|mono probe|center openloop|fresh optical|range_depth|depth QA|ERROR \(' \
  log/real_robot/reach_fsm.log | grep -v 'waiting for fine' | tail -40 | tee -a "$LOG"

SESS=$(ls -1dt log/real_robot/qa/*/ | head -1)
log "SESS=$SESS"
if [[ -f "${SESS}center_depth_qa.json" ]]; then
  cat "${SESS}center_depth_qa.json" | tee -a "$LOG"
fi

/usr/bin/python3 scripts/refine_replay.py --qa-dir "$SESS" 2>&1 | tee -a "$LOG" | tail -5
/usr/bin/python3 scripts/refine_entry_pose.py restore 2>&1 | tail -2 | tee -a "$LOG"
