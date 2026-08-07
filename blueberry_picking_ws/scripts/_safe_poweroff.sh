#!/usr/bin/env bash
# Safe power-off sequence: home arm → disable → kill stack.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:${HOME}/.local/bin:${PATH}
export HOME=/home/user RMW_FASTRTPS_USE_SHM=0 PYTHONNOUSERSITE=1
export ROS_LOG_DIR="${ROOT}/log/real_robot/ros_logs"
mkdir -p log/real_robot

set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
set +u

echo "[poweroff] abort + confirm_reset"
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: abort}" || true
sleep 0.5
timeout 3 ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: confirm_reset}" || true
timeout 5 ros2 service call /move_home std_srvs/srv/Trigger "{}" || true

echo "[poweroff] wait joints toward home (up to 15s)"
for i in $(seq 1 15); do
  if timeout 2 ros2 topic echo /feedback/joint_states --once 2>/dev/null | python3 -c '
import sys,math
t=sys.stdin.read()
# crude parse first 6 floats after position:
import re
m=re.search(r"position:\n((?:\s*-\s*-?[0-9.]+\n){1,6})", t)
if not m: raise SystemExit(1)
vals=[float(x) for x in re.findall(r"-?\d+\.?\d*", m.group(1))][:6]
ok=all(abs(v)<0.15 for v in vals)
print("t=%d deg="%i, [round(v*57.3,1) for v in vals], "HOME" if ok else "")
raise SystemExit(0 if ok else 2)
' 2>/dev/null; then
    echo "[poweroff] near home"
    break
  fi
  sleep 1
done

echo "[poweroff] disable arm"
timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: false}" || true
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: false}" || true
sleep 1

echo "[poweroff] real_robot_shutdown"
bash scripts/real_robot_shutdown.sh --quiet || true
sleep 1

# Force leftover nodes
for pat in \
  'scripts/reach_fsm_node.py' \
  'scripts/align_vlm_decider.py' \
  'agx_arm_ctrl_single' \
  'start_single_agx_arm_moveit' \
  'orbbec_camera dabai' \
  'v4l2_camera_node' \
  'global_detector_node' \
  'fine_detector_node' \
  'move_group' \
  'controller_manager' \
  'component_container' \
  'fixed_camera.launch'
do
  pkill -9 -f "$pat" 2>/dev/null || true
done

sleep 1
echo "[poweroff] remaining:"
pgrep -af 'agx_arm_ctrl|orbbec_camera|v4l2_camera|reach_fsm|global_detector|fine_detector|move_group' || echo cleared
echo "[poweroff] DONE — safe to power off PC / arm supply"
