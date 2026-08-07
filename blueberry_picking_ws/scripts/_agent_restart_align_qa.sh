#!/usr/bin/env bash
# One-shot: recycle real-robot stack + run file/agent position-control ALIGN QA.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:${HOME}/.local/bin:${PATH}
export HOME=/home/user
export RMW_FASTRTPS_USE_SHM=0
export PYTHONNOUSERSITE=1
export ROS_LOG_DIR="${ROOT}/log/real_robot/ros_logs"
export ALIGN_JUDGE_MODE=file
export ALIGN_VLM_PROVIDER=agent
mkdir -p "${ROS_LOG_DIR}" log/real_robot/qa

echo "[agent_qa] killing old stack patterns..."
for pat in \
  'scripts/reach_fsm_node.py' \
  'scripts/align_vlm_decider.py' \
  'scripts/run_align_qa.sh' \
  'ros2 service call /enable_agx_arm' \
  'agx_arm_ctrl_single' \
  'start_single_agx_arm_moveit.launch.py' \
  'orbbec_camera dabai.launch.py' \
  'v4l2_camera_node' \
  'global_detector_node' \
  'fine_detector_node' \
  'move_group' \
  'component_container' \
  'controller_manager' \
  'fixed_camera.launch.py'
do
  pkill -9 -f "$pat" 2>/dev/null || true
done
sleep 2
rm -f log/real_robot/stack.pids

echo "[agent_qa] ensuring can0 up..."
if ! ip link show can0 >/dev/null 2>&1; then
  echo "[agent_qa] ERROR: can0 missing" >&2
  exit 2
fi
ip link set can0 up 2>/dev/null || true
ip -brief link show can0

echo "[agent_qa] bringup..."
bash scripts/real_robot_bringup.sh --fixed-cam --reach-perception --no-wait \
  | tee log/real_robot/_bringup_agent_qa.txt

echo "[agent_qa] wait for joints..."
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
source /home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash 2>/dev/null || true
export LD_LIBRARY_PATH="/home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/orbbec_camera/lib:${LD_LIBRARY_PATH:-}"
set +u

ok=0
for i in $(seq 1 45); do
  if timeout 2 ros2 topic echo /feedback/joint_states --once 2>/dev/null | grep -q position; then
    echo "[agent_qa] joints OK t=$i"
    ok=1
    break
  fi
  echo "[agent_qa] wait joints t=$i"
  sleep 1
done
if [[ "$ok" != 1 ]]; then
  echo "[agent_qa] ERROR: no joint feedback" >&2
  tail -40 log/real_robot/arm.log 2>/dev/null || true
  exit 3
fi

timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}"
timeout 8 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}"

echo "[agent_qa] starting run_align_qa (file+agent) ..."
: > log/real_robot/_align_qa_agent_pos.txt
# Run in background so this script can return the session path quickly for agent decisions.
nohup bash scripts/run_align_qa.sh > log/real_robot/_align_qa_agent_pos.txt 2>&1 &
echo "QA_WRAPPER_PID=$!"
sleep 10
tail -30 log/real_robot/_align_qa_agent_pos.txt || true
tail -20 log/real_robot/reach_fsm.log || true
ls -1dt log/real_robot/qa/*/ | head -5
echo "[agent_qa] waiting for align_*_request.json ..."
for i in $(seq 1 60); do
  sess=$(ls -1dt log/real_robot/qa/[0-9][0-9][0-9][0-9]* 2>/dev/null | head -1 || true)
  if [[ -n "$sess" ]] && ls "$sess"/align_*_request.json >/dev/null 2>&1; then
    echo "SESSION=$sess"
    ls -1 "$sess"/align_*_request* 2>/dev/null | head -20
    exit 0
  fi
  sleep 1
done
echo "[agent_qa] WARN: no request yet"
tail -40 log/real_robot/reach_fsm.log || true
exit 4
