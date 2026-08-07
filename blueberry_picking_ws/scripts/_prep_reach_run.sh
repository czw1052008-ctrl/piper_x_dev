#!/usr/bin/env bash
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
export PYTHONNOUSERSITE=1
export HOME=/home/user
export ROS_LOG_DIR=$PWD/log/real_robot/ros_logs
export LD_LIBRARY_PATH=/home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/orbbec_camera/lib:${LD_LIBRARY_PATH:-}
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
source /home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash
set +u
[[ -f config/real_robot.env ]] && set +u && source config/real_robot.env && set +u

echo '=== restart fixed cam ==='
pkill -f 'v4l2_camera_node.*camera_fixed' 2>/dev/null
# do not pkill static_transform broadly — only exact child frame
ps -eo pid,args | while read -r pid args; do
  case "$args" in
    *static_transform_publisher*camera_fixed_optical_frame*) kill "$pid" 2>/dev/null; echo killed_tf "$pid";;
  esac
done
sleep 2
v4l2-ctl -d "${FIXED_CAMERA_DEVICE:-/dev/video0}" --set-parm=10 2>/dev/null || true
: > log/real_robot/fixed_cam.log
nohup ros2 launch picking_perception fixed_camera.launch.py \
  video_device:="${FIXED_CAMERA_DEVICE:-/dev/video0}" \
  camera_name:=camera_fixed \
  frame_id:=camera_fixed_optical_frame \
  tx:="${FIXED_CAM_TX:-0.60}" ty:="${FIXED_CAM_TY:-0.33}" tz:="${FIXED_CAM_TZ:-0.48}" \
  qx:="${FIXED_CAM_QX:-0}" qy:="${FIXED_CAM_QY:-0}" qz:="${FIXED_CAM_QZ:-0}" qw:="${FIXED_CAM_QW:-1}" \
  > log/real_robot/fixed_cam.log 2>&1 &
echo FIXED_PID=$!
sleep 5
timeout 6 ros2 topic hz /camera_fixed/image_raw 2>&1 | head -10

echo '=== stack ==='
ps -eo pid,args | grep -E 'reach_fsm_node|global_detector_node|fine_detector_node|v4l2_camera' | grep -v grep

# restart FSM with calib params if needed
if ! ps -eo args | grep -q '[p]ython3 scripts/reach_fsm_node.py'; then
  bash scripts/_retest_fsm.sh
fi

timeout 8 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | tail -2
timeout 5 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>&1 | tail -2
timeout 3 ros2 topic echo /feedback/joint_states --once 2>&1 | sed -n '/^position:/,/^velocity:/p' | head -10
timeout 3 ros2 topic echo /perception/global/berries --once 2>&1 | head -20
timeout 3 ros2 topic echo /perception/fine/berries --once 2>&1 | head -20
echo PREP_DONE
