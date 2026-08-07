#!/usr/bin/env bash
# Restart FSM for full REFINING: center → depth → contact oneshot (no stop-after-center).
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
export PYTHONNOUSERSITE=1
export HOME=/home/user
export ROS_LOG_DIR=/home/user/codes/piper_x_dev/blueberry_picking_ws/log/real_robot/ros_logs
export LD_LIBRARY_PATH=/home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/orbbec_camera/lib:${LD_LIBRARY_PATH:-}
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
source /home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash
set +u

kill $(ps -eo pid,args | awk '/python3 scripts\/reach_fsm_node.py/ {print $1}') 2>/dev/null
kill $(ps -eo pid,args | awk '/python3 scripts\/align_vlm_decider.py/ {print $1}') 2>/dev/null
sleep 2

: > log/real_robot/reach_fsm.log
: > log/real_robot/align_vlm_decider.log
/usr/bin/python3 scripts/align_vlm_decider.py --qa-dir log/real_robot/qa --provider agent \
  > log/real_robot/align_vlm_decider.log 2>&1 &
echo VLM=$!
# Full center path: probe → center (+live-replan) → near (offset=0). Keep optical pitch TF.
/usr/bin/python3 scripts/reach_fsm_node.py \
  --ee-link link6 --cup-contact-offset 0.0 \
  --align-judge-mode file --align-traj-s 4.0 \
  --fine-assoc-max-m 0.0 --fine-max-age-s 0.5 \
  --align-max-steps 10 --refine-timeout-s 120 \
  --servo-step-m 0.02 --servo-traj-s 2.2 --servo-settle-s 0.7 \
  --servo-motion-snap-s 0.15 \
  --servo-orient-frac 0.0 \
  --servo-ang-deadband-deg 3.0 \
  --servo-near-handoff-z 0.05 --servo-near-handoff-dist-m 0.12 \
  --servo-near-confirm --refine-fruit-assoc-max-m 0.12 \
  --refine-lock-min-conf 0.40 --refine-track-min-conf 0.15 \
  --refine-fresh-lock-min-conf 0.10 \
  --servo-mono-probe --servo-mono-probe-steps 1 --servo-mono-probe-step-m 0.02 \
  --servo-mono-probe-yaw-frac 0.25 \
  --servo-center-pix-tol-px 35 --servo-center-ok-frames 3 \
  --servo-center-oneshot-gain 1.0 \
  --servo-center-oneshot-max 2 \
  --servo-center-live-replan \
  --servo-center-oneshot-max-m 0.0 \
  --servo-center-step-m 0.012 --servo-center-traj-s 2.6 \
  --refine-z-cam-max-jump-m 0.22 --refine-lost-frames-for-near 8 \
  --servo-estimate-max-age-s 8.0 \
  --servo-ibvs-reach-pix-px 80 --servo-ibvs-reach-pix-soft-px 280 \
  --servo-ibvs-reach-min-scale 0.30 \
  --servo-max-dj1-deg 2.5 --servo-max-dj5-deg 3.0 \
  --servo-max-dj2-deg 4 --servo-max-dj3-deg 4 \
  --servo-pixel-tol-px 45 --align-joint6-limit-rad 0.0 \
  --servo-near-oneshot-max 3 \
  --qa-dir log/real_robot/qa > log/real_robot/reach_fsm.log 2>&1 &
echo FSM=$!
sleep 3
head -5 log/real_robot/reach_fsm.log
ps -eo pid,args | grep 'python3 scripts/reach_fsm' | grep -v grep
