#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/bin:/bin:${PATH}"
export RMW_FASTRTPS_USE_SHM=0 PYTHONNOUSERSITE=1 HOME=/home/user
export ROS_LOG_DIR="${ROOT}/log/real_robot/ros_logs"
mkdir -p "${ROS_LOG_DIR}" log/real_robot/qa
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source "${ROOT}/install/setup.bash"
set +u
: > log/real_robot/reach_fsm.log
: > log/real_robot/align_vlm_decider.log
/usr/bin/python3 scripts/align_vlm_decider.py --qa-dir log/real_robot/qa --provider agent \
  > log/real_robot/align_vlm_decider.log 2>&1 &
echo VLM_PID=$!
/usr/bin/python3 scripts/reach_fsm_node.py \
  --ee-link tcp_link --cup-contact-offset 0.015 \
  --align-judge-mode file --align-traj-s 4.0 \
  --align-fine-visible-conf 0.55 --fine-assoc-max-m 0.0 --fine-max-age-s 0.5 \
  --align-max-steps 10 --refine-timeout-s 90 \
  --servo-step-m 0.02 --servo-traj-s 2.2 --servo-settle-s 0.6 \
  --servo-yaw-gain 0.35 --servo-pitch-gain 0.30 --servo-reach-gain 4.0 \
  --servo-ang-deadband-deg 2.0 --servo-approach-ang-deg 8.0 \
  --servo-pixel-tol-px 45 \
  --align-joint6-limit-rad 0.0 \
  --qa-dir log/real_robot/qa \
  > log/real_robot/reach_fsm.log 2>&1 &
echo FSM_PID=$!
sleep 3
head -20 log/real_robot/reach_fsm.log
