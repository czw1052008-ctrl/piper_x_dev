#!/usr/bin/env bash
# Standalone REFINING validator for the PBVS+KF pipeline (pbvs-vlm-reach-v2).
#
# Replaces _run_refine_contact.sh for the new PBVS path.
# Does NOT use REFINING_WAIT_NEAR / confirm_near — PBVS handles approach autonomously.
#
# Usage:
#   bash scripts/_run_refine_pbvs.sh                  # single run
#   bash scripts/_run_refine_pbvs.sh --loops 5        # stress-test
#   bash scripts/_run_refine_pbvs.sh --no-restore     # arm already at entry pose
#   bash scripts/_run_refine_pbvs.sh --collect-data   # enable BC data collection
#
# Prerequisites:
#   - agx_arm_ros bringup running (arm must be energised)
#   - refine_entry_pose.json saved:
#       python scripts/refine_entry_pose.py save   (arm at ALIGN exit pose)
#
# State progression (PBVS):
#   confirm_reset (if WAIT_CONFIRM) → restore entry → start_refine
#   REFINING lock → DIRECT_ONESHOT (cup→locked surface contact point)
#                 → optional residual → WAIT_CONFIRM
# No pre-grasp / depth_final. Locked UV+depth = surface point; cup clearance only.
set +e
cd /home/user/codes/piper_x_dev/blueberry_picking_ws
export PATH=/usr/bin:/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HOME=/home/user
export ROS_LOG_DIR=/home/user/codes/piper_x_dev/blueberry_picking_ws/log/real_robot/ros_logs
export LD_LIBRARY_PATH=/home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/orbbec_camera/lib:${LD_LIBRARY_PATH:-}
set +u
source /opt/ros/humble/setup.bash
source /home/user/codes/piper_x_dev/agx_arm_ros/install/setup.bash
source install/setup.bash
source /home/user/codes/piper_x_dev/OrbbecSDK_ROS2_main/install/setup.bash
set +u

mkdir -p log/real_robot log/real_robot/ros_logs

LOG=log/real_robot/_refine_pbvs.txt
: > "$LOG"
log() { echo "$*" | tee -a "$LOG"; }

# Wrist static TF must exist before fine_detector (uses base_link→optical lookup).
_republish_wrist_tf() {
  set +u
  # shellcheck disable=SC1091
  source config/real_robot.env
  set -u
  kill $(ps -eo pid,args | awk '/static_transform_publisher.*camera_wrist_color_optical_frame/ {print $1}') 2>/dev/null || true
  sleep 0.4
  IFS=',' read -r _roll _pitch _yaw <<< "${CAMERA_MOUNT_RPY:-0,0,0}"
  nohup ros2 run tf2_ros static_transform_publisher \
    --x "${CAMERA_MOUNT_TX}" --y "${CAMERA_MOUNT_TY}" --z "${CAMERA_MOUNT_TZ}" \
    --roll "${_roll:-0}" --pitch "${_pitch:-0}" --yaw "${_yaw:-0}" \
    --frame-id link6 --child-frame-id camera_wrist_color_optical_frame \
    >> log/real_robot/camera_wrist_tf.log 2>&1 &
  sleep 1.2
  log "wrist TF republished from real_robot.env (TX=${CAMERA_MOUNT_TX} TY=${CAMERA_MOUNT_TY})"
}

_republish_wrist_tf

# ── Pass-through flags to run_refine_pbvs.py ───────────────────────────────
RUNNER_ARGS=()
COLLECT_DATA=0
NO_RESTART=0
for arg in "$@"; do
  case "$arg" in
    --collect-data) COLLECT_DATA=1 ;;
    --no-restart) NO_RESTART=1 ;;
    *) RUNNER_ARGS+=("$arg") ;;
  esac
done

if [[ "$NO_RESTART" == 1 ]]; then
  log '=== skip FSM/detector restart (--no-restart) ==='
  log "=== running PBVS REFINING validator ==="
  /usr/bin/python3 scripts/run_refine_pbvs.py "${RUNNER_ARGS[@]}" 2>&1 | tee -a "$LOG"
  exit $?
fi

# ── fine_detector_node (always restart — pick up src changes) ───────────────
log '=== restarting fine_detector_node ==='
kill $(ps -eo pid,args | awk '/fine_detector_node/ {print $1}') 2>/dev/null
sleep 2
: > log/real_robot/fine_detector.log
nohup bash scripts/run_fine_detector_node.sh \
  -p depth_pose_source:=depth \
  -p lock_bbox_file:=/dev/null \
  >> log/real_robot/fine_detector.log 2>&1 &
log "fine_detector PID=$! (YOLO lock only — no calib file_bbox)"
sleep 5
if ! pgrep -f 'fine_detector_node' >/dev/null; then
  log 'ERROR: fine_detector failed to start'
  tail -5 log/real_robot/fine_detector.log | tee -a "$LOG"
  exit 1
fi
grep -m1 'depth_pose_source\|YOLO\|ultralytics\|unavailable' log/real_robot/fine_detector.log | tee -a "$LOG" || true

# ── reach_fsm_node (PBVS mode) ──────────────────────────────────────────────
log '=== restarting FSM (PBVS mode) ==='
kill $(ps -eo pid,args | awk '/python3 scripts\/reach_fsm_node.py/ {print $1}') 2>/dev/null
sleep 2
: > log/real_robot/reach_fsm.log

FSM_CMD=(
  /usr/bin/python3 scripts/reach_fsm_node.py
  --ee-link link6
  --use-pbvs
  --pbvs-mode single
  --align-judge-mode file
  --align-traj-s 4.0
  --align-max-steps 10
  --fine-assoc-max-m 0.0 --fine-max-age-s 0.5
  --refine-timeout-s 120
  --servo-step-m 0.02     --servo-traj-s 2.2  --servo-settle-s 0.7
  --servo-max-dj1-deg 2.5 --servo-max-dj5-deg 3.0
  --servo-max-dj2-deg 4   --servo-max-dj3-deg 4
  --servo-pixel-tol-px 45
  --servo-near-handoff-dist-m 0.08
  --cup-surface-clearance-m 0.0
  --berry-radius 0.0
  --pbvs-vision-lost-frames 3
  --pbvs-vision-z-cam-min 0.08
  --pbvs-near-d-cam-m 0.12
  --refine-fruit-assoc-max-m 0.12
  --refine-lock-min-conf 0.10
  --refine-lock-max-base-y-m 0.42
  --refine-track-min-conf 0.15
  --refine-fresh-lock-min-conf 0.10
  --no-refine-probe-tri-mono-chord
  --qa-dir log/real_robot/qa
)
if [[ "$COLLECT_DATA" == 1 ]]; then
  FSM_CMD+=(--collect-data --collect-data-dir data/align_episodes)
  log '  BC data collection ENABLED → data/align_episodes/'
fi

"${FSM_CMD[@]}" > log/real_robot/reach_fsm.log 2>&1 &
FSM_PID=$!
log "FSM PID=$FSM_PID"
sleep 3

if ! kill -0 $FSM_PID 2>/dev/null; then
  log "ERROR: FSM failed to start — last 10 lines:"
  tail -10 log/real_robot/reach_fsm.log | tee -a "$LOG"
  exit 1
fi
head -5 log/real_robot/reach_fsm.log | tee -a "$LOG"

# ── run_refine_pbvs.py ──────────────────────────────────────────────────────
log ""
log "=== running PBVS REFINING validator ==="
/usr/bin/python3 scripts/run_refine_pbvs.py "${RUNNER_ARGS[@]}" 2>&1 | tee -a "$LOG"
STATUS=$?

log ""
log "=== FSM key events ==="
grep -E 'PBVS|SERVO|APPROACH|probe|contact|cup_dist|KF|WAIT_CONFIRM|ERROR ' \
  log/real_robot/reach_fsm.log | tail -30 | tee -a "$LOG"

log ""
log "=== PBVS QA replay (per-tick jsonl + 4 cameras) ==="
LATEST_QA=$(ls -1d log/real_robot/qa/20* 2>/dev/null | sort | tail -1)
if [[ -n "$LATEST_QA" && -f "$LATEST_QA/pbvs_replay.html" ]]; then
  log "  open: $LATEST_QA/pbvs_replay.html"
elif [[ -n "$LATEST_QA" && -f "$LATEST_QA/pbvs_stream.jsonl" ]]; then
  /usr/bin/python3 scripts/pbvs_qa_recorder.py "$LATEST_QA" | tee -a "$LOG"
else
  log "  (no pbvs_stream.jsonl in latest QA session yet)"
fi

exit $STATUS
