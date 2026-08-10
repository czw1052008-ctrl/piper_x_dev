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
#   REFINING (INIT) → SERVO → APPROACH → WAIT_CONFIRM (contact ✓)
#                                      └→ ERROR
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

# ── Pass-through flags to run_refine_pbvs.py ───────────────────────────────
RUNNER_ARGS=()
COLLECT_DATA=0
for arg in "$@"; do
  case "$arg" in
    --collect-data) COLLECT_DATA=1 ;;
    *) RUNNER_ARGS+=("$arg") ;;
  esac
done

# ── fine_detector_node ──────────────────────────────────────────────────────
if ! pgrep -f 'fine_detector_node' >/dev/null; then
  log 'starting fine_detector_node …'
  nohup /usr/bin/python3 -u \
    install/picking_perception/lib/picking_perception/fine_detector_node \
    --ros-args \
    -p enable_foundation_pose:=false \
    -p publish_hz:=10.0 \
    -p mask_source:=yolo \
    --params-file src/picking_perception/config/foundation_pose.yaml \
    >> log/real_robot/fine_detector.log 2>&1 &
  log "fine_detector PID=$!"
  sleep 5
else
  log 'fine_detector_node already running'
fi

# ── reach_fsm_node (PBVS mode) ──────────────────────────────────────────────
log '=== restarting FSM (PBVS mode) ==='
kill $(ps -eo pid,args | awk '/python3 scripts\/reach_fsm_node.py/ {print $1}') 2>/dev/null
sleep 2
: > log/real_robot/reach_fsm.log

FSM_CMD=(
  /usr/bin/python3 scripts/reach_fsm_node.py
  --ee-link link6
  --use-pbvs
  --align-judge-mode file
  --align-traj-s 4.0
  --align-max-steps 10
  --fine-assoc-max-m 0.0 --fine-max-age-s 0.5
  --refine-timeout-s 120
  --servo-step-m 0.02     --servo-traj-s 2.2  --servo-settle-s 0.7
  --servo-max-dj1-deg 2.5 --servo-max-dj5-deg 3.0
  --servo-max-dj2-deg 4   --servo-max-dj3-deg 4
  --servo-pixel-tol-px 45
  --refine-fruit-assoc-max-m 0.12
  --refine-lock-min-conf 0.40
  --refine-track-min-conf 0.15
  --refine-fresh-lock-min-conf 0.10
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

exit $STATUS
