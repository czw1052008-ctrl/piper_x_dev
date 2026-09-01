#!/usr/bin/env bash
# One-click real-time pick system (PBVS mainline + pick-cycle FSM).
# Replaces run_real_reach.sh for production picking.
#
# Usage:
#   bash scripts/run_pick_system.sh
#   bash scripts/run_pick_system.sh --test-config config/pick_test_fruit_task.json
#   bash scripts/run_pick_system.sh --dry-run
#
# Operator flow:
#   1. ros2 topic pub --once /pick/cmd std_msgs/String "{data: start_mission}"
#   2. Terminal 2: python3 scripts/suction_keyboard_sim.py
#      s=ON (during touch)  y=touch_ok  e=OFF (after retract)  f=touch_fail
#   3. On failure: analyze log/real_robot/pick_failures/ then:
#      ros2 topic pub --once /pick/cmd std_msgs/String "{data: retry_fruit}"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
CONFIG="${ROOT}/config/real_robot.env"
DRY_RUN=false
TEST_CONFIG=""
EXTRA=()

usage() {
  cat <<EOF
Usage: bash scripts/run_pick_system.sh [--dry-run] [--config PATH]
                                     [--test-config PATH]

Starts: arm + Orbbec + perception + reach_fsm (PBVS) + pick_cycle_fsm.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --test-config) TEST_CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
fi

bash "${ROOT}/scripts/real_robot_bringup.sh" \
  --config "${CONFIG}" \
  --fixed-cam \
  --reach-perception \
  --no-wait \
  "${EXTRA[@]+"${EXTRA[@]}"}"

set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
if [[ -f "${AGX_ARM_WS:-}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${AGX_ARM_WS}/install/setup.bash"
fi
set -u 2>/dev/null || true

QA_DIR="${ROOT}/log/real_robot/qa"
mkdir -p "${QA_DIR}" "${ROOT}/log/real_robot/pick_failures"

# fine_detector (YOLO lock only)
if ! pgrep -f 'fine_detector_node' >/dev/null 2>&1; then
  echo "[run_pick_system] starting fine_detector_node"
  nohup bash "${ROOT}/scripts/run_fine_detector_node.sh" \
    -p depth_pose_source:=depth \
    -p lock_bbox_file:=/dev/null \
    >> "${ROOT}/log/real_robot/fine_detector.log" 2>&1 &
  sleep 4
fi

# global_detector (fixed clusters + tracking)
if ! pgrep -f 'global_detector_node' >/dev/null 2>&1; then
  echo "[run_pick_system] starting global_detector_node"
  nohup bash "${ROOT}/scripts/run_global_detector_node.sh" \
    >> "${ROOT}/log/real_robot/global_detector.log" 2>&1 &
  sleep 2
fi

# unified scene_graph (clusters + berries, stable ids)
if ! pgrep -f 'scene_graph_node' >/dev/null 2>&1; then
  echo "[run_pick_system] starting scene_graph_node"
  nohup bash "${ROOT}/scripts/run_scene_graph_node.sh" \
    >> "${ROOT}/log/real_robot/scene_graph.log" 2>&1 &
  sleep 1
fi

# obstacle occupancy (depth − fruit − arm) → /perception/obstacles
if ! pgrep -f 'obstacle_extractor_node' >/dev/null 2>&1; then
  echo "[run_pick_system] starting obstacle_extractor_node"
  nohup bash "${ROOT}/scripts/run_obstacle_extractor_node.sh" \
    >> "${ROOT}/log/real_robot/obstacle_extractor.log" 2>&1 &
  sleep 1
fi

# Wrist static TF (same as _run_refine_pbvs.sh)
_republish_wrist_tf() {
  set +u
  # shellcheck disable=SC1091
  source "${ROOT}/config/real_robot.env" 2>/dev/null || true
  set -u
  kill $(ps -eo pid,args | awk '/static_transform_publisher.*camera_wrist_color_optical_frame/ {print $1}') 2>/dev/null || true
  sleep 0.3
  IFS=',' read -r _roll _pitch _yaw <<< "${CAMERA_MOUNT_RPY:-0,0,0}"
  nohup ros2 run tf2_ros static_transform_publisher \
    --x "${CAMERA_MOUNT_TX:-0}" --y "${CAMERA_MOUNT_TY:-0}" --z "${CAMERA_MOUNT_TZ:-0}" \
    --roll "${_roll:-0}" --pitch "${_pitch:-0}" --yaw "${_yaw:-0}" \
    --frame-id link6 --child-frame-id camera_wrist_color_optical_frame \
    >> "${ROOT}/log/real_robot/camera_wrist_tf.log" 2>&1 &
  sleep 0.8
}
_republish_wrist_tf

DRY_FLAG=()
[[ "${DRY_RUN}" == "true" ]] && DRY_FLAG=(--dry-run)

# reach_fsm — PBVS only
if pgrep -f 'python3 scripts/reach_fsm_node.py' >/dev/null; then
  kill $(pgrep -f 'python3 scripts/reach_fsm_node.py') 2>/dev/null || true
  sleep 1
fi

# P0b executor (optional): PBVS publishes tool_trajectory_4s, executor drives arm.
if [[ "${PBVS_VIA_EXECUTOR:-false}" == "true" ]]; then
  if ! pgrep -f 'trajectory_executor_node.py' >/dev/null 2>&1; then
    echo "[run_pick_system] starting trajectory_executor (--drive-arm)"
    nohup bash "${ROOT}/scripts/run_trajectory_executor.sh" --drive-arm \
      >> "${ROOT}/log/real_robot/trajectory_executor.log" 2>&1 &
    sleep 1
  fi
fi

FSM_CMD=(
  python3 "${ROOT}/scripts/reach_fsm_node.py"
  --ee-link link6
  --use-pbvs
  --pbvs-mode single
  --align-judge-mode file
  --align-traj-s 4.0
  --refine-timeout-s 120
  --servo-traj-s 2.2
  --servo-settle-s 0.7
  --cup-surface-clearance-m 0.0
  --berry-radius 0.0
  --refine-lock-min-conf 0.10
  --refine-lock-max-base-y-m 0.42
  --pbvs-press-m 0
  --qa-dir "${QA_DIR}"
  "${DRY_FLAG[@]}"
)
if [[ "${PBVS_VIA_EXECUTOR:-false}" == "true" ]]; then
  FSM_CMD+=(--pbvs-via-executor)
fi
nohup "${FSM_CMD[@]}" >> "${ROOT}/log/real_robot/reach_fsm.log" 2>&1 &
sleep 2

if ! timeout 12 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | grep -q 'success=True'; then
  echo "[run_pick_system] WARN: enable_agx_arm failed" >&2
fi
timeout 8 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>/dev/null || true

PICK_ARGS=(
  --qa-dir "${QA_DIR}"
  --entry-pose "${ROOT}/log/real_robot/refine_entry_pose.json"
  --retract-m 0.05
  --retract-traj-s 0.8
  --entry-restore-traj-s 2.0
)
[[ -n "${TEST_CONFIG}" ]] && PICK_ARGS+=(--test-config "${ROOT}/${TEST_CONFIG}")

cat <<EOF

================================================================================
  Pick system ready (PBVS + pick_cycle_fsm).
  ros2 topic echo /pick/status
  ros2 topic pub --once /pick/cmd std_msgs/String "{data: start_mission}"

  Keyboard (separate terminal):
    cd ${ROOT} && python3 scripts/suction_keyboard_sim.py

  Test fruit-task only:
    bash scripts/run_pick_system.sh --test-config config/pick_test_fruit_task.json

  Shutdown: bash scripts/_safe_poweroff.sh
================================================================================

EOF

exec python3 "${ROOT}/scripts/pick_cycle_fsm_node.py" "${PICK_ARGS[@]}"
