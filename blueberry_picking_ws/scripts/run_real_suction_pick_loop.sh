#!/usr/bin/env bash
# Full real-robot suction pipeline (pick loop only — stack must already be running).
#
# One-click (stack + pick):  bash scripts/run_real_suction_pick.sh
#
# This script alone:
#   bash scripts/run_real_suction_pick_loop.sh --move --once --auto-retreat
#
# Prereq if using this script directly: bash scripts/real_robot_bringup.sh --perception
#   R / H  stop + go home (teleop zero joints)
#   G      at grasp: suction on, then retreat
#   Q      quit
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log/real_robot"
mkdir -p "${LOG_DIR}"

conda deactivate 2>/dev/null || true
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"

if ! ros2 topic list 2>/dev/null | grep -q '/camera_wrist/color/image_raw'; then
  echo "[pick-loop] ERROR: camera not running (no /camera_wrist/color/image_raw)." >&2
  echo "  Arm stack may be up, but Orbbec camera is not." >&2
  echo "  Terminal 1 — full stack:" >&2
  echo "    bash scripts/real_robot_bringup.sh --perception" >&2
  echo "  Or camera only + perception in two steps:" >&2
  echo "    bash scripts/real_robot_bringup.sh --camera-only" >&2
  echo "    bash scripts/run_perception_only.sh" >&2
  echo "  Check: ros2 topic hz /camera_wrist/color/image_raw" >&2
  exit 1
fi

if ! ros2 service list 2>/dev/null | grep -q '/trigger_fine_detection'; then
  echo "[pick-loop] ERROR: perception not ready." >&2
  echo "  bash scripts/run_perception_only.sh" >&2
  exit 1
fi

if ! ros2 action list 2>/dev/null | grep -q '/move_action'; then
  echo "[pick-loop] ERROR: move_action not available (arm/MoveIt not running)." >&2
  echo "  Terminal 1: bash scripts/real_robot_bringup.sh --perception" >&2
  exit 1
fi

if ! timeout 4 ros2 topic echo /joint_states --once >/dev/null 2>&1; then
  echo "[pick-loop] ERROR: no /joint_states (arm driver dead or not enabled)." >&2
  echo "  Check: tail -40 log/real_robot/arm.log" >&2
  echo "  Fix:" >&2
  echo "    bash scripts/real_robot_shutdown.sh" >&2
  echo "    bash scripts/real_robot_bringup.sh --perception" >&2
  echo "  Or enable manually: ros2 service call /enable_agx_arm std_srvs/srv/SetBool \"{data: true}\"" >&2
  exit 1
fi

if ! timeout 5 ros2 run tf2_ros tf2_echo base_link camera_wrist_link 2>/dev/null | head -3 | grep -q Translation; then
  echo "[pick-loop] WARN: TF base_link -> camera_wrist_link not ready." >&2
  echo "  Ensure bringup finished (camera_tf + Orbbec). Z-search may skip; patrol still runs." >&2
fi

bash "${ROOT}/scripts/run_grasp_planner.sh" || true

# Detection viz (YOLO lock overlay)
if ! pgrep -f 'visualize_blueberry_detection.py' >/dev/null 2>&1; then
  echo "[pick-loop] Starting detection viz (background) ..."
  nohup bash "${ROOT}/scripts/run_detection_viz.sh" --show \
    >"${LOG_DIR}/detection_viz.log" 2>&1 &
  sleep 2
else
  echo "[pick-loop] detection viz already running"
fi

echo "[pick-loop] Pipeline: HOME -> DETECT -> PATROL? -> LOCK -> PLAN -> MOVE"
echo "[pick-loop] Film: bash scripts/run_real_suction_pick_loop.sh --move --once --auto-retreat"
echo "[pick-loop] Teleop poses: config/pick_scan_recorded.txt (direct, dwell 3s)"
echo ""

exec /usr/bin/python3 "${ROOT}/scripts/real_suction_pick_loop.py" "$@"
