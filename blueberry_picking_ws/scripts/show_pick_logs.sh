#!/usr/bin/env bash
# Show the latest ros2 launch session log and key node logs (filters TF spam).
set -euo pipefail

ROS_LOG="${HOME}/.ros/log"
if [[ ! -d "${ROS_LOG}" ]]; then
  echo "No ROS logs at ${ROS_LOG}" >&2
  exit 1
fi

LATEST_DIR="$(ls -td "${ROS_LOG}"/2026-* 2>/dev/null | head -1 || true)"
if [[ -z "${LATEST_DIR}" ]]; then
  echo "No launch session directory under ${ROS_LOG}" >&2
  exit 1
fi

echo "=== Launch session ==="
echo "${LATEST_DIR}"
echo

if command -v ros2 >/dev/null 2>&1; then
  echo "=== ROS graph sanity (duplicate servers break picking) ==="
  proc_count="$(pgrep -afc 'lib/picking_task/pick_action_server' 2>/dev/null || echo 0)"
  echo "pick_action_server processes (pgrep): ${proc_count}"
  ros2 action info /pick_blueberry 2>/dev/null | sed -n '1,12p' || true
  svc_count="$(ros2 service list 2>/dev/null | grep -c '/trigger_global_detection$' || true)"
  if [[ "${svc_count}" -gt 1 ]]; then
    echo "WARN: ${svc_count} trigger_global_detection services (expected 1)"
    ros2 service list 2>/dev/null | grep trigger_global_detection || true
  fi
  echo
fi

if [[ -f "${LATEST_DIR}/launch.log" ]]; then
  echo "=== launch.log (errors + pick flow, no tf2_buffer) ==="
  grep -v 'tf2_buffer' "${LATEST_DIR}/launch.log" \
    | grep -E 'ERROR|WARN|Pick |plan_vibration|GlobalDetect|Cartesian|MoveTo|send_pick|grasp_planner|BT |SUCCESS|FAILED|died' \
    | tail -80 || true
  echo
fi

echo "=== Recent flat node logs in ${ROS_LOG} ==="
ls -lt "${ROS_LOG}"/*.log 2>/dev/null | head -8 || true
echo

launch_mtime="$(stat -c %Y "${LATEST_DIR}" 2>/dev/null || echo 0)"

for pattern in pick_action_server grasp_planner send_pick_goal fake_perception; do
  f="$(ls -t "${ROS_LOG}/${pattern}"_*.log 2>/dev/null | head -1 || true)"
  if [[ -n "${f}" ]]; then
    file_mtime="$(stat -c %Y "${f}" 2>/dev/null || echo 0)"
    if [[ "${file_mtime}" -lt "${launch_mtime}" ]]; then
      echo "=== $(basename "${f}") (skipped: older than latest launch session) ==="
      continue
    fi
    echo "=== $(basename "${f}") (last 30 lines, no tf2) ==="
    grep -v 'tf2_buffer' "${f}" | tail -30 || true
    echo
  fi
done

echo "Tip: full launch log -> ${LATEST_DIR}/launch.log"
echo "Tip: save terminal output -> ros2 launch ... 2>&1 | tee ~/pick_run.log"
