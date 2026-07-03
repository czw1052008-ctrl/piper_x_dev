#!/usr/bin/env bash
# Start fine_detector_node when camera is already running (no full bringup).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/log/real_robot"
PIDFILE="${LOG_DIR}/perception.pid"

mkdir -p "${LOG_DIR}"

if [[ -f "${PIDFILE}" ]]; then
  if kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    if ros2 service list 2>/dev/null | grep -q '/trigger_fine_detection'; then
      echo "[perception] Already running (pid $(cat "${PIDFILE}"))."
      exit 0
    fi
    echo "[perception] WARN: stale pid $(cat "${PIDFILE}") — restarting ..."
    kill "$(cat "${PIDFILE}")" 2>/dev/null || true
  fi
  rm -f "${PIDFILE}"
fi

conda deactivate 2>/dev/null || true
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL 2>/dev/null || true

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"

if ! ros2 topic list 2>/dev/null | grep -q '/camera_wrist/color/image_raw'; then
  echo "[perception] ERROR: /camera_wrist/color/image_raw not found." >&2
  echo "  Start camera first: bash scripts/real_robot_bringup.sh --camera-only" >&2
  exit 1
fi

_perception_ready() {
  ros2 node list 2>/dev/null | grep -q '/fine_detector_node' \
    && ros2 service list 2>/dev/null | grep -q '/trigger_fine_detection'
}

if _perception_ready; then
  echo "[perception] fine_detector_node already running."
  exit 0
fi

# Ghost node or stale service after pkill / crash
if ros2 node list 2>/dev/null | grep -q '/fine_detector_node' \
   || ros2 service list 2>/dev/null | grep -q '/trigger_fine_detection'; then
  echo "[perception] WARN: stale fine_detector DDS entry — cleaning up ..."
  pkill -f 'picking_perception.fine_detector_node' 2>/dev/null || true
  pkill -f 'run_fine_detector_node.sh' 2>/dev/null || true
  sleep 2
fi

echo "[perception] Starting fine_detector_node ..."
nohup bash "${ROOT}/scripts/run_fine_detector_node.sh" >"${LOG_DIR}/perception.log" 2>&1 &
echo $! >"${PIDFILE}"

for i in $(seq 1 90); do
  if _perception_ready; then
    echo "[perception] Ready: /trigger_fine_detection (log: ${LOG_DIR}/perception.log)"
    exit 0
  fi
  if ! kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    echo "[perception] ERROR: fine_detector exited early — see ${LOG_DIR}/perception.log" >&2
    rm -f "${PIDFILE}"
    exit 1
  fi
  sleep 1
done

echo "[perception] WARN: service not listed after 90s — check ${LOG_DIR}/perception.log" >&2
exit 1
