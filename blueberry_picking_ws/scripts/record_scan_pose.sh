#!/usr/bin/env bash
# Record current arm joints as a patrol scan pose (use after teleop).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda deactivate 2>/dev/null || true
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"

if ! timeout 4 ros2 topic echo /joint_states --once >/dev/null 2>&1; then
  echo "[record] ERROR: no /joint_states — start bringup first." >&2
  exit 1
fi

exec /usr/bin/python3 "${ROOT}/scripts/record_scan_pose.py" "$@"
