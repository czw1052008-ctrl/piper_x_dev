#!/usr/bin/env bash
# Detect (YOLO+FP) -> plan nearest berry suction -> optional arm move.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda deactivate 2>/dev/null || true
export PATH="/usr/bin:/bin:/opt/ros/jazzy/bin:${PATH}"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"

bash "${ROOT}/scripts/run_grasp_planner.sh" || true

exec /usr/bin/python3 "${ROOT}/scripts/real_suction_approach.py" "$@"
