#!/usr/bin/env bash
# fine_detector_node with FoundationPose conda env + ROS 2 Jazzy (Python 3.12).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${FOUNDATIONPOSE_CONDA_ENV:-foundationpose}"

if [[ -f "${ROOT}/config/real_robot.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT}/config/real_robot.env"
fi

set +u
eval "$(conda shell.bash hook)"
if ! conda activate "${ENV_NAME}" 2>/dev/null; then
  echo "[fine_detector] ERROR: conda env '${ENV_NAME}' not found." >&2
  echo "  Run: bash scripts/setup_foundationpose_env.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1090
source "${ROOT}/install/setup.bash"

ROS_SITE="/opt/ros/jazzy/lib/python3.12/site-packages"
WS_SITE="${ROOT}/install/picking_perception/lib/python3.12/site-packages"
MSG_SITE="${ROOT}/install/picking_msgs/lib/python3.12/site-packages"
export PYTHONPATH="${WS_SITE}:${MSG_SITE}:${ROS_SITE}:${ROOT}/src/picking_perception:${PYTHONPATH:-}"

FP_CONFIG="${ROOT}/src/picking_perception/config/foundation_pose.yaml"
ARGS=(--ros-args)
if [[ -f "${FP_CONFIG}" ]]; then
  ARGS+=(--params-file "${FP_CONFIG}")
fi
ARGS+=("$@")

exec python -m picking_perception.fine_detector_node "${ARGS[@]}"
