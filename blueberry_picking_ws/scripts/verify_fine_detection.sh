#!/usr/bin/env bash
# Verify weights + FoundationPose wrapper (+ optional live ROS service).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP_ROOT="${FOUNDATIONPOSE_ROOT:-$(cd "${ROOT}/.." && pwd)/FoundationPose}"
ENV_NAME="${FOUNDATIONPOSE_CONDA_ENV:-foundationpose}"
SKIP_WRAPPER=false
ROS_SERVICE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-wrapper) SKIP_WRAPPER=true; shift ;;
    --ros-service) ROS_SERVICE=true; shift ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

bash "${ROOT}/scripts/extract_foundationpose_weights.sh"

eval "$(conda shell.bash hook)"
if conda activate "${ENV_NAME}" 2>/dev/null; then
  export PYTHONPATH="${ROOT}/src/picking_perception:${PYTHONPATH:-}"
  ARGS=(--foundation-pose-root "${FP_ROOT}")
  [[ "${SKIP_WRAPPER}" == "true" ]] && ARGS+=(--skip-wrapper)
  [[ "${ROS_SERVICE}" == "true" ]] && ARGS+=(--ros-service)
  python "${ROOT}/scripts/verify_fine_detection.py" "${ARGS[@]}"
else
  echo "WARN: conda env ${ENV_NAME} missing — weights-only check"
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash 2>/dev/null || true
  # shellcheck disable=SC1090
  source "${ROOT}/install/setup.bash" 2>/dev/null || true
  python3 "${ROOT}/scripts/verify_fine_detection.py" \
    --foundation-pose-root "${FP_ROOT}" \
    --skip-wrapper \
    "${ROS_SERVICE:+--ros-service}"
fi
