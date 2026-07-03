#!/usr/bin/env bash
# Source the correct ROS 2 distro for this machine.
set -eo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${VERSION_CODENAME:-}" in
    jammy) ROS_DISTRO=humble ;;
    noble) ROS_DISTRO=jazzy ;;
    *) echo "Unsupported Ubuntu: ${VERSION_CODENAME:-?}" >&2; exit 1 ;;
  esac
else
  ROS_DISTRO="${ROS_DISTRO:-humble}"
fi

SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ ! -f "${SETUP}" ]]; then
  echo "ROS 2 ${ROS_DISTRO} not installed. Run: bash scripts/install_deps.sh" >&2
  exit 1
fi

# ROS setup.bash references optional unset vars — disable nounset while sourcing
set +u
# shellcheck disable=SC1090
source "${SETUP}"
if [[ -f "${WS_ROOT}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${WS_ROOT}/install/setup.bash"
fi
# Keep nounset off: colcon setup.bash uses optional vars (e.g. COLCON_TRACE).
# Re-enabling set -u breaks a second `source install/setup.bash` after colcon build.

export ROS_DISTRO
export ROSDISTRO_INDEX_URL="${ROSDISTRO_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml}"

# Conda base env prepends its lib path and breaks Gazebo / gz_ros2_control plugin loading.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  _clean_path=""
  IFS=':' read -ra _ld_parts <<< "${LD_LIBRARY_PATH:-}"
  for _p in "${_ld_parts[@]}"; do
    [[ -z "${_p}" ]] && continue
    [[ "${_p}" == "${CONDA_PREFIX}/lib"* ]] && continue
    _clean_path="${_clean_path:+${_clean_path}:}${_p}"
  done
  export LD_LIBRARY_PATH="${_clean_path}"
fi
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/${ROS_DISTRO}/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"

# Gazebo Harmonic mesh lookup (package:// → model://)
if [[ -f "${WS_ROOT}/install/agx_arm_description/share/agx_arm_description/package.xml" ]]; then
  _GZ_AGX="${WS_ROOT}/install/agx_arm_description/share"
  _GZ_PICK="${WS_ROOT}/install/picking_description/share"
  export GZ_SIM_RESOURCE_PATH="${_GZ_AGX}:${_GZ_PICK}:${GZ_SIM_RESOURCE_PATH:-/opt/ros/jazzy/share}"
fi

echo "Sourced ROS 2 ${ROS_DISTRO} + workspace ${WS_ROOT}"
