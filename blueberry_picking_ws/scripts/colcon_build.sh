#!/usr/bin/env bash
# Build workspace (avoids conda Python/OpenSSL breaking ament/MoveIt link)
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u 2>/dev/null || true

# Conda paths break MoveIt/resource_retriever OpenSSL linking.
filter_path() {
  local var_name="$1"
  local val="${!var_name:-}"
  if [[ -n "${val}" ]]; then
    val="$(echo "${val}" | tr ':' '\n' | grep -v miniconda | paste -sd: -)"
    export "${var_name}=${val}"
  fi
}
filter_path LD_LIBRARY_PATH
filter_path LIBRARY_PATH
filter_path CMAKE_PREFIX_PATH
filter_path PKG_CONFIG_PATH
export CMAKE_IGNORE_PATH="${CMAKE_IGNORE_PATH:-}/home/ziwei/miniconda3"

cd "${WS}"
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 "$@"

echo ""
echo "Build OK. Run: source ${WS}/install/setup.bash"
