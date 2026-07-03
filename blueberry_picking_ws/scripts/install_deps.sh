#!/usr/bin/env bash
# Install ROS 2 dependencies for blueberry_picking_ws
# - Ubuntu 22.04 (jammy)  → ROS 2 Humble + Gazebo Classic
# - Ubuntu 24.04 (noble)  → ROS 2 Jazzy   + Gazebo Harmonic (gz)
set -euo pipefail

if [[ ! -r /etc/os-release ]]; then
  echo "ERROR: cannot read /etc/os-release" >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release

case "${VERSION_CODENAME:-}" in
  jammy)
    ROS_DISTRO=humble
    GAZEBO_PROFILE=classic
    ;;
  noble)
    ROS_DISTRO=jazzy
    GAZEBO_PROFILE=harmonic
    ;;
  *)
    echo "ERROR: Unsupported Ubuntu codename: ${VERSION_CODENAME:-unknown}" >&2
    echo "       Supported: jammy (22.04) or noble (24.04)" >&2
    echo "       Your design doc targets Humble — use Ubuntu 22.04, Docker, or accept Jazzy on 24.04." >&2
    exit 1
    ;;
esac

echo "==> Detected Ubuntu ${VERSION_ID} (${VERSION_CODENAME}), will install ROS 2 ${ROS_DISTRO}"

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

ensure_ros_apt_repo() {
  if [[ -f /etc/apt/sources.list.d/ros2.list ]]; then
    echo "==> ROS 2 apt repository already configured"
    return
  fi

  echo "==> Adding ROS 2 apt repository (requires sudo)..."
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y curl gnupg lsb-release software-properties-common

  ${SUDO} curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu ${VERSION_CODENAME} main" \
    | ${SUDO} tee /etc/apt/sources.list.d/ros2.list > /dev/null
}

install_ros_packages() {
  local prefix="ros-${ROS_DISTRO}"
  local -a core_pkgs=(
    "${prefix}-desktop"
    "${prefix}-moveit"
    "${prefix}-ros2-control"
    "${prefix}-ros2-controllers"
    "${prefix}-behaviortree-cpp"
    "${prefix}-xacro"
    "${prefix}-tf2-ros"
    "${prefix}-cv-bridge"
    "${prefix}-robot-state-publisher"
    "${prefix}-launch-ros"
    python3-colcon-common-extensions
    python3-pytest
    python3-rosdep
    python3-catkin-pkg
    build-essential
    cmake
    git
    libeigen3-dev
  )

  local -a sim_pkgs=()
  if [[ "${GAZEBO_PROFILE}" == "classic" ]]; then
    sim_pkgs+=(
      "${prefix}-gazebo-ros-pkgs"
      "${prefix}-gazebo-ros2-control"
      gazebo
    )
  else
    sim_pkgs+=(
      "${prefix}-ros-gz"
      "${prefix}-ros-gz-sim"
      "${prefix}-gz-ros2-control"
    )
  fi

  echo "==> Installing core ROS packages..."
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y "${core_pkgs[@]}"

  echo "==> Installing simulation packages (optional for sim_bt_only)..."
  if ! ${SUDO} apt-get install -y "${sim_pkgs[@]}"; then
    echo "WARN: Simulation packages failed — you can still build and run sim_bt_only.launch.py" >&2
  fi
}

init_rosdep() {
  if ! command -v rosdep >/dev/null 2>&1; then
    return
  fi
  if [[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    rosdep update || true
    return
  fi
  echo "==> Initializing rosdep via mirror (raw.githubusercontent.com often times out)..."
  local init_script
  init_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/init_rosdep.sh"
  if [[ -x "${init_script}" ]]; then
    bash "${init_script}" || echo "WARN: rosdep init failed — you can skip it and run colcon build directly" >&2
  else
    ${SUDO} rosdep init || true
    rosdep update || true
  fi
}

write_env_hint() {
  local setup_file="/opt/ros/${ROS_DISTRO}/setup.bash"
  cat <<EOF

============================================================
ROS 2 ${ROS_DISTRO} installation finished.
============================================================

Add to ~/.bashrc (optional):
  echo 'source ${setup_file}' >> ~/.bashrc

Build workspace:
  source ${setup_file}
  cd $(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  rosdep install --from-paths src --ignore-src -r -y || true
  colcon build --symlink-install
  source install/setup.bash

Quick test (no Gazebo required):
  ros2 launch picking_bringup sim_bt_only.launch.py

EOF
  if [[ "${GAZEBO_PROFILE}" == "harmonic" ]]; then
    cat <<EOF
NOTE (Ubuntu 24.04):
  - You are on ROS 2 Jazzy, not Humble.
  - Gazebo Classic launch files may need migration to gz sim.
  - Use sim_bt_only.launch.py first; full Gazebo sim is WIP on Noble.

EOF
  fi
}

ensure_ros_apt_repo
install_ros_packages
init_rosdep
write_env_hint
