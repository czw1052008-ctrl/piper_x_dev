#!/usr/bin/env bash
# Initialize rosdep using Tsinghua mirror (works when raw.githubusercontent.com times out)
set -eo pipefail

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${VERSION_CODENAME:-}" in
    jammy) ROS_DISTRO=humble ;;
    noble) ROS_DISTRO=jazzy ;;
    *) ROS_DISTRO="${ROS_DISTRO:-jazzy}" ;;
  esac
else
  ROS_DISTRO="${ROS_DISTRO:-jazzy}"
fi

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u 2>/dev/null || true
else
  echo "ERROR: /opt/ros/${ROS_DISTRO}/setup.bash not found. Run scripts/install_deps.sh first." >&2
  exit 1
fi

export ROS_DISTRO
export ROS_PYTHON_VERSION="${ROS_PYTHON_VERSION:-3}"

if [[ "${EUID}" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi

MIRROR="${ROSDEP_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/rosdistro}"
export ROSDISTRO_INDEX_URL="${ROSDISTRO_INDEX_URL:-${MIRROR}/index-v4.yaml}"

if [[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  echo "rosdep already initialized at /etc/ros/rosdep/sources.list.d/20-default.list"
else
  echo "==> Creating rosdep sources list (mirror: ${MIRROR})..."
  ${SUDO} mkdir -p /etc/ros/rosdep/sources.list.d
  ${SUDO} tee /etc/ros/rosdep/sources.list.d/20-default.list > /dev/null <<EOF
yaml ${MIRROR}/rosdep/osx-homebrew.yaml osx
yaml ${MIRROR}/rosdep/base.yaml
yaml ${MIRROR}/rosdep/python.yaml
yaml ${MIRROR}/rosdep/ruby.yaml
gbpdistro ${MIRROR}/releases/fuerte.yaml fuerte
EOF
fi

echo "==> rosdep update (index: ${ROSDISTRO_INDEX_URL})..."
rosdep update

echo "Done. Add to ~/.bashrc to persist mirror settings:"
echo "  export ROSDISTRO_INDEX_URL=${ROSDISTRO_INDEX_URL}"
echo ""
echo "Then run:"
echo "  rosdep install --from-paths src --ignore-src -r -y"
