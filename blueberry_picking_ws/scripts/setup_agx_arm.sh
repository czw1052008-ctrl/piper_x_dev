#!/usr/bin/env bash
# Clone AgileX arm description and link into this workspace.
set -eo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${WS_ROOT}/src"
VENDOR="${SRC}/external/agx_arm_ros"
LINK="${SRC}/agx_arm_description"

mkdir -p "${SRC}/external"

if [[ ! -d "${VENDOR}/.git" ]]; then
  echo "Cloning agx_arm_ros ..."
  git clone --depth 1 https://github.com/agilexrobotics/agx_arm_ros.git "${VENDOR}"
else
  echo "Updating agx_arm_ros ..."
  git -C "${VENDOR}" pull --ff-only
  git -C "${VENDOR}" submodule update --init --recursive
fi

ln -sfn external/agx_arm_ros/src/agx_arm_description "${LINK}"
echo "Linked ${LINK} -> external/agx_arm_ros/src/agx_arm_description"
