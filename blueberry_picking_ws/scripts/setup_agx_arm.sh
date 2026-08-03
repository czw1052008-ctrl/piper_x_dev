#!/usr/bin/env bash
# Link local agx_arm_description into this workspace (prefer monorepo, no network).
set -eo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${WS_ROOT}/src"
LINK="${SRC}/agx_arm_description"
PIPER_X_DEV="$(cd "${WS_ROOT}/.." && pwd)"
LOCAL_DESC="${PIPER_X_DEV}/agx_arm_ros/src/agx_arm_description"
VENDOR="${SRC}/external/agx_arm_ros"

mkdir -p "${SRC}/external"

if [[ -d "${LOCAL_DESC}" ]]; then
  ln -sfn "${LOCAL_DESC}" "${LINK}"
  echo "Linked ${LINK} -> ${LOCAL_DESC}"
  exit 0
fi

if [[ ! -d "${VENDOR}/.git" ]]; then
  echo "Local agx_arm_description missing; cloning agx_arm_ros ..."
  git clone --depth 1 https://github.com/agilexrobotics/agx_arm_ros.git "${VENDOR}"
else
  echo "Updating agx_arm_ros ..."
  git -C "${VENDOR}" pull --ff-only || true
  git -C "${VENDOR}" submodule update --init --recursive || true
fi

ln -sfn external/agx_arm_ros/src/agx_arm_description "${LINK}"
echo "Linked ${LINK} -> external/agx_arm_ros/src/agx_arm_description"
