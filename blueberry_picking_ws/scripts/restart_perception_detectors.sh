#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export HOME=/home/user
source /opt/ros/humble/setup.bash
source install/setup.bash
mkdir -p log/real_robot
: > log/real_robot/global_det.log
: > log/real_robot/fine_det.log
bash scripts/run_global_detector_node.sh >> log/real_robot/global_det.log 2>&1 &
echo $! > log/real_robot/global_det.pid
bash scripts/run_fine_detector_node.sh >> log/real_robot/fine_det.log 2>&1 &
echo $! > log/real_robot/fine_det.pid
echo "started global=$(cat log/real_robot/global_det.pid) fine=$(cat log/real_robot/fine_det.pid)"
