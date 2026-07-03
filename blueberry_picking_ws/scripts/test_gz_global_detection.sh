#!/usr/bin/env bash
# Smoke-test Gazebo global detection via link_states / pose/info bridge.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$WS/scripts/setup_env.sh"

cleanup() {
  if [[ -n "${LAUNCH_PID:-}" ]]; then
    kill "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ros2 launch picking_bringup sim_gz_vibration.launch.py fine_perception:=gz auto_pick:=false &
LAUNCH_PID=$!

echo "Waiting for Gazebo pose/info ..."
for _ in $(seq 1 30); do
  if ros2 topic echo /world/blueberry_picking/pose/info --once 2>/dev/null | rg -q "x: 0.38"; then
    break
  fi
  sleep 1
done

echo "Waiting for trigger_global_detection ..."
for _ in $(seq 1 40); do
  if ros2 service list 2>/dev/null | rg -q '/trigger_global_detection$'; then
    break
  fi
  sleep 1
done

  sleep 12
OUT=$(ros2 service call /trigger_global_detection picking_msgs/srv/TriggerGlobalDetection "{}" 2>&1)
echo "$OUT"
echo "$OUT" | rg -q "success=True" || { echo "Global detection failed"; exit 1; }
echo "Global detection smoke test passed"
