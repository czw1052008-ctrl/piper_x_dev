---
name: piper-reach
description: >-
  Guides Piper X blueberry reach-pipeline work on Ubuntu 22.04 / ROS 2 Humble:
  keyboard teleop, fixed mono target lock, wrist RGB-D fine reach, topic-based
  FSM, human confirm reset. Use when editing reach/teleop/perception topics,
  fixed camera, Orbbec wrist, real_robot bringup, or blueberry touch-and-reset.
---

# Piper Reach Pipeline

## Before any code change

1. Read [docs/REACH_PIPELINE.md](../../../blueberry_picking_ws/docs/REACH_PIPELINE.md).
2. If continuing **ALIGNING / Gate A / agent set_joints**: read [docs/ALIGN_HANDOFF_2026-08-04.md](../../../blueberry_picking_ws/docs/ALIGN_HANDOFF_2026-08-04.md) first (current protocol; do not revive fixed whole_arm ±steps).
3. Read [docs/CLEANUP.md](../../../blueberry_picking_ws/docs/CLEANUP.md) if touching repo layout.
4. Prefer reusing existing nodes over new packages.

## Hard rules

- **Topics for task/perception** — do not add business services (`trigger_*`, `plan_*` for the real reach path).
- **Reach only** — suction tip to berry surface, publish reached, wait `confirm_reset`, home. No GPIO pump.
- **Camera roles** — fixed mono selects/locks target (`/perception/target_lock`); wrist RGB-D owns 3D for contact.
- **Orbbec DaBai DC1** — use **`OrbbecSDK_ROS2_main` (SDK v1 / OpenNI)**; v2 `No matched`. Keep `LD_LIBRARY_PATH` with `install/orbbec_camera/lib` (`libob_usb`, `liblive555`).
- **Never** whole-card xHCI reset while arm enabled — `real_robot_shutdown` first.
- **Single-track planning** — Python cup-axis plan → `/reach/plan`; do not revive C++ `plan_suction` + Python refine dual path for reach.
- **No drive-by refactors** — leave Gazebo/BT/vibration alone unless the user asks.
- **Do not edit** archived trees under `archive/` unless restoring them.

## Key paths

| Role | Path |
|------|------|
| Reach FSM | `blueberry_picking_ws/scripts/reach_fsm_node.py` |
| ALIGN judge | `scripts/align_judge.py` + `write_align_decision.py` + `run_align_qa.sh` |
| ALIGN handoff | `docs/ALIGN_HANDOFF_2026-08-04.md` |
| Global detector | `picking_perception/.../global_detector_node.py` |
| Fine detector | `picking_perception/.../fine_detector_node.py` |
| Teleop | `picking_bringup/.../link6_teleop_node.py` |
| Env | `blueberry_picking_ws/config/real_robot.env` (`ORBBEC_WS=.../OrbbecSDK_ROS2_main`) |
| Entry | `blueberry_picking_ws/scripts/run_real_reach.sh` |

## Cmd / status

- `/reach/cmd`: `start` | `confirm_reset` | `abort` | `clear_lock`
- `/reach/status`: `IDLE` | `LOCKING` | `ALIGNING` | `REFINING` | `PLANNING` | `APPROACHING` | `REACHED` | `WAIT_CONFIRM` | `RESETTING` | `ERROR`
- Stage smoke: `bash scripts/reach_stage_smoke.sh [0|1|2|3|4|all]`

## Hardware smoke (before claiming broken)

```bash
ip link show can0
lsusb | rg -i 'orbbec|1d50:606f|2bdf'
ros2 topic hz /camera_wrist/color/image_raw
ros2 topic hz /camera_fixed/image_raw
timeout 3 ros2 run tf2_ros tf2_echo link6 camera_wrist_color_optical_frame
```
