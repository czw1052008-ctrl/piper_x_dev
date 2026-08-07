---
name: piper-reach
description: >-
  Guides Piper X blueberry reach-pipeline work on Ubuntu 22.04 / ROS 2 Humble:
  keyboard teleop, fixed mono region lock (agent), wrist RGB-D visual-servo
  reach, topic-based FSM, human confirm reset. Require offline QA replay on
  log/real_robot/qa before real-robot retests after refine/IK/aim changes. Use
  when editing reach/teleop/perception topics, fixed camera, Orbbec wrist,
  real_robot bringup, or blueberry touch-and-reset.
---

# Piper Reach Pipeline

## Before any code change

1. Read [docs/REACH_PIPELINE.md](../../../blueberry_picking_ws/docs/REACH_PIPELINE.md) — **唯一批准三段逻辑**.
2. If continuing **ALIGNING / Gate A / agent set_joints**: read [docs/ALIGN_HANDOFF_2026-08-04.md](../../../blueberry_picking_ws/docs/ALIGN_HANDOFF_2026-08-04.md).
3. Read [docs/CLEANUP.md](../../../blueberry_picking_ws/docs/CLEANUP.md) if touching repo layout.
4. Prefer reusing existing nodes over new packages.

## Offline QA before real robot (mandatory)

**任何**影响 REFINING / 居中 / 接触 / IK / 瞄准 / 内参外参 / 门控 的改动，必须先用历史数据验证，**通过后再上真机**。禁止「改完直接 `_run_refine_contact`」。

### 数据与做法

1. 选最近相关 session：`blueberry_picking_ws/log/real_robot/qa/<YYYYMMDD_HHMMSS>/`（至少含 `servo_*.json`、`mono_probe_*.json`；有腕图更好）。
2. 用改动后的公式/IK **离线重放**该 session 的冻结量（`berry_uv` / `berry_cam` / `cup_cam` / `berry_base` / 关节），对比改动前后：
   - aim UV、到点 `pix_off`、门控 PASS/FAIL
   - 若动 IK：模型预测 UV vs 记录的 `berry_uv_after`
3. 在回复里写清：**必要但不充分** 也要标明（例如只修正了 aim，执行轨迹仍超门限）。
4. **只有**离线显示指标朝预期方向改善（或证明无害且诊断成立）后，才跑真机；真机仍以固定相机 + 腕图为准，不单信数学 SUCCESS。

### 常用对照

| 检查 | 实机真值来源 |
|------|----------------|
| 腕内参 fx,fy,cx,cy | `/camera_wrist/color/camera_info`（勿默认 f=550、主点图心） |
| `T_link6_cam` | `tf2_echo link6 camera_wrist_color_optical_frame`（平移 `[0,-0.08,-0.04]` + **roll≈−23.5°** 光轴相对 j6+Z 下俯；`CAMERA_MOUNT_RPY`） |
| 杯轴 aim | `cx,cy + (fx,fy)*cup_cam_xy/z`（验收也用 cup-aim，不是光心） |
| 居中到点验收 | live 仅在 cup-aim `live_max`(~80px) 内采纳；采纳后用 **live UV ray** 更新 `berry_base`，残差交给 near oneshot 吸收，**禁止**因略超 `pix_tol`(35) 硬 ERROR；远框仍拒；无 live 则 reproject/aim 继续 |
| 居中到点图 | **无论成功/ERROR** 写 `center_oneshot_arrive_annotated.png`：绿=AIM、品红=REPROJECT(model)、青=LIVE@aim、黄=全部 YOLO。`yolo_miss` 时 after UV=reproject，叠图“准”只证明 IK 自洽，**不能**证明真果在 aim |
| 居中后 depth | fresh optical lock 已有 `berry_base`（mono/tri）→ **跳过**二次 contact mono probe，直接接触；sync 超时 re-pin 仅 UV/cup-aim 邻域，禁止无约束 cam-nearest |
| 居中控制 | oneshot **关节/EE 规划终点≈执行终点**（历史 session |ΔEE|≲0.5mm）；像面残差来自 **3D（mono 深度）≠真果**，不是跟踪没跟上。抬高是**规划**按错深度/满杯轴约束算出的 Δz，不是执行偏离。修正应改深度/闭环量测，禁止用 aim 比例凑抬升 |
| 接触 near | **优先** chunked `cup_axis_ik`（杯口 link6+Z 朝果）到 tip 接触。禁止 `keep_orient_chunked` 作主路（161209：tip 到了但杯口漂、腕图空）。相机 `approach_axis` 仅兜底（cam 偏 tip ~8cm）。实验路径：`--refine-skip-center` + `cup_contact_offset=0` → probe 后直接 oneshot 到果点（不做居中/近距伺服） |



| 居中标定 | probe 双视 baseline 常 ~1cm → tri 易拒；aim 用 `probe_mono_ray`（**禁止** soft-scale / overclimb_deepen 改深度）。`_obs_to_ray_base` 必须用 `camera_info` fx/fy/cx/cy。QA 写 `reject_reason` |
| 居中后 | **clear_lock** 清旧 ID → 等 live@cup-aim（`refine_post_center_live_wait_s`~1s @10Hz，空 1–2 帧正常）→ 新框 re-pin + live UV 重建 3D；残差大则二次 aim_uv_ik。超时无 aim 邻域 live 才退回 probe_mono_ray。禁止 soft-scale / overclimb deepen |


Replay 辅助：`scripts/refine_replay.py --qa-dir log/real_robot/qa/<session>`。

## Approved three-stage logic (do not deviate)

```
IDLE → LOCKING(agent region) → ALIGNING(agent joints) → REFINING(wrist servo) → WAIT_CONFIRM
```

1. **LOCKING (before ALIGN)** — agent on **fixed mono** picks a **collection region** (plant/cluster). File: `lock_region_decision.json` via `write_lock_region_decision.py`. **No** auto max-confidence lock.
2. **ALIGNING** — `set_joints` / hold / `coarse_ok` (coarse direction only).
3. **REFINING** — wrist YOLO mid-range center + approach; when `z_cam`/cup-dist ≤ `servo_near_handoff_z` (~0.12 m) **or wrist lost**, hand off to **last berry pose + `/feedback/tcp_pose` + fixed mono yaw** (camera≠cup). Contact when **cup↔berry** dist ≤ `cup_contact_offset` (~15 mm, soft cup on surface). **j6=0**.

**Keep:** mono depth when RGB-D invalid; `fine_max_age_s=0.5`; tracker lower/near preference (temporary); j6=0.

**Forbidden:** open-loop `build_cup_axis_plan` MoveIt contact; `with_orient=False` on contact path; YOLO→HSV fake berries; refine-timeout→coarse-lock plan; silent ALIGN skip / fake `coarse_ok`.

## Hard rules

- **Offline-before-hardware** — refine/servo/IK/aim/intrinsics/gate changes: validate on `log/real_robot/qa/<session>` first; only then real robot (see section above).
- **Topics for task/perception** — do not add business services for the real reach path.
- **Reach only** — tip to berry surface, publish reached, wait `confirm_reset`, home. No GPIO pump.
- **Camera roles** — fixed mono = region lock for ALIGN; wrist RGB-D owns fine berry + contact servo.
- **Orbbec DaBai DC1** — use **`OrbbecSDK_ROS2_main` (SDK v1 / OpenNI)**; keep `LD_LIBRARY_PATH` with `install/orbbec_camera/lib`.
- **Never** whole-card xHCI reset while arm enabled — `real_robot_shutdown` first.
- **No silent degradations** in reach FSM.
- **No drive-by refactors** — leave Gazebo/BT/vibration alone unless asked.
- **Do not edit** archived trees under `archive/` unless restoring them.

## Key paths

| Role | Path |
|------|------|
| Reach FSM | `blueberry_picking_ws/scripts/reach_fsm_node.py` |
| Region lock writer | `scripts/write_lock_region_decision.py` |
| ALIGN judge | `scripts/align_judge.py` + `write_align_decision.py` + `run_align_qa.sh` |
| ALIGN handoff | `docs/ALIGN_HANDOFF_2026-08-04.md` |
| Global detector | `picking_perception/.../global_detector_node.py` (no YOLO→HSV) |
| Fine detector | `picking_perception/.../fine_detector_node.py` (no YOLO→HSV) |
| Teleop | `picking_bringup/.../link6_teleop_node.py` |
| Env | `blueberry_picking_ws/config/real_robot.env` |
| Entry | `blueberry_picking_ws/scripts/run_real_reach.sh` / `_start_full_reach_agent.sh` |

## Cmd / status

- `/reach/cmd`: `start` | `confirm_reset` | `abort` | `clear_lock`
- `/reach/status`: `IDLE` | `LOCKING` | `ALIGNING` | `REFINING` | … | `WAIT_CONFIRM` | `ERROR`
  (`PLANNING`/`APPROACHING` are legacy; approved path must not enter them)
- Stage smoke: `bash scripts/reach_stage_smoke.sh [0|1|2|3|4|all]`

## Hardware smoke (before claiming broken)

```bash
ip link show can0
lsusb | rg -i 'orbbec|1d50:606f|2bdf'
ros2 topic hz /camera_wrist/color/image_raw
ros2 topic hz /camera_fixed/image_raw
timeout 3 ros2 run tf2_ros tf2_echo link6 camera_wrist_color_optical_frame
```
