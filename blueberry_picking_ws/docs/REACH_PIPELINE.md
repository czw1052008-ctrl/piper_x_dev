# 蓝莓触达管线（Reach Pipeline）

**唯一批准主路径（三段）：**  
1. **LOCK_REGION（ALIGN 前）** — agent 看固定单目，锁**采集区域**（植株/簇）  
2. **ALIGNING** — agent `set_joints` / hold / `coarse_ok`（大方向）  
3. **FINE（腕部视觉伺服）** — 腕部 YOLO 锁果 + 测距闭环小步触达 → `WAIT_CONFIRM`

真机不做气泵/GPIO 采摘。任务与感知走 **topic**，不新增业务 service。

## 环境

| 项 | 值 |
|----|-----|
| OS | Ubuntu 22.04 |
| ROS | Humble |
| 臂 | Piper X via `can0` (gs_usb, 1 Mbps) |
| 手腕相机 | Orbbec DaBai (`camera_wrist`) |
| 固定相机 | 4K USB → `/camera_fixed/image_raw` |

## Topic 约定

| Topic | 类型 | 说明 |
|-------|------|------|
| `/camera_fixed/image_raw` | `sensor_msgs/Image` | 固定单目 |
| `/perception/global/berries` | `picking_msgs/DetectedBerryArray` | 全局候选（区域候选列表） |
| `/perception/target_lock` | `picking_msgs/DetectedBerry` | FSM 发布的区域锁 / 精调果锁 |
| `/perception/fine/berries` | `picking_msgs/DetectedBerryArray` | 手腕 YOLO+depth/mono ~2–5 Hz |
| `/reach/status` | `std_msgs/String` | 状态名 |
| `/reach/cmd` | `std_msgs/String` | `start` / `start_refine` / `confirm_near` / `confirm_reset` / `abort` / `clear_lock` |
| `/reach/reached` | `std_msgs/Bool` | 触达完成边沿 |

`/reach/plan` 仍可发布诊断，**不再**作为开环 cup_axis 触达主路径。

## 状态机（批准）

```
IDLE → LOCKING → ALIGNING → REFINING(腕部伺服) → REACHED → WAIT_CONFIRM → RESETTING → IDLE
```

| 阶段 | 行为 |
|------|------|
| `LOCKING` | `align_judge_mode=file`：**必须**等人/agent 写 `lock_region_decision.json`（`action=lock_region`, `index=N`）。禁止按 confidence 自动吞锁。启发式模式仅用于 dry-run/调试。 |
| `ALIGNING` | agent 估关节 → 位置控制 → hold → `coarse_ok`。见 [ALIGN_HANDOFF_2026-08-04.md](ALIGN_HANDOFF_2026-08-04.md)。 |
| `REFINING` | 中距：**闭环笛卡尔**（杯口→果 3D 路点）→ `/compute_ik` → 关节轨迹；图像误差相对杯口投影软门控步长（不硬切 0）。近距：`REFINING_WAIT_NEAR` → `confirm_near` 后 pose+单目触达。禁止开环整段 `cup_axis` contact。 |
| `confirm_reset` | 回 home 关节 |

**明确废除（不得复活）：**

- ALIGN 前按全局 YOLO 最高分自动锁  
- YOLO → HSV 冒充检测  
- 一次 `build_cup_axis_plan` + MoveIt `with_orient=False` 开环触达  
- refine timeout → 用粗锁/全局锁规划  
- 静默跳过 ALIGN / 伪造 `coarse_ok`

运动期间请停 teleop。FSM 用异步 MoveIt / FollowJointTrajectory（非阻塞 poll）。

### Agent 决策文件

Session 目录：`log/real_robot/qa/<session>/`

| 文件 | 阶段 |
|------|------|
| `lock_region_request.json` + QA 图 | LOCKING 写出 |
| `lock_region_decision.json` | agent：`write_lock_region_decision.py` |
| `align_*_request.json` / `align_decision.json` | ALIGNING：`write_align_decision.py` |
| `refine_fruit_lock.json` / `servo_XX.json` + `servo_XX_after_*.png` | REFINING 步进落盘 |
| 复盘 | `python3 scripts/refine_replay.py --qa-dir log/real_robot/qa/<session>` → `refine_tune_viz/replay_<session>/replay.html` |

```bash
python3 scripts/write_lock_region_decision.py \
  --session-dir log/real_robot/qa/<session> --index 0 \
  --reason 'fixed mono: target plant cluster'
```

### 阶段冒烟

```bash
bash scripts/reach_stage_smoke.sh 0
bash scripts/reach_stage_smoke.sh all
```

## 相机分工

1. **固定单目**：区域候选 + agent 锁采集区域（ALIGN 朝向用）。  
2. **手腕 RGB-D**：精调锁果；深度中值，失败时 **RGB 尺寸 mono**（保留）。主导触达判据。  
3. Tracker「偏靠下/近」偏好暂保留（后续另有选果逻辑）。

### 固定相机外参（卷尺粗标，2026-08-04）

`FIXED_CAM_TX/TY/TZ=(0.60,0.33,0.48)`；朝向按「斜下看工作区/植株」look-at 生成四元数（见 `fixed_camera_to_base.yaml`）。

## 启动

```bash
cp config/real_robot.env.example config/real_robot.env
# 编辑 PIPER_X_DEV、CAN、FIXED_CAMERA_DEVICE、ORBBEC_USB_PORT

source /opt/ros/humble/setup.bash
source ../agx_arm_ros/install/setup.bash
source ../OrbbecSDK_ROS2_main/install/setup.bash
export LD_LIBRARY_PATH="../OrbbecSDK_ROS2_main/install/orbbec_camera/lib:${LD_LIBRARY_PATH}"
source install/setup.bash
export PYTHONNOUSERSITE=1

# 触达管线（推荐 agent file 模式）
bash scripts/run_real_reach.sh --align-judge-mode file
# 或：bash scripts/_start_full_reach_agent.sh

ros2 topic pub --once /reach/cmd std_msgs/String "{data: start}"
# LOCKING：写 lock_region_decision.json
# ALIGNING：写 align_decision.json（set_joints / coarse_ok）
# 触达后：
ros2 topic pub --once /reach/cmd std_msgs/String "{data: confirm_reset}"
```

## 真机验收

- [ ] ALIGN 前必须等人写区域决策才进 ALIGN  
- [ ] YOLO 不可用时 fine 为空且不进触达（无 HSV 假果）  
- [ ] 精调过程腕部图果心逐步居中、距离下降；触达后停在 WAIT_CONFIRM；全程 j6≈0  
- [ ] 无「refine timeout → 粗锁规划」  

## Orbbec / 臂注意

DaBai DC1 走 **SDK v1 + `OrbbecSDK_ROS2_main`**。动 USB 前先 `real_robot_shutdown`；禁止臂使能时整卡 xHCI reset。  
`agx_arm_ctrl` 需系统侧 NumPy 1.24.x（`PYTHONNOUSERSITE=1`）。

## 明确不做

GPIO 吸盘、振动真机、BT 真机、FoundationPose 主路径、重写 `agx_arm_ros`、开环 cup_axis 触达、YOLO→HSV。
