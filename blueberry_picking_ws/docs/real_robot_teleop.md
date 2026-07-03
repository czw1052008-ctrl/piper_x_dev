# Piper X 真机键盘遥操 Bringup 指南

本文档记录 **WSL2 + Ubuntu 24.04 + ROS 2 Jazzy** 环境下，使用 `agx_arm_ros` 驱动 Piper X 真机，并运行 `picking_bringup` 键盘遥操的完整依赖与常见踩坑点。协作同事请按此流程操作，避免重复排障。

---

## 1. 环境与仓库依赖

| 组件 | 版本 / 说明 |
|------|-------------|
| 操作系统 | Windows + WSL2，发行版 **Ubuntu 24.04 (Noble)** |
| ROS 2 | **Jazzy**（24.04 不要用 Humble） |
| 机械臂驱动 | [`agx_arm_ros`](https://github.com/agilexrobotics/agx_arm_ros) 分支 `ros2` |
| 底层 SDK | [`pyAgxArm`](https://github.com/agilexrobotics/pyAgxArm)（须装到 **系统 Python**，见 §4） |
| 采摘工作区 | 本仓库 `blueberry_picking_ws` |
| USB-CAN | candleLight（示例 VID:PID `1d50:606f`），Windows 侧 `usbipd-win` 挂载到 WSL |

建议目录布局（与本文档示例一致）：

```text
~/piper_x_dev/
├── agx_arm_ros/          # git clone -b ros2
├── pyAgxArm/             # git clone
└── blueberry_picking_ws/ # 本仓库
```

USB-CAN 在 WSL2 中的挂载、驱动、`can0` 配置详见：

- [`pyAgxArm/docs/wsl2_usb_can_guide.md`](../../pyAgxArm/docs/wsl2_usb_can_guide.md)

---

## 2. 编译前必读：Conda 与 ROS Python 冲突

**每次打开新终端跑 ROS 前，先执行：**

```bash
conda deactivate
```

### 2.1 现象

- `colcon build` 失败，或 MoveIt / `resource_retriever` 链接 OpenSSL 报错
- `ros2 run` 找不到消息类型，或 Python 版本与已编译的 `agx_arm_msgs` 不一致（如 3.12 vs 3.13）
- 节点能起但 CAN / 控制无响应

### 2.2 原因

Conda 的 `Python`、`LD_LIBRARY_PATH`、`CMAKE_PREFIX_PATH` 会污染 ament 构建与运行时，导致扩展模块与 ROS 系统 Python 3.12 不匹配。

### 2.3 正确做法

1. **构建** `agx_arm_ros` 与 `blueberry_picking_ws` 前：`conda deactivate`
2. **推荐** 使用本仓库脚本（自动过滤 conda 路径并指定 `/usr/bin/python3`）：

   ```bash
   cd ~/piper_x_dev/blueberry_picking_ws
   bash scripts/colcon_build.sh
   ```

3. **pyAgxArm** 安装到系统 Python（示例）：

   ```bash
   conda deactivate
   /usr/bin/python3 -m pip install --user -e ~/piper_x_dev/pyAgxArm
   ```

4. 若曾用 conda Python 编过包，需 **删掉 `build/`、`install/`、`log/` 后干净重编**。

---

## 3. 编译 agx_arm_ros

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
cd ~/piper_x_dev/agx_arm_ros
colcon build --symlink-install
source install/setup.bash
```

注意：

- `agx_arm_moveit/scripts/agx_arm_control_gate` 须有可执行权限（`chmod +x`），否则 `auto_control_gate:=true` 时门控节点无法启动。
- 首次克隆后若 `ros2 launch` 报找不到包，确认已 `source install/setup.bash`。

---

## 4. 编译 blueberry_picking_ws

```bash
conda deactivate
cd ~/piper_x_dev/blueberry_picking_ws
bash scripts/colcon_build.sh
source install/setup.bash
```

遥操节点位于包 `picking_bringup`：`link6_teleop_node.py`。

---

## 4. 一键启动（推荐）

首次复制配置文件并按机器修改路径/型号：

```bash
cd ~/piper_x_dev/blueberry_picking_ws
cp config/real_robot.env.example config/real_robot.env
# 编辑 ORBBEC_LAUNCH（dabai_dcw2 / dabai_a 等）、USBIP_CAN_BUSID、相机安装位姿等
```

**每次真机开工一条命令**（自动：CAN 拉起 → 臂+MoveIt → joint_states relay → Orbbec → 相机 TF）：

```bash
conda deactivate
bash scripts/real_robot_bringup.sh --attach-usb
```

带键盘遥操（栈起来后前台进 teleop）：

```bash
bash scripts/real_robot_bringup.sh --attach-usb --teleop
```

带 FoundationPose 感知：

```bash
bash scripts/real_robot_bringup.sh --attach-usb --perception
```

收工（失能机械臂 + 停所有子进程）：

```bash
bash scripts/real_robot_shutdown.sh
```

仅遥操（栈已在跑）：

```bash
bash scripts/real_robot_teleop.sh
```

日志目录：`log/real_robot/`（`arm.log`、`camera.log`、`relay.log` 等）。

预检（不启动节点）：

```bash
bash scripts/real_robot_bringup.sh --check-only
```

---

## 5. 手动分终端启动（调试用）

以下假设 USB-CAN 已在 WSL 中表现为 `can0` 且 `ip link` 状态为 `UP`。
若已使用 §4 一键脚本，**可跳过本节**。

### 终端 0 — Windows 管理员 PowerShell（每次插拔 USB 后）

```powershell
usbipd list
usbipd bind --busid 1-5          # 首次需要 bind，BUSID 以 list 输出为准
usbipd attach --wsl --busid 1-5
```

WSL 内确认：

```bash
ip link show can0
# 期望: state UP
```

### 终端 1 — 驱动 + MoveIt + ros2_control

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
source ~/piper_x_dev/agx_arm_ros/install/setup.bash

ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py \
  can_port:=can0 \
  arm_type:=piper_x \
  effector_type:=none \
  auto_control_gate:=false \
  speed_percent:=20 \
  use_rviz:=false
```

说明：

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `auto_control_gate` | `false` | 遥操期控制链路常开；`true` 时仅 MoveIt 执行轨迹瞬间开门 |
| `speed_percent` | `20`（可按需调） | 真机首调建议低速 |
| `follow` | 默认 `true` | MoveIt 使用 `/feedback/joint_states` 作为当前状态 |

启动后检查：

```bash
ros2 node list | grep -E 'move_group|agx_arm'
ros2 action list | grep move_action    # 应有 /move_action
```

### 终端 2 — joint_states 中继（必须）

MoveIt / `robot_state_publisher` 默认订阅 `/joint_states`，而真机反馈在 `/feedback/joint_states`：

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
source ~/piper_x_dev/agx_arm_ros/install/setup.bash

ros2 run topic_tools relay /feedback/joint_states /joint_states
```

**漏开此 relay 时：** TF 不更新、IK 用错关节角、遥操“动了但 RViz/规划状态不对”。

### 终端 3 — 键盘遥操

```bash
conda deactivate
source /opt/ros/jazzy/setup.bash
source ~/piper_x_dev/blueberry_picking_ws/install/setup.bash

ros2 run picking_bringup link6_teleop_node --ros-args \
  -p linear_speed_m_s:=0.03 \
  -p angular_speed_deg_s:=10.0
```

焦点须在运行遥操的终端；按住键运动，松开即停。

---

## 6. 键盘映射

| 按键 | 功能 |
|------|------|
| W / S | 末端上下（`base_link` ±Z） |
| A / D | 左右（`base_link` ±Y） |
| Q / E | 前后（`base_link` ±X） |
| O / K | 滚转 roll（`link6` 工具系） |
| I / J | 俯仰 pitch（`link6`） |
| U / H | 偏航 yaw（`link6`） |
| `[` / `]` | 降低 / 提高线速度与角速度 |
| **R** | **MoveIt 规划并执行回 home**（六关节全 0） |
| Space | 切换振动（仅仿真 `vibration_motor_controller` 有效，真机无效） |
| `?` | 打印帮助 |
| Esc | 退出 |

参数 `linear_jog_frame` 默认 `base_link`；若改为 `link6`，平移会随末端姿态旋转（一般不建议）。

---

## 7. R 键回零：必须用 MoveIt，不要用 `/move_home`

### 7.1 控制链路说明

```text
键盘遥操 / MoveIt 回零
        ↓
  arm_controller (FollowJointTrajectory)
        ↓
  ros2_control → agx_arm_ctrl → CAN → 真机
```

`/move_home` 服务在 `agx_arm_ctrl` 内直接调用 `pyAgxArm.move_j()`，**绕过** `arm_controller`，与遥操 **抢控制权**。服务类型为 `std_srvs/Empty`，**无 success 字段**，失败也会静默返回。

因此：

- **遥操运行期间不要** `ros2 service call /move_home`
- **R 键** 通过 `/move_action`（`moveit_msgs/action/MoveGroup`）规划到 SRDF `home` 并执行，与 jog 走同一条 `arm_controller` 路径

### 7.2 R 键相关参数

| ROS 参数 | 默认值 | 说明 |
|----------|--------|------|
| `move_group_action` | `/move_action` | MoveIt move_group action |
| `home_joint_positions` | 六个 `0.0` | 与 `agx_arm.srdf` 中 `home` 一致 |
| `home_velocity_scale` | `0.15` | 回零速度比例，可调到 `0.25` |

成功日志示例：

```text
planning home joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] vel_scale=0.15
home move completed link6=(...)
```

失败时会打印 `home move failed code=...` 或 `move_group action not available`。

---

## 8. 常见问题

### 8.1 `can0` 不存在或 DOWN

- Windows 侧重新 `usbipd attach`
- WSL 内：`sudo ip link set can0 up type can bitrate 1000000`（波特率以硬件为准）
- 详见 [wsl2_usb_can_guide.md](../../pyAgxArm/docs/wsl2_usb_can_guide.md)

### 8.2 遥操无反应 / TF 超时

1. 终端 2 的 `joint_states` relay 是否在跑
2. `ros2 topic hz /feedback/joint_states` 是否有数据
3. `ros2 run tf2_ros tf2_echo base_link link6` 是否持续更新

### 8.3 IK 失败 `NO_IK_SOLUTION`

- 目标位姿超出工作空间或接近奇异点；先 jog 到中间区域再试
- 检查当前关节角是否已通过 relay 同步到 MoveIt

### 8.4 R 键不动但日志“成功”（旧版）

旧版曾调用 `/move_home` 或硬编码笛卡尔 home + IK，均已废弃。请更新代码并确认：

```bash
ros2 action list | grep move_action
```

### 8.5 机械臂振动 / 两个控制源同时写 CAN

- 确保只有一个 launch 实例
- 遥操松开键后应打印 `jog stop`；若仍抖，检查是否有其他节点发布 `control/joint_states`

### 8.6 使能相关（非遥操场景）

手动使能/示教模式（停遥操后）：

```bash
ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}"
ros2 service call /exit_teach_mode std_srvs/srv/Trigger "{}"
```

---

## 9. 诊断命令速查

```bash
# 节点与 action
ros2 node list | grep agx
ros2 action list | grep -E 'move_action|follow_joint_trajectory'

# 关节反馈
ros2 topic echo /feedback/joint_states --once
ros2 topic hz /feedback/joint_states

# TF
ros2 run tf2_ros tf2_echo base_link link6

# 控制器状态
ros2 control list_controllers
```

---

## 10. 与仿真环境的差异

| 项目 | 仿真 (Gazebo) | 真机 |
|------|---------------|------|
| 启动 launch | `sim_gz_*.launch.py` | `agx_arm_ctrl/start_single_agx_arm_moveit.launch.py` |
| joint_states | 自动生成 | 需 **relay** `/feedback/joint_states` → `/joint_states` |
| Space 振动 | 有效 | 无效（无 `vibration_motor_controller`） |
| USB-CAN | 不需要 | 必须 `usbipd` + `can0` |
| Conda | 仍建议 deactivate | **必须** deactivate |

---

## 11. 相关文档

- [blueberry_picking_ws README](../README.md) — 仿真分层启动
- [ARCHITECTURE.md](../ARCHITECTURE.md) — 系统架构
- [agx_arm_ros README](https://github.com/agilexrobotics/agx_arm_ros/blob/ros2/README.md) — 官方驱动与 MoveIt 参数
- [pyAgxArm WSL2 USB-CAN 指南](../../pyAgxArm/docs/wsl2_usb_can_guide.md)

---

*最后更新：2026-06-22 — 含 MoveIt `/move_action` 回零、joint_states relay、Conda 排障。*
