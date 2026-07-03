# Blueberry Picking Workspace

ROS 2 Humble workspace for Piper X blueberry picking (suction + vibration modes).

## Packages

| Package | Description |
|---------|-------------|
| `picking_msgs` | Messages, services, actions |
| `picking_description` | URDF, Gazebo world, scene models |
| `picking_moveit_config` | MoveIt / ros2_control config |
| `picking_perception` | Global/fine/fake perception + FoundationPose wrapper |
| `picking_grasp` | Suction & vibration planners |
| `picking_task` | BehaviorTree task coordinator |
| `picking_bringup` | Top-level launch files |

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design.

## Prerequisites

- **Ubuntu 22.04** → ROS 2 **Humble** + Gazebo Classic（与设计文档一致）
- **Ubuntu 24.04** → ROS 2 **Jazzy** + Gazebo Harmonic（本机 Noble 自动选此项）
- NVIDIA GPU（FoundationPose 可选；仿真 fake 模式不需要）
- `agx_arm_description`（运行 `bash scripts/setup_agx_arm.sh` 自动克隆并软链）

```bash
bash scripts/install_deps.sh   # 自动检测系统并配置 ROS apt 源
```

> 若在 24.04 上安装失败，是因为 **Humble 不支持 Noble**。脚本会改装 Jazzy。
> 完整 Gazebo + MoveIt 仿真见 README「Tier 3」；MoveIt-only 见「Tier 2」。

## Clone & dependencies

```bash
git clone https://github.com/<your-user>/blueberry_picking_ws.git
cd blueberry_picking_ws
bash scripts/setup_agx_arm.sh    # Piper X URDF meshes
bash scripts/install_deps.sh
```

## Build

```bash
bash scripts/colcon_build.sh          # 推荐；规避 conda Python 冲突
source install/setup.bash
```

`rosdep` **可选**（依赖已通过 apt 安装）。若 init 超时：

```bash
bash scripts/init_rosdep.sh           # 清华镜像
rosdep install --from-paths src --ignore-src -r -y || true
```

> 若使用 conda，构建前请 `conda deactivate`，或始终用 `scripts/colcon_build.sh`。

## Run (three tiers)

### Tier 1 — BT only (no MoveIt, no Gazebo)

```bash
ros2 launch picking_bringup sim_bt_only.launch.py
```

Static fake perception + stub motion. Fastest sanity check.

### Tier 2 — MoveIt + mock hardware (no Gazebo window)

```bash
ros2 launch picking_bringup sim_moveit_suction.launch.py
ros2 launch picking_bringup sim_moveit_vibration.launch.py
```

Real OMPL planning + `joint_trajectory_controller` on mock ros2_control. Auto-sends pick goal after ~30s.

> **Important:** Only run one launch at a time. Multiple `ros2_control_node` instances on `/controller_manager` will conflict.

### Tier 3 — Gazebo Harmonic + MoveIt (full sim)

```bash
ros2 launch picking_bringup sim_gz_suction.launch.py
ros2 launch picking_bringup sim_gz_vibration.launch.py
```

Aliases (same as Tier 3):

```bash
ros2 launch picking_bringup sim_suction.launch.py
ros2 launch picking_bringup sim_vibration.launch.py
```

WSL2 needs a working OpenGL/display for the gz GUI; use Tier 2 if the window fails to open.

## Manual pick goal

```bash
ros2 run picking_bringup send_pick_goal --ros-args \
  -p end_effector_mode:=suction -p max_retries:=3
```

## FoundationPose

1. Download weights into `/home/ziwei/piper_x_dev/FoundationPose/weights/` (see upstream readme)
2. Placeholder mesh: `picking_perception/meshes/blueberry.obj`
3. Set `use_fake_perception:=false` to use real perception pipeline

## Real robot teleop (Piper X + agx_arm_ros)

真机 **一键启动**：`cp config/real_robot.env.example config/real_robot.env` 后执行 `bash scripts/real_robot_bringup.sh --attach-usb`。  
完整说明、手动分终端流程、踩坑见 **[docs/real_robot_teleop.md](docs/real_robot_teleop.md)**。

## GPIO (future real robot)

USB relay module on PC controls pump (CH1) and vibration motor (CH2). See ARCHITECTURE.md §6.

## Tests

```bash
colcon test
colcon test-result --verbose
```
