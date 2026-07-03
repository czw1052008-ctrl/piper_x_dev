# piper_x_dev

Piper X 蓝莓采摘真机开发环境单体仓库，包含主工作空间与全部本地依赖。

## 目录结构

| 目录 | 说明 | 上游 |
|------|------|------|
| `blueberry_picking_ws/` | 蓝莓采摘 ROS 2 主工作空间（感知、规划、真机脚本） | 本项目核心 |
| `agx_arm_ros/` | AgileX 机械臂 ROS 2 驱动与 MoveIt | [agilexrobotics/agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros) |
| `piper_ros/` | Piper 机械臂 ROS 1 包（历史参考） | [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros) |
| `pyAgxArm/` | Python CAN 控制库 | [agilexrobotics/pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) |
| `OrbbecSDK_ROS2_main/` | Orbbec 手腕相机 ROS 2 驱动（**真机使用**） | [orbbec/OrbbecSDK_ROS2](https://github.com/orbbec/OrbbecSDK_ROS2) |
| `OrbbecSDK_ROS2/` | Orbbec 驱动备用副本 | 同上 |
| `FoundationPose/` | 6D 姿态估计（可选感知后端） | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) |
| `agilex_open_class/` | AgileX 公开课资料 | [agilexrobotics/agilex_open_class](https://github.com/agilexrobotics/agilex_open_class) |

## 快速开始

```bash
# 1. 克隆本仓库
git clone https://github.com/czw1052008-ctrl/piper_x_dev.git ~/piper_x_dev
cd ~/piper_x_dev

# 2. 本地配置（不提交 git）
cp blueberry_picking_ws/config/real_robot.env.example \
   blueberry_picking_ws/config/real_robot.env
# 编辑 CAN、USB、相机等参数

# 3. 编译各工作空间
source /opt/ros/jazzy/setup.bash
cd agx_arm_ros && colcon build --symlink-install && source install/setup.bash
cd ../OrbbecSDK_ROS2_main && colcon build --symlink-install && source install/setup.bash
cd ../blueberry_picking_ws && bash scripts/setup_agx_arm.sh
colcon build --symlink-install && source install/setup.bash

# 4. 真机采摘
bash scripts/run_real_suction_pick.sh --move --once
```

## FoundationPose 权重

GitHub 单文件限制 100MB，以下权重需本地单独下载后放入 `FoundationPose/weights/`：

- `2024-01-11-20-02-45/model_best.pth`
- `2024-01-11-20-02-45-20260623T132207Z-3-001.zip`

参见 [FoundationPose 官方说明](https://github.com/NVlabs/FoundationPose)。

## 环境变量

`PIPER_X_DEV` 默认指向 `~/piper_x_dev`，各脚本通过 `config/real_robot.env` 引用子目录路径。

## 许可证

各子目录保留其上游开源许可证；`blueberry_picking_ws` 为本项目自有代码。
