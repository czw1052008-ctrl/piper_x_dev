# piper_x_dev

Piper X 蓝莓**触达**真机开发单体仓库（Ubuntu 22.04 + ROS 2 Humble）。

主能力：键盘遥操、固定单目锁定采集目标、手腕 RGB-D 精细触达、人工确认后 reset。  
详细约定见 [`blueberry_picking_ws/docs/REACH_PIPELINE.md`](blueberry_picking_ws/docs/REACH_PIPELINE.md)。

## 目录结构

| 目录 | 说明 |
|------|------|
| `blueberry_picking_ws/` | 感知、触达 FSM、遥操、真机脚本（核心） |
| `agx_arm_ros/` | AgileX 臂 ROS 2 驱动与 MoveIt |
| `pyAgxArm/` | Python CAN 控制库 |
| `OrbbecSDK_ROS2_main/` | **真机手腕相机驱动（SDK v1 / OpenNI，DaBai DC1）** |
| `OrbbecSDK_ROS2/` | Orbbec SDK v2 树（DC1 会 `No matched`，勿作主路径） |
| `FoundationPose/` | 可选 6D 后端（触达主路径默认关闭） |
| `archive/` | 已移出日常编译的冗余树（见 `docs/CLEANUP.md`） |

## 快速开始

```bash
# 1. 配置
cp blueberry_picking_ws/config/real_robot.env.example \
   blueberry_picking_ws/config/real_robot.env
# 编辑 PIPER_X_DEV（本机常为 /home/user/codes/piper_x_dev）、CAN、固定相机设备号

# 2. 编译
source /opt/ros/humble/setup.bash
cd agx_arm_ros && colcon build --symlink-install && source install/setup.bash
cd ../OrbbecSDK_ROS2_main && colcon build --symlink-install && source install/setup.bash
cd ../blueberry_picking_ws && bash scripts/setup_agx_arm.sh
colcon build --symlink-install && source install/setup.bash

# 3. 触达（默认 dry-run 可加 --dry-run；真机运动去掉）
bash scripts/run_real_reach.sh --dry-run
# 另开终端
ros2 topic pub --once /reach/cmd std_msgs/String "{data: start}"
# 触达完成后
ros2 topic pub --once /reach/cmd std_msgs/String "{data: confirm_reset}"

# 键盘遥操（可叠加相机/YOLO 可视化）
bash scripts/real_robot_teleop.sh --viz
```

## 真机配置要点

- 原生 Ubuntu：用本地 `can0`，**不要**再开 WSL `USBIP_*`。
- `FIXED_CAMERA_DEVICE`：4K USB 的 `/dev/videoX`（先用 `v4l2-ctl --list-devices` 确认）。
- 手腕相机：`ORBBEC_WS=.../OrbbecSDK_ROS2_main`；`ORBBEC_PUBLISH_TF=false`，由 bringup 发 `link6→camera_wrist_color_optical_frame`。
- 固定相机外参：`FIXED_CAM_*` / `calibration/fixed_camera_to_base.yaml`（当前为粗占位，需实测精调）。
- 遥操可视化：`bash scripts/real_robot_teleop.sh --viz`（固定 HSV + 手腕 YOLO + 深度伪彩）。
- **禁止**在臂使能时整卡 xHCI reset（见 `docs/REACH_PIPELINE.md`）。

## 许可证

各子目录保留上游许可证；`blueberry_picking_ws` 为本项目自有代码。
