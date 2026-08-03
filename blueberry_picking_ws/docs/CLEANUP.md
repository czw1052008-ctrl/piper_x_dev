# 仓库清理与归档

## 已归档到 `archive/`（移出日常 colcon）

| 原路径 | 原因 |
|--------|------|
| `piper_ros/` | ROS 1 历史参考，真机用 `agx_arm_ros` |
| `OrbbecSDK_ROS2/` | 与 `OrbbecSDK_ROS2_main` 重复；真机只用 `_main` |
| `agilex_open_class/` | 教学资料，采摘/触达栈不引用 |

恢复：`mv archive/<name> .` 后按需编译。

## 保留（日常依赖）

- `blueberry_picking_ws/` — 主工作空间  
- `agx_arm_ros/` — 臂驱动 + MoveIt  
- `pyAgxArm/` — CAN SDK  
- `OrbbecSDK_ROS2/` — 手腕相机（v2 SDK，真机使用）  
- `OrbbecSDK_ROS2_main/` — 旧 SDK 1.x（保留但不编；Humble 缺 live555）  
- `FoundationPose/` — 可选；触达主路径默认关闭  

## 本阶段不删、不维护

仿真 Gazebo / BehaviorTree / 振动相关包与 launch：保留在 `blueberry_picking_ws`，不作为真机入口。

`picking_task` 在 Humble 下因 MoveIt 头文件路径差异暂加 `COLCON_IGNORE`（真机触达不依赖 BT）。需要仿真 BT 时再恢复并适配 `#include <moveit/move_group_interface/move_group_interface.h>`。

## 真机入口

| 用途 | 脚本 |
|------|------|
| 触达（主） | `scripts/run_real_reach.sh` |
| 键盘遥操 | `scripts/real_robot_teleop.sh` |
| 旧采摘环 | `scripts/run_real_suction_pick.sh`（deprecated，指向 reach 文档） |
