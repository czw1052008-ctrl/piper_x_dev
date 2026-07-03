# Blueberry Picking System — Architecture (v1.0)

## 1. 目标与范围

| 项 | 决策 |
|----|------|
| 机械臂 | Piper X（`agx_arm_urdf`），固件 ≥ S-V1.6-3 |
| 末端 | 吸盘 / 振动棒（launch 参数切换） |
| 第一版 | Phase 0+1+2：BT + MoveIt mock 仿真已通；Gazebo Harmonic 场景可加载 |
| 感知 | 统一 `DetectedBerry[]`；仿真默认 `use_fake_perception:=true` |
| 真机 | 预留 `agx_arm_ros` 接口，第一版不启用 |

## 2. 包依赖图

```
picking_bringup
  ├── picking_description  (→ agx_arm_description)
  ├── picking_moveit_config
  ├── picking_perception   (→ picking_msgs, FoundationPose*)
  ├── picking_grasp        (→ picking_msgs, moveit)
  └── picking_task         (→ picking_msgs, behaviortree_cpp, moveit)

picking_msgs  ← 所有功能包
```

## 3. 坐标系

| Frame | 说明 |
|-------|------|
| `base_link` | Piper X 基座 |
| `link6` | 腕部法兰，手腕相机固定点 |
| `eef_link` | 吸盘接触面 / 振动棒尖端（MoveIt 规划目标） |
| `camera_wrist_color_optical_frame` | 手腕 RGB 光学系 |
| `camera_fixed_optical_frame` | 固定单目光学系（静态 TF） |

## 4. Topic / Service / Action

### Topics
| 名称 | 类型 | 发布者 |
|------|------|--------|
| `/camera_fixed/image_raw` | sensor_msgs/Image | Gazebo / v4l2 |
| `/camera_wrist/color/image_raw` | sensor_msgs/Image | Gazebo RGB-D |
| `/camera_wrist/depth/image_raw` | sensor_msgs/Image | Gazebo RGB-D |
| `/blueberry_cluster_pose` | geometry_msgs/PoseStamped | global_detector |
| `/detected_berries` | picking_msgs/DetectedBerry[] | fine_detector / fake |

### Services
| 名称 | 类型 | 说明 |
|------|------|------|
| `/trigger_global_detection` | TriggerGlobalDetection | 粗定位 |
| `/trigger_fine_detection` | TriggerFineDetection | 精细检测 |
| `/plan_suction` | PlanSuction (自定义 srv) | 吸盘规划 |
| `/plan_vibration` | PlanVibration (自定义 srv) | 振动规划 |

### Actions
| 名称 | 类型 |
|------|------|
| `/pick_blueberry` | PickBlueberry |

## 5. 仿真数据流

```
Gazebo World (桌面 + 蓝莓球 + 枝条)
    │
    ├─ RGB-D plugins → /camera_wrist/*
    ├─ fixed camera plugin → /camera_fixed/*
    └─ /gazebo/model_states
           │
    use_fake_perception=true (默认)
           │
    fake_perception_node ──→ TriggerFineDetection / TriggerGlobalDetection
           │
    picking_grasp (suction | vibration planner)
           │
    picking_task (BT) ──→ MoveIt move_group ──→ gazebo_ros2_control
```

`use_fake_perception=false` 时走 `global_detector_node` + `fine_detector_node` + `FoundationPoseWrapper`。

## 6. GPIO 设计（真机预留）

机械臂直连 PC，末端 DC 电机/气泵通过 **USB 继电器模块**（如 CH340 + 4路继电器）控制：

```
PC USB → USB-Relay Board → 继电器 NO/COM
                          ├─ CH1: 气泵 (吸盘模式)
                          └─ CH2: 振动电机 (振动模式)
```

仿真阶段：`FakeGPIOController` 仅打印日志，BT 用 `Wait` 节点代替吸附检测。

## 7. MoveIt 策略

- 规划组：`arm`（joint1–6）
- 笛卡尔运动：`computeCartesianPath`（兼容性好，仿真够用）
- 吸盘/振动 SRDF 分别定义 `eef_link` 为规划 tip

## 8. 测试策略

| 层级 | 内容 |
|------|------|
| pytest | perception 掩码分割、PCA 逻辑 |
| gtest | suction/vibration planner 单元测试 |
| launch | `sim_suction.launch.py` / `sim_vibration.launch.py` 端到端 |

## 9. 外部依赖

```bash
# ROS 2 Humble + MoveIt2 + Gazebo Classic
sudo apt install ros-humble-desktop ros-humble-moveit* \
  ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
  ros-humble-behaviortree-cpp ros-humble-ros2-control \
  ros-humble-ros2-controllers python3-colcon-common-extensions

# BehaviorTree.CPP v4（可选，高于 apt 版本时源码安装）
# git clone https://github.com/BehaviorTree/BehaviorTree.CPP -b 4.6.0

# FoundationPose 权重（首次使用需下载到 FoundationPose/weights/）
```

## 10. 已知限制（v1）

- `agx_arm_ros` 真机驱动未接入 bringup
- FoundationPose 需 GPU + 权重；无权重时 fine_detector 降级返回空结果
- 振动棒 PWM 未实现，仅开关
- 手眼标定使用占位 TF，需后续替换
