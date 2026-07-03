# 蓝莓采摘机器人系统设计文档

## 1. 项目概述

| 项目 | 配置 |
|------|------|
| 机械臂 | 松灵 Piper 6轴 |
| 末端执行器 | **双模式可换装**：真空吸盘 / 振动棒 |
| 感知方案 | FoundationPose (BundleSDF 重建 mesh + 运行时 register/track) |
| 双相机 | 固定单目（全局定位）+ 手腕双目（精细对准） |
| 规划框架 | MoveIt 2 |
| 控制框架 | ros2_control |
| 任务协调 | BehaviorTree.CPP v4 |
| 开发环境 | ROS 2 Humble，C++/Python 混合 |
| 开发阶段 | 先 Gazebo 仿真验证，再上真机 |

---

## 2. 两种末端执行器方案对比

### 2.1 方案一：真空吸盘

- **动作原理**：吸盘直接接触单颗蓝莓表面，气泵开启产生负压，将果实从枝条上拉离
- **控制接口**：GPIO 控制气泵继电器（开/关）
- **感知要求**：需要单颗蓝莓的精确 6D 位姿（精度 ±3mm）
- **适用场景**：蓝莓分布稀疏、单颗采摘、对果实损伤要求低
- **优点**：采摘目标明确，每次采一颗，损失率低
- **缺点**：效率低（一次一颗），对小球面的吸附可靠性受表面水分/角度影响

### 2.2 方案二：振动棒（批量采摘）

- **动作原理**：振动棒将枝条卡入两颗螺栓形成的槽中，往返电机产生振动，使成熟蓝莓因惯性脱落；调节转速可选择性振落熟果、保留青果
- **控制接口**：GPIO 控制往返电机继电器（开/关）；PWM 或调速器控制转速
- **感知要求**：蓝莓簇 3D 位置 + 枝条轴向（由蓝莓位置 PCA 推算，无需单独感知枝条）
- **适用场景**：蓝莓成簇生长、批量采摘、追求效率
- **优点**：一次振动一簇（5~20颗），效率远高于吸盘；频率可调实现选择性采摘
- **缺点**：果实脱落后收集是独立问题（本文档暂不涉及）；不能单颗精选

### 2.3 振动棒物理结构

```
振动棒结构（棒轴向右，螺栓穿透棒身两侧对称伸出）：

        ┃bolt1    ┃bolt2
[tip]━━━╋━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━[motor/法兰]
  0    2cm      6cm      ...         30cm

└── 2cm ──┘└──── 4cm 卡槽 ────┘
     tip区        枝条工作区
```

**卡槽工作原理：**
- 螺栓穿透棒身，两侧各伸出约 1~2cm，与棒身一起形成对枝条的限位
- 枝条从侧面进入两螺栓之间的 **4cm 卡槽**（枝条直径约 5~12mm，4cm 槽宽容错充裕）
- 往返电机沿棒轴方向振动，两颗螺栓交替推动枝条，使果实受惯性力脱落
- 转速控制振动频率，恰当频率下：成熟果脱落（果柄连接力低），青果保留

**接近运动几何（俯视）：**
```
枝条（横截面 ●，枝条轴向纵深）：

接近前（棒在枝条外侧）    接近后（枝条进入卡槽）

  ┃     ┃                    ┃  ●  ┃
  ╋━━━━━╋━━━[motor]    →     ╋━━━━━╋━━━[motor]
bolt1  bolt2              bolt1  bolt2

← 棒横向平移（⊥棒轴，⊥枝条轴）→
```

关键约束：进槽前，**枝条必须已经在棒的 2~6cm 段的侧面位置**（沿棒轴方向），才能在横向平移时正确入槽。这个定位精度（±1.5cm）由蓝莓簇 PCA 感知保证。

---

## 3. 系统架构

### 3.1 统一感知架构设计原则

**两种模式使用完全相同的感知流水线，差异仅在规划层。**

```
               ┌─────────────────────────────────────┐
               │         统一感知流水线                │
               │                                     │
               │  固定单目 → 蓝莓簇粗定位              │  ← 两种模式共用
               │       ↓                             │
               │  手腕双目 + FoundationPose            │  ← 两种模式共用
               │  → DetectedBerry[] (蓝莓位姿数组)     │    同一推理引擎
               │       ↓                             │    同一参考图片
               └──────────┬──────────────────────────┘
                          │ 输出: DetectedBerry[]
              ┌───────────┴───────────┐
              ↓                       ↓
     [吸盘规划器]               [振动规划器]
     取最佳单颗蓝莓               对所有蓝莓位置做PCA
     → 接触点在果实顶部          → 枝条方向 + 侧压接触点
```

**为什么两种模式可以共用相同的 FoundationPose：**
- 吸盘模式：取置信度最高的单颗蓝莓 6D 位姿，规划吸附点
- 振动模式：取检测到的全部蓝莓位置，做 PCA 估算枝条方向，规划棒尖侧压点
- 感知包输出格式相同（`DetectedBerry[]`），规划包各自解读

### 3.2 整体数据流（含模式切换）

```
                        launch 参数: end_effector_mode = "suction" | "vibration"
                                            │
固定单目相机                                 ▼
  └─→ [全局检测] 蓝莓簇粗3D位置 ──────→ [任务协调 BT]
                                      (按模式加载不同BT树)
手腕双目相机（靠近后激活）                    │
  └─→ [统一精细感知节点]              ┌────┴────┐
       FoundationPose 检测蓝莓        │         │
       → DetectedBerry[]            │ 吸盘BT   │  振动BT
       (两种模式相同输出)             │         │
                                    └────┬────┘
                                         │
                              ┌──────────┴──────────┐
                              ↓                      ↓
                        [吸盘规划器]           [振动规划器]
                        单颗接触点             枝条侧压点
                              └──────────┬──────────┘
                                         ↓
                                   MoveGroupInterface
                                         │
                                   ros2_control
                                   ├─ arm_controller
                                   ├─ suction_controller (气泵GPIO)
                                   └─ vibration_controller (电机GPIO)
```

### 3.2 双相机分工

| | 固定单目相机 | 手腕双目相机 |
|---|---|---|
| **位置** | 固定在环境支架 | 安装在 Piper 腕部（link6/7） |
| **吸盘模式作用** | 全局找蓝莓簇，引导机械臂到预观测位姿 | FoundationPose 精确单颗6D位姿 |
| **振动模式作用** | 全局找蓝莓簇，引导机械臂靠近 | FoundationPose 检测多颗蓝莓 → PCA 推算枝条方向 → 棒尖侧压接触点 |
| **型号** | 普通 USB RGB 单目（品牌不限） | Orbbec Dabai（结构光，直接输出对齐 RGB+Depth） |
| **ROS 2 驱动** | `v4l2_camera` 或 `usb_cam` | `OrbbecSDK_ROS2`（`depth_align:=true`） |
| **关键 topics** | `/camera_fixed/image_raw` | `/camera_wrist/color/image_raw`、`/camera_wrist/depth/image_raw` |
| **深度获取** | 无（单目，只用于全局检测） | 驱动直接输出对齐深度图（uint16 mm），**无需 stereo_image_proc** |
| **坐标系** | `camera_fixed_frame` → `base_link` | `camera_wrist_color_optical_frame` → `ee_link` → `base_link` |
| **标定** | Eye-to-hand | Eye-in-hand（随机械臂TF树传播） |

---

## 4. 包结构

```
blueberry_picking_ws/
└── src/
    ├── picking_msgs/            # 消息/服务/Action 接口
    ├── picking_description/     # URDF（双模式）、meshes、标定结果
    ├── picking_moveit_config/   # MoveIt 配置（双EEF）
    ├── picking_perception/      # 感知节点（Python）
    ├── picking_grasp/           # 抓取/振动规划（C++）
    ├── picking_task/            # BehaviorTree 任务协调（C++）
    └── picking_bringup/         # 顶层 launch 文件
```

---
## 5. 各包详细设计

### 5.1 `picking_msgs` — 接口定义

**优先建立，其他所有包依赖它。感知输出格式对两种模式完全统一。**

```
picking_msgs/
├── msg/
│   ├── DetectedBerry.msg       # 单颗蓝莓检测结果（两种模式共用输出）
│   ├── SuctionGraspPlan.msg    # 吸盘规划结果
│   └── VibrationPlan.msg       # 振动规划结果
├── srv/
│   ├── TriggerGlobalDetection.srv  # 触发全局粗检测（两种模式共用）
│   └── TriggerFineDetection.srv    # 触发精细感知（两种模式共用同一服务）
└── action/
    └── PickBlueberry.action        # 顶层采摘 Action（两种模式共用）
```

**关键消息定义：**

```
# DetectedBerry.msg  ← 两种模式共用，感知层统一输出格式
std_msgs/Header header
geometry_msgs/PoseStamped pose   # 单颗蓝莓 6D 位姿（base_link frame）
float32 confidence               # 置信度 0~1
```

```
# TriggerFineDetection.srv  ← 两种模式共用同一服务接口
---
DetectedBerry[] detected_berries  # 检测到的全部蓝莓（0到多颗）
bool success
string message
```

```
# VibrationPlan.msg
std_msgs/Header header
geometry_msgs/PoseStamped pre_approach   # 枝条侧方 25cm，姿态已对齐（棒轴∥枝条）
geometry_msgs/PoseStamped slot_pose      # 横向推入后，枝条位于 2~6cm 卡槽内
geometry_msgs/PoseStamped retract_pose   # 振动结束后原路退出（同 pre_approach）
geometry_msgs/Vector3 branch_dir         # 估算的枝条方向（调试/可视化用）
float32 vibration_duration_sec
bool pca_reliable                        # PCA 是否可靠（false 说明使用了降级估算）
```

```
# PickBlueberry.action
# Goal
string end_effector_mode    # "suction" 或 "vibration"
int32 max_retries
---
# Result
bool success
string message
int32 total_picked
---
# Feedback
string state
geometry_msgs/PoseStamped current_target
```

---

### 5.2 `picking_description` — 机器人描述

```
picking_description/
├── urdf/
│   ├── piper_base.urdf.xacro           # Piper 6轴臂本体（不含末端）
│   ├── suction_eef.urdf.xacro          # 吸盘末端（含气管接口link）
│   ├── vibration_eef.urdf.xacro        # 振动棒+漏斗末端（含集果漏斗link）
│   ├── piper_with_suction.urdf.xacro   # 组合：臂 + 吸盘
│   └── piper_with_vibration.urdf.xacro # 组合：臂 + 振动棒漏斗
├── meshes/
│   ├── suction_cup.STL
│   └── vibration_rod.STL
├── calibration/
│   ├── fixed_camera_to_base.yaml       # 固定单目 eye-to-hand 标定结果
│   └── wrist_camera_to_ee.yaml         # 手腕双目 eye-in-hand 标定结果
└── launch/
    ├── display_suction.launch.py
    └── display_vibration.launch.py
```

**末端关键坐标系定义：**

| 坐标系名 | 吸盘模式 | 振动模式 |
|---------|---------|---------|
| `eef_link` | 吸盘接触面中心（MoveIt规划目标点） | 振动棒尖端（接触枝条的点） |
| `funnel_center` | 不存在 | 漏斗口几何中心（对准蓝莓簇时的参考点） |
| `camera_wrist_link` | 固连在 link6，两种模式相同 | 同左 |

---

### 5.3 `picking_moveit_config` — MoveIt 配置

```
picking_moveit_config/
├── config/
│   ├── kinematics.yaml                  # KDL（可选 TRAC-IK 提升成功率）
│   ├── joint_limits.yaml
│   ├── moveit_controllers.yaml          # FollowJointTrajectory
│   ├── ros2_controllers.yaml            # 含气泵 + 电机 GPIO 控制器
│   ├── piper_suction.srdf               # 吸盘模式 SRDF
│   └── piper_vibration.srdf             # 振动模式 SRDF
└── launch/
    ├── move_group_suction.launch.py
    └── move_group_vibration.launch.py
```

**两个 SRDF 的差异：**
```xml
<!-- piper_suction.srdf -->
<end_effector name="suction_eef"
              parent_link="link8"
              parent_group="arm" />

<!-- piper_vibration.srdf -->
<end_effector name="vibration_eef"
              parent_link="link8"
              parent_group="arm" />
<!-- 漏斗 link 加入 arm 规划组的碰撞检查 -->
```

**`ros2_controllers.yaml` 末端执行器控制器：**
```yaml
# 两种模式都是 GPIO 开关控制，接口一致
suction_controller:
  type: gpio_controllers/GpioCommandController
  ros__parameters:
    gpios: ['pump_relay']
    command_interfaces: ['pump_relay/cmd']
    state_interfaces:  ['pump_relay/state']

vibration_controller:
  type: gpio_controllers/GpioCommandController
  ros__parameters:
    gpios: ['motor_relay']
    command_interfaces: ['motor_relay/cmd']
    state_interfaces:  ['motor_relay/state']
```

> **设计要点**：两种模式下末端控制器接口类型完全一致（GPIO on/off），代码层面只是 topic 名不同，便于统一管理。

---
### 5.4 `picking_perception` — 感知节点（Python）

**核心设计：两种模式使用同一个精细感知节点，输出相同格式（`DetectedBerry[]`）。**

```
picking_perception/
├── picking_perception/
│   ├── global_detector_node.py      # 固定单目：蓝莓簇粗定位（两种模式共用）
│   ├── fine_detector_node.py        # 手腕双目 + FoundationPose（两种模式共用）
│   └── foundation_pose_wrapper.py   # FoundationPose 推理封装（见 §6.4）
├── meshes/
│   └── blueberry.obj                # ← BundleSDF 离线生成，运行时必须存在
├── ref_data/
│   └── blueberry/                   # BundleSDF 训练数据（离线用，不随包部署）
│       ├── rgb/
│       ├── depth/
│       ├── mask/
│       └── K.txt
├── config/
│   ├── global_detector.yaml
│   └── foundation_pose.yaml         # mesh_path / iter / score_threshold
├── setup.py
└── launch/
    └── perception.launch.py         # 两种模式共用同一个 launch
```

#### 全局检测节点（两种模式共用）

```python
class GlobalDetectorNode(Node):
    """固定单目 → 蓝莓簇粗 3D 定位
    输入: /camera_fixed/image_raw
    输出: /blueberry_cluster_pose (PoseStamped, base_link frame)
    方法: HSV 颜色分割（蓝紫色范围）或轻量 YOLO
    深度估计: 利用蓝莓直径先验 ~15mm 反推距离（误差 ±5cm，够引导机械臂靠近）
    """
```

#### 统一精细感知节点（两种模式共用同一节点）

> **重要**：FoundationPose 没有"直接用参考图片推理"的模式。所谓 model-free 实际是先用 **BundleSDF** 对参考视频做一次离线 NeRF 训练，生成 `blueberry.obj`，之后运行时才能调用标准推理 API。详见 §6.5。

```python
# fine_detector_node.py
class FineDetectorNode(Node):
    """Orbbec Dabai（手腕）+ FoundationPose → DetectedBerry[]

    Orbbec Dabai 结构光相机直接输出对齐的 RGB+Depth，
    无需 stereo_image_proc，订阅驱动发布的话题即可。
    
    输入:  /camera_wrist/color/image_raw      (BGR8, Orbbec驱动)
           /camera_wrist/depth/image_raw      (16UC1, mm, Orbbec驱动对齐输出)
           /camera_wrist/color/camera_info    (内参)
    输出:  /detected_berries (DetectedBerry[], base_link frame)
    触发:  TriggerFineDetection 服务（按需推理，避免持续占用 GPU）
    """
    def __init__(self):
        super().__init__('fine_detector_node')
        self.fp = FoundationPoseWrapper(
            mesh_path=self.get_parameter('mesh_path').value,
            est_refine_iter=5,
            track_refine_iter=2,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.srv = self.create_service(
            TriggerFineDetection, 'trigger_fine_detection', self.detect_cb)

        # Orbbec Dabai 驱动话题（启动时设置 depth_align:=true 保证深度与彩色对齐）
        self.create_subscription(Image, '/camera_wrist/color/image_raw',  self._rgb_cb,   1)
        self.create_subscription(Image, '/camera_wrist/depth/image_raw',  self._depth_cb, 1)
        self.create_subscription(CameraInfo, '/camera_wrist/color/camera_info', self._info_cb, 1)

    def detect_cb(self, request, response):
        rgb   = self.latest_rgb                          # (H,W,3) uint8, BGR→RGB
        depth = self.latest_depth.astype(np.float32) / 1000.0  # uint16 mm → float32 m
        # Orbbec Dabai 输出 uint16 深度（单位 mm），FoundationPose 需要 float32（单位 m）
        K     = self.camera_K      # (3,3)

        # 1. 用 HSV 分割生成蓝莓候选掩码（register() 必须的输入）
        mask = segment_blueberry_hsv(rgb)   # 见下方

        if mask.sum() < 100:               # 掩码太小，说明没有蓝莓
            response.success = False
            return response

        # 2. 调用 FoundationPoseWrapper 推理
        #    返回视野内每个连通分量（每颗蓝莓）的 4x4 pose + score
        detections = self.fp.detect_all(rgb, depth, K, mask)

        # 3. 坐标系转换: camera_wrist_optical_frame → base_link
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_wrist_optical_frame', rclpy.time.Time())
        except Exception:
            response.success = False
            return response

        T_cam_to_base = transform_to_matrix(tf)   # (4,4)

        response.detected_berries = []
        for pose_cam, score in detections:
            pose_base = T_cam_to_base @ pose_cam   # 变换到 base_link
            berry = DetectedBerry()
            berry.header.frame_id = 'base_link'
            berry.pose = matrix_to_pose_stamped(pose_base)
            berry.confidence = float(score)
            response.detected_berries.append(berry)

        response.success = True
        return response
```

**蓝莓 HSV 掩码生成（`register()` 所需）：**

```python
def segment_blueberry_hsv(rgb: np.ndarray) -> np.ndarray:
    """返回 (H,W) uint8 掩码，蓝莓区域为 1"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # 蓝莓颜色：蓝紫色，H≈120~160, S≥50, V≥30
    lo = np.array([110, 50, 30])
    hi = np.array([165, 255, 200])
    mask = cv2.inRange(hsv, lo, hi)
    # 形态学去噪 + 填洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask > 0).astype(np.uint8)
```

```yaml
# foundation_pose.yaml
mesh_path: "$(find picking_perception)/meshes/blueberry.obj"
est_refine_iter: 5          # 首帧 register() 细化迭代次数（精度高，慢）
track_refine_iter: 2         # 跟踪 track_one() 迭代次数（精度适中，快）
score_threshold: 0.3         # 低于此分数的检测丢弃
max_detections: 8            # 单帧最多返回蓝莓数
```

> 感知节点本身不关心模式，只负责"返回视野内所有蓝莓位姿"。模式差异完全由下游规划器处理。

---

### 5.5 `picking_grasp` — 执行规划（C++）

```
picking_grasp/
├── src/
│   ├── grasp_planner_node.cpp       # 统一规划节点入口，按模式路由
│   ├── suction_planner.cpp          # 吸盘模式：从单颗位姿计算接触点
│   └── vibration_planner.cpp        # 振动模式：从蓝莓位姿数组推算枝条侧压点
├── include/picking_grasp/
│   ├── suction_planner.hpp
│   └── vibration_planner.hpp
├── config/
│   ├── suction_params.yaml
│   └── vibration_params.yaml
└── launch/
    └── grasp_planner.launch.py      # 通过参数切换模式
```

#### 吸盘规划逻辑

```cpp
// 输入：DetectedBerry[]（取置信度最高的单颗）
// 输出：SuctionGraspPlan（pre_grasp / grasp / post_grasp）

// 选最优蓝莓：置信度最高 & 位置可达
DetectedBerry best = selectBestBerry(berries);

Vector3 approach = {0, 0, 1};              // 从正上方接近
Point3  contact  = best.position           // 接触点：果实顶部
                   + BERRY_RADIUS * approach;

plan.grasp      = contact;
plan.pre_grasp  = contact + 0.15 * approach;   // 安全距离 15cm
plan.post_grasp = contact + 0.12 * approach;   // 采摘后提升
```

#### 振动规划逻辑（核心细化）

振动棒工作方式：**棒轴垂直于枝条，从枝条侧面横向推入，枝条滑入 2~6cm 卡槽，沿棒轴往返振动**。

**三步坐标计算：**

**第一步：从蓝莓位置 PCA 估算枝条方向**

蓝莓在枝条上串状排列，多颗蓝莓的坐标分布沿枝条方向最"细长"，PCA 第一主成分即枝条轴向：

```cpp
// 输入: DetectedBerry[]
std::vector<Eigen::Vector3d> pos;
for (auto& b : berries) pos.push_back(toEigen(b.pose.position));
Eigen::Vector3d centroid = mean(pos);

Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
for (auto& p : pos) { auto d = p - centroid; cov += d * d.transpose(); }
cov /= pos.size();

Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
// 特征值升序排列，最大特征值(col(2))对应枝条方向
Eigen::Vector3d branch_dir = solver.eigenvectors().col(2).normalized();

// 可靠性：最大/次大特征值之比 > 2 说明分布够线形
bool pca_reliable = solver.eigenvalues()(2) / solver.eigenvalues()(1)
                    > params.pca_elongation_threshold;
if (!pca_reliable || pos.size() < params.min_berries_for_pca) {
    branch_dir = fallbackBranchDir(berries[0]);  // 降级：用单颗位姿Z轴
}
```

**第二步：确定棒的目标姿态和槽对齐位置**

目标：棒轴平行于枝条方向（振动沿棒轴传递），槽的 4cm 中心点对齐簇质心。

```cpp
// 棒轴对齐枝条方向
// 槽中心（棒的 4cm 处 = tip 前方 4cm）对齐簇质心
// → rod_tip_target = centroid - 4cm * branch_dir（沿棒轴退 4cm 到尖端位置）
// 注意：branch_dir 需指向从 tip 到 motor 的方向，否则取反
if (branch_dir.dot(motor_side_dir) < 0) branch_dir = -branch_dir;
Eigen::Vector3d tip_target = centroid - 0.04 * branch_dir;

// 接近方向：从机械臂外侧横向推入，垂直于棒轴（=枝条轴），取水平分量
Eigen::Vector3d to_arm   = (base_origin - centroid).normalized();
Eigen::Vector3d approach = (to_arm - to_arm.dot(branch_dir) * branch_dir).normalized();
// approach 垂直于枝条，朝向机械臂，即横向推入方向

// 棒的姿态：棒轴 = branch_dir，接近面朝 -approach
Quaternion rod_orientation = frameFromAxes(branch_dir, approach.cross(branch_dir));
```

**第三步：生成三段运动位姿**

```cpp
// 预接近：棒在枝条侧方 25cm，姿态已对齐，尚未接触
plan.pre_approach.position    = tip_target + 0.25 * approach;
plan.pre_approach.orientation = rod_orientation;

// 入槽：横向推入直到枝条进槽（枝条在 2~6cm 卡槽内）
// tip_target 在 centroid 后方 4cm（沿棒轴），此时 centroid 正好在槽中心
plan.slot_pose.position    = tip_target;
plan.slot_pose.orientation = rod_orientation;
// 实际执行时用笛卡尔路径，可配合力控检测入槽（电机电流突增）

// 退出：原路横向退出
plan.retract_pose = plan.pre_approach;
```

**降级策略（仅检测到 1~2 颗蓝莓时）：**

```cpp
Eigen::Vector3d fallbackBranchDir(const DetectedBerry& b) {
    // 假设枝条水平：取单颗蓝莓位姿 Z 轴投影到水平面
    Eigen::Quaterniond q = toEigen(b.pose.orientation);
    Eigen::Vector3d z_axis = q * Eigen::Vector3d::UnitZ();
    z_axis.z() = 0;
    return z_axis.norm() > 0.1 ? z_axis.normalized()
                                : Eigen::Vector3d::UnitX();  // 兜底
}
```

```yaml
# vibration_params.yaml
slot_center_from_tip: 0.04    # 槽中心在 tip 前方 4cm（bolt1=2cm, bolt2=6cm 中点）
pre_approach_dist: 0.25       # 预接近横向距离 25cm
vibration_duration_sec: 3.0   # 振动时长（初始值，实际按转速调整）
min_berries_for_pca: 3        # PCA 可靠所需最少蓝莓数
pca_elongation_threshold: 2.0 # PCA 线形可靠性阈值
```

**为什么不需要单独感知枝条：**

| 定位需求 | 精度要求 | 蓝莓 PCA 精度 | 是否满足 |
|---------|---------|-------------|---------|
| 沿棒轴：枝条在 2~6cm 槽内 | ±2cm | ~±1cm | ✅ |
| 垂直棒轴：横向接近距离 | 不需精确，推到接触即停 | — | ✅ |
| 棒轴对齐枝条方向 | ±15° 以内 | PCA ±10° | ✅ |

蓝莓果梗极短（~5mm），果簇质心与枝条位置几乎重合，**感知层无需额外增加枝条检测**，统一 FoundationPose 输出（`DetectedBerry[]`）即可驱动振动规划。

---
### 5.6 `picking_task` — BehaviorTree 任务协调（C++）

```
picking_task/
├── src/
│   ├── pick_action_server.cpp          # 顶层 Action Server，按模式加载 BT
│   └── bt_nodes/
│       ├── common/
│       │   ├── global_detect_node.cpp
│       │   ├── move_to_pose_node.cpp   # 通用：MoveIt 关节空间运动
│       │   └── cartesian_move_node.cpp # 通用：笛卡尔直线运动
│       ├── suction/
│       │   ├── fine_detect_node.cpp
│       │   ├── plan_suction_node.cpp
│       │   ├── activate_suction_node.cpp
│       │   └── check_suction_node.cpp
│       └── vibration/
│           ├── cluster_analyze_node.cpp
│           ├── plan_vibration_node.cpp
│           ├── activate_vibration_node.cpp
│           └── dump_berries_node.cpp
├── bt_xml/
│   ├── pick_suction.xml            # 吸盘模式 BT
│   └── pick_vibration.xml          # 振动模式 BT
├── config/
│   └── task_params.yaml
└── launch/
    └── task_server.launch.py
```

**模式切换（`pick_action_server.cpp`）：**
```cpp
// 根据 action goal 中的 end_effector_mode 参数加载对应 BT
if (goal->end_effector_mode == "suction") {
    tree_ = factory_.createTreeFromFile("pick_suction.xml", blackboard_);
} else {
    tree_ = factory_.createTreeFromFile("pick_vibration.xml", blackboard_);
}
```

#### 吸盘模式 BT

```xml
<!-- pick_suction.xml -->
<BehaviorTree ID="PickSuction">
  <RetryUntilSuccessful num_attempts="{max_retries}">
    <Sequence>
      <!-- 阶段1：全局粗定位 -->
      <GlobalDetect cluster_pose="{cluster_pose}" />
      <MoveToPose target="{cluster_pose}" offset_z="0.30" velocity_scale="0.6" />

      <!-- 阶段2：精细感知 -->
      <RetryUntilSuccessful num_attempts="3">
        <FineDetect precise_pose="{berry_pose}" />
      </RetryUntilSuccessful>

      <!-- 阶段3：规划 + 执行抓取 -->
      <PlanSuction berry_pose="{berry_pose}" plan="{suction_plan}" />
      <MoveToPose target="{suction_plan.pre_grasp}" velocity_scale="0.5" />
      <CartesianMove target="{suction_plan.grasp}" velocity_scale="0.2" />

      <!-- 阶段4：吸附 + 验证 -->
      <ActivateSuction enable="true" />
      <Wait msec="300" />
      <CheckSuction min_pressure="0.5" />  <!-- 失败触发外层重试 -->

      <!-- 阶段5：提升 + 放置 -->
      <CartesianMove target="{suction_plan.post_grasp}" velocity_scale="0.3" />
      <MoveToPose target="{dump_pose}" velocity_scale="0.6" />
      <ActivateSuction enable="false" />
    </Sequence>
  </RetryUntilSuccessful>
</BehaviorTree>
```

#### 振动模式 BT

```xml
<!-- pick_vibration.xml -->
<BehaviorTree ID="PickVibration">
  <RetryUntilSuccessful num_attempts="{max_retries}">
    <Sequence>
      <!-- 阶段1：全局粗定位 -->
      <GlobalDetect cluster_pose="{cluster_pose}" />
      <MoveToPose target="{cluster_pose}" offset_z="0.35" velocity_scale="0.6" />

      <!-- 阶段2：精细簇分析（质心 + 枝条方向） -->
      <RetryUntilSuccessful num_attempts="3">
        <ClusterAnalyze cluster_info="{cluster_info}" />
      </RetryUntilSuccessful>

      <!-- 阶段3：规划漏斗对准 + 接近 -->
      <PlanVibration cluster_info="{cluster_info}" plan="{vib_plan}" />
      <MoveToPose target="{vib_plan.pre_approach}" velocity_scale="0.5" />

      <!-- 阶段4：缓慢推进，振动棒接触枝条（笛卡尔）-->
      <CartesianMove target="{vib_plan.vibrate_pose}" velocity_scale="0.15" />

      <!-- 阶段5：振动采摘 -->
      <ActivateVibration enable="true" />
      <Wait msec="{vib_plan.vibration_duration_ms}" />
      <ActivateVibration enable="false" />

      <!-- 阶段6：退出 + 倾倒 -->
      <CartesianMove target="{vib_plan.pre_approach}" velocity_scale="0.3" />
      <DumpBerries dump_pose="{dump_pose}" />
    </Sequence>
  </RetryUntilSuccessful>
</BehaviorTree>
```

---

### 5.7 `picking_bringup` — 顶层启动

```
picking_bringup/
└── launch/
    ├── real_suction.launch.py        # 真机 + 吸盘模式
    ├── real_vibration.launch.py      # 真机 + 振动模式
    ├── sim_suction.launch.py         # Gazebo + 吸盘模式
    ├── sim_vibration.launch.py       # Gazebo + 振动模式
    └── debug/
        ├── perception_debug.launch.py
        ├── suction_grasp_debug.launch.py
        └── vibration_grasp_debug.launch.py
```

---

## 6. 关键技术决策

### 6.1 手眼标定

**固定单目（Eye-to-hand）：**
- 工具：easy_handeye2 + ChArUco 棋盘格
- 结果：`calibration/fixed_camera_to_base.yaml`，作为静态 TF 发布

**手腕双目（Eye-in-hand）：**
- 结果固化进 URDF（`wrist_camera_to_ee` transform），随机械臂 TF 树自动传播
- 两种末端执行器换装后相机与机械臂的相对关系不变，标定结果复用

### 6.2 振动模式的枝条方向估计精度要求

枝条方向误差对振动效果的影响分析：

| 枝条估计误差 | 结果 |
|------------|------|
| ±15° 以内 | 振动棒有效接触枝条，采摘正常 |
| 15°~30° | 可能接触到果实而非枝条，效率下降 |
| >30° | 振动棒可能碰撞果实或滑脱，需重试 |

因此 PCA 枝条估计误差在 ±15° 以内即满足要求，双目点云精度完全够用。

### 6.3 仿真阶段感知替代策略

仿真阶段用 fake 节点替代真实感知，加速开发迭代：

```python
# fake_perception_node.py
# 从 Gazebo /gazebo/model_states 读取真实位姿
# 加入高斯噪声（吸盘模式 σ=3mm，振动模式 σ=15mm）
# 发布到真实感知节点的相同 topic
# 通过 launch 参数 use_fake_perception:=true 切换
```
### 6.4 FoundationPose 完整调用方式

#### 两阶段说明

FoundationPose 分为**离线预处理**和**在线推理**两个阶段。没有直接把参考图片喂给网络的接口——"model-free" 的含义是用 BundleSDF 从视频自动建 mesh，之后走完全相同的 model-based 推理路径。

```
离线（一次性）               在线（每次采摘触发）
─────────────                ────────────────────────────
蓝莓参考视频                  RGB + Depth（手腕双目）
   ↓ BundleSDF NeRF 训练       ↓ HSV 颜色分割
blueberry.obj   ──────────→  object mask
                              ↓
                     FoundationPose.register()    ← 首帧/新目标
                        或 .track_one()           ← 连续跟踪
                              ↓
                     4×4 pose（camera frame）
                              ↓ TF 变换
                     PoseStamped（base_link frame）
```

#### 离线步骤：用 BundleSDF 生成蓝莓 mesh

**采集策略：转台拍摄（离线，仅需做一次）**

将一颗蓝莓固定在小转台上，手腕相机（Orbbec Dabai）固定不动，转台每次旋转约 10°，共采集约 36 帧。  
这样可以精确计算 `cam_in_ob`（相机相对目标的位姿），避免依赖 BundleSDF 自身的 SLAM 估计。

```
采集示意（俯视）：

  固定相机
     ↓
  [Orbbec Dabai]
       ↑ 约 20~30cm
  ┌──────────┐
  │  蓝莓    │ ← 转台，每步 10°
  └──────────┘
```

**Step 1：启动 Orbbec Dabai 驱动**

```bash
# 启动 OrbbecSDK_ROS2 驱动（手腕相机，depth_align:=true 使深度与 RGB 对齐）
ros2 launch orbbec_camera dabai.launch.py \
  depth_align:=true \
  color_width:=640 color_height:=480 color_fps:=15 \
  depth_width:=640 depth_height:=480 depth_fps:=15

# 验证 topic 存在
ros2 topic list | grep camera_wrist
# 期望看到:
#   /camera_wrist/color/image_raw
#   /camera_wrist/depth/image_raw
#   /camera_wrist/color/camera_info
```

**Step 2：运行采集脚本**

```python
#!/usr/bin/env python3
# scripts/capture_bundlesdf_data.py
# 使用 ROS 2 topic 从 Orbbec Dabai 采集 BundleSDF 所需数据
# 用法：
#   ros2 run picking_perception capture_bundlesdf_data \
#       --output ref_data/blueberry --num_frames 36

import argparse, os, sys, math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class BundleSdfCapture(Node):
    """
    每次按回车采集一帧（用于转台场景）。
    同时自动生成 K.txt 和基于转台角度的 cam_in_ob/*.txt。

    目录输出结构（BundleSDF 要求）：
      <output>/ob_0000001/
        rgb/         %06d.png   uint8  RGB
        depth/       %06d.png   uint16 mm
        mask/        %06d.png   uint8  0/255
        cam_in_ob/   %06d.txt   4×4 float（相机→目标坐标系）
        K.txt                   3×3 相机内参
    """

    def __init__(self, output_dir: str, num_frames: int):
        super().__init__('bundlesdf_capture')
        self.bridge = CvBridge()
        self.output_dir = os.path.join(output_dir, 'ob_0000001')
        self.num_frames = num_frames
        self.frame_idx  = 0

        os.makedirs(f'{self.output_dir}/rgb',       exist_ok=True)
        os.makedirs(f'{self.output_dir}/depth',     exist_ok=True)
        os.makedirs(f'{self.output_dir}/mask',      exist_ok=True)
        os.makedirs(f'{self.output_dir}/cam_in_ob', exist_ok=True)

        self.latest_rgb   = None
        self.latest_depth = None
        self.K            = None

        self.create_subscription(Image, '/camera_wrist/color/image_raw',
                                 self._rgb_cb,   1)
        self.create_subscription(Image, '/camera_wrist/depth/image_raw',
                                 self._depth_cb, 1)
        self.create_subscription(CameraInfo, '/camera_wrist/color/camera_info',
                                 self._info_cb,  1)

    def _rgb_cb(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

    def _depth_cb(self, msg):
        # Orbbec depth: uint16, 单位 mm；保留原始 uint16 供 BundleSDF 使用
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            k_path = os.path.join(self.output_dir, 'K.txt')
            np.savetxt(k_path, self.K, fmt='%.6f')
            self.get_logger().info(f'K 已保存: {k_path}')

    def _build_cam_in_ob(self, frame_idx: int) -> np.ndarray:
        """
        转台拍摄：目标每帧旋转 step_deg，相机固定。
        cam_in_ob 表示"相机坐标系在目标坐标系中的位姿"。
        假设相机在目标正前方 d=0.25m 处，转台沿 Y 轴旋转。
        """
        step_deg = 360.0 / self.num_frames
        angle_rad = math.radians(frame_idx * step_deg)
        d = 0.25  # 相机到目标中心距离（m），根据实际测量调整

        # 目标绕 Y 轴转 θ 等价于相机在水平面绕目标转 -θ
        c, s = math.cos(-angle_rad), math.sin(-angle_rad)
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]], dtype=np.float64)
        # 相机在目标坐标系中的位置（d 沿 -Z 方向，即面对目标）
        t = np.array([0.0, 0.0, d], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        return T

    def _segment_mask(self, rgb: np.ndarray) -> np.ndarray:
        """HSV 分割蓝莓，返回 uint8 mask（0/255）"""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lo  = np.array([110, 50,  30])
        hi  = np.array([165, 255, 200])
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask   # 0 或 255

    def capture_one_frame(self):
        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warn('等待相机数据...')
            return False
        if self.K is None:
            self.get_logger().warn('等待 CameraInfo...')
            return False

        idx = self.frame_idx
        stem = f'{idx:06d}'

        # 保存 RGB
        rgb_bgr = cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{self.output_dir}/rgb/{stem}.png', rgb_bgr)

        # 保存 depth（uint16, mm）
        cv2.imwrite(f'{self.output_dir}/depth/{stem}.png', self.latest_depth)

        # 保存 mask
        mask = self._segment_mask(self.latest_rgb)
        cv2.imwrite(f'{self.output_dir}/mask/{stem}.png', mask)

        # 保存 cam_in_ob
        T = self._build_cam_in_ob(idx)
        np.savetxt(f'{self.output_dir}/cam_in_ob/{stem}.txt', T, fmt='%.8f')

        self.frame_idx += 1
        self.get_logger().info(f'帧 {idx+1}/{self.num_frames} 已保存')
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',     default='ref_data/blueberry')
    parser.add_argument('--num_frames', type=int, default=36)
    args = parser.parse_args()

    rclpy.init()
    node = BundleSdfCapture(args.output, args.num_frames)

    print(f'\n[采集说明] 共需 {args.num_frames} 帧，每帧转台旋转约 {360/args.num_frames:.1f}°')
    print('按 Enter 采集当前帧，采集完成后自动退出\n')

    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    while node.frame_idx < args.num_frames:
        input(f'  → 按 Enter 采集第 {node.frame_idx+1}/{args.num_frames} 帧 '
              f'（当前转台角度约 {node.frame_idx * 360/args.num_frames:.0f}°）...')
        node.capture_one_frame()

    print(f'\n采集完成，数据保存至: {args.output}/ob_0000001/')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```
**Step 3：修改 `run_nerf.py` 支持 custom 数据集**

`BundleSDF/bundlesdf/run_nerf.py` 原始只支持 `ycbv` 和 `linemod`，需在末尾添加 `run_custom()` 入口：

```python
# 追加到 /home/ziwei/piper_x_dev/FoundationPose/bundlesdf/run_nerf.py 末尾

def run_custom(ref_view_dir: str, ob_id: int = 1):
    """custom 对象入口，适配转台采集数据"""
    import yaml
    with open(f'{code_dir}/config_ycbv.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    # 蓝莓较小（直径约 15mm），降低分辨率阈值
    cfg['mesh_resolution'] = 0.001   # 1mm
    cfg['dbscan_eps']      = 0.005   # 5mm 聚类半径

    base_dir = f'{ref_view_dir}/ob_{ob_id:07d}'
    mesh = run_one_ob(base_dir=base_dir, cfg=cfg)

    os.makedirs(f'{base_dir}/model', exist_ok=True)
    out_file = f'{base_dir}/model/model.obj'
    mesh.export(out_file)
    print(f'[run_custom] mesh 已保存: {out_file}')
    return out_file


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_view_dir', required=True)
    parser.add_argument('--dataset',      default='ycbv',
                        choices=['ycbv', 'linemod', 'custom'])
    parser.add_argument('--ob_id',        type=int, default=1)
    args = parser.parse_args()

    if args.dataset == 'custom':
        run_custom(args.ref_view_dir, args.ob_id)
    elif args.dataset == 'ycbv':
        run_ycbv(args.ref_view_dir)
    elif args.dataset == 'linemod':
        run_linemod(args.ref_view_dir)
```

**Step 4：运行 BundleSDF 训练并部署 mesh**

```bash
# BundleSDF NeRF 训练（约 10~30 分钟，需 GPU，显存 ≥ 8GB）
cd /home/ziwei/piper_x_dev/FoundationPose
python bundlesdf/run_nerf.py \
  --ref_view_dir ref_data/blueberry \
  --dataset custom \
  --ob_id 1

# 验证 mesh 是否合理（顶点数应在 500~5000 之间，无穿模）
python -c "
import trimesh
m = trimesh.load('ref_data/blueberry/ob_0000001/model/model.obj')
print(f'顶点数: {len(m.vertices)}, 面数: {len(m.faces)}')
print(f'包围盒: {m.bounds}')   # 应约为 15mm 球形
m.show()                       # 3D 预览（可选）
"

# 部署到感知包
cp ref_data/blueberry/ob_0000001/model/model.obj \
   ~/picking_ws/src/picking_perception/meshes/blueberry.obj
```

**常见问题：**

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| mask 全黑 | 蓝莓颜色不在 HSV 范围内 | 用 `cv2.imshow` 调整 lo/hi 范围 |
| mesh 缺面/有孔洞 | 某些角度遮挡严重 | 增加帧数（48 或 60 帧），多角度倾斜拍摄 |
| NeRF 不收敛 | `cam_in_ob` 误差大 | 用实际尺子测量 d 值，或在转台上贴 ChArUco 标定 |
| depth 全 0 | 结构光距离太近（< 15cm）或太远（> 1m） | 调整相机到目标距离至 20~30cm |

#### 在线推理：`FoundationPoseWrapper`

源码路径：`/home/ziwei/piper_x_dev/FoundationPose/estimater.py`

核心 API：
- `FoundationPose.register(K, rgb, depth, ob_mask, iteration=5)` → 首帧估计，需要 mask，慢（~1s）
- `FoundationPose.track_one(rgb, depth, K, iteration=2)` → 连续跟踪，无需 mask，快（~100ms）

```python
# foundation_pose_wrapper.py
import sys
sys.path.insert(0, '/home/ziwei/piper_x_dev/FoundationPose')

import numpy as np
import trimesh
import nvdiffrast.torch as dr
from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor

class FoundationPoseWrapper:
    """
    封装 FoundationPose 推理逻辑，供 FineDetectorNode 调用。

    使用模式：
      detect_all()  → 每次服务请求时调用，返回当前帧所有蓝莓位姿
                      内部对每个连通分量单独调用 register()

    注意：register() 每次都重新估计，适合"靠近后检测一次"的场景。
    如果需要连续跟踪单颗蓝莓（如吸盘模式调整接近过程），
    可切换为 track_one() 模式（需保存上一帧 est 实例）。
    """

    # 权重路径（相对 FoundationPose 源码根目录）
    SCORE_CKPT  = 'weights/2024-01-11-20-02-45'
    REFINE_CKPT = 'weights/2023-10-28-18-33-37'

    def __init__(self, mesh_path: str, est_refine_iter=5, track_refine_iter=2,
                 score_threshold=0.3):
        self.est_refine_iter   = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.score_threshold   = score_threshold

        # 加载 mesh
        self.mesh = trimesh.load(mesh_path)

        # 加载预训练模型（只加载一次，常驻 GPU）
        self.scorer  = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx   = dr.RasterizeCudaContext()

    def _make_estimator(self) -> FoundationPose:
        """每颗蓝莓独立创建一个 estimator 实例"""
        return FoundationPose(
            model_pts     = np.array(self.mesh.vertices),
            model_normals = np.array(self.mesh.vertex_normals),
            mesh          = self.mesh,
            scorer        = self.scorer,
            refiner       = self.refiner,
            glctx         = self.glctx,
            debug         = 0,
        )

    def detect_all(self,
                   rgb:   np.ndarray,   # (H,W,3) uint8
                   depth: np.ndarray,   # (H,W)   float32, 单位 m
                   K:     np.ndarray,   # (3,3)   相机内参
                   mask:  np.ndarray,   # (H,W)   uint8 蓝莓掩码（HSV分割结果）
                   ) -> list[tuple[np.ndarray, float]]:
        """
        返回: [(pose_4x4, score), ...]
          pose_4x4: (4,4) float64, object-in-camera，OpenCV 坐标系
          score:    float, 0~1 置信度

        实现方式：对掩码的每个连通分量分别调用 register()
        （一帧内检测多颗蓝莓时，每颗独立估计）
        """
        import cv2

        # 找连通分量：每个分量对应一颗可能的蓝莓
        num_labels, labels = cv2.connectedComponents(mask)
        results = []

        for label_id in range(1, num_labels):   # 0 是背景
            component_mask = (labels == label_id).astype(np.uint8)

            # 过滤太小的区域（噪点）
            if component_mask.sum() < 200:
                continue

            est = self._make_estimator()
            try:
                pose = est.register(
                    K         = K,
                    rgb       = rgb,
                    depth     = depth,
                    ob_mask   = component_mask,
                    iteration = self.est_refine_iter,
                )
                # est.pose_last 同时保存了 score，从 scorer 中取最终分数
                score = float(est.pose_last_score) \
                        if hasattr(est, 'pose_last_score') else 0.5

                if score >= self.score_threshold:
                    results.append((pose, score))
            except Exception as e:
                pass   # 单颗估计失败不影响其他颗

        return results   # [(4x4 numpy, float), ...]
```

#### 坐标系说明

FoundationPose 输出的 pose 是 **OpenCV 相机坐标系**下的 object-in-camera 变换：

```
p_camera = pose @ p_object_homogeneous

OpenCV 相机坐标系：
  X → 右
  Y ↓ 下
  Z → 前（光轴方向）
```

转换到 ROS `base_link` 的完整链路：

```python
# TF 链: camera_wrist_optical_frame → camera_wrist_link
#         → ee_link → link6 → ... → base_link
#（通过 URDF 和手眼标定自动构建，无需手动计算）

T_base_cam = transform_to_matrix(
    tf_buffer.lookup_transform('base_link',
                               'camera_wrist_optical_frame',
                               rclpy.time.Time()))

pose_in_base = T_base_cam @ pose_in_camera   # (4,4) @ (4,4)
```

#### `register()` vs `track_one()` 选择策略

| 场景 | 推荐 API | 原因 |
|------|---------|------|
| 机械臂停止，触发一次性检测 | `register()` | 精度最高，不依赖上一帧 |
| 机械臂缓慢接近中，连续更新位姿 | `track_one()` | 速度快（~100ms），适合实时更新 |
| 检测到新目标或切换蓝莓 | `register()` | 必须重新初始化，track 无效 |
| 仅振动模式（只需簇位置） | `register()` 一次 | 只调用一次，精度够用 |

**吸盘模式推荐用法：**

```python
# 1. 机械臂到预观测点后静止，调用 register() 获得精确位姿
poses = fp_wrapper.detect_all(rgb, depth, K, mask)  # → register()

# 2. 机械臂缓慢执行笛卡尔路径接近，可选用 track_one() 实时更新
#    （需保存 est 实例，每帧调用 est.track_one(rgb, depth, K, iteration=2)）
```

### 6.5 末端换装工作流

换装时需要的系统配置切换：

```bash
# 换装后重新启动，指定末端类型
ros2 launch picking_bringup real_vibration.launch.py
# launch 内部自动切换：
# - URDF (piper_with_vibration.urdf.xacro)
# - SRDF (piper_vibration.srdf)
# - MoveIt EEF 定义
# - 感知节点（cluster_analyzer 替代 fine_pose_estimator）
# - BT 树（pick_vibration.xml）
```

---
## 7. 开发里程碑

### Phase 0：基础环境（1周）
- [ ] 建立 workspace，创建所有包骨架
- [ ] 定义全部 `picking_msgs` 接口（包含双模式消息）
- [ ] 完成两种末端 URDF（吸盘 + 振动漏斗，含双相机占位）

### Phase 1：吸盘模式仿真（2周）
- [ ] Gazebo 场景：桌面蓝莓模型
- [ ] `fake_perception_node`（高精度噪声，模拟吸盘模式）
- [ ] 验证 MoveIt 规划 + 笛卡尔路径执行
- [ ] 调通吸盘模式完整 BT 流程

### Phase 2：振动模式仿真（2周）
- [ ] Gazebo 中建立蓝莓簇 + 枝条模型
- [ ] `fake_perception_node`（低精度噪声，模拟振动模式）
- [ ] 验证枝条卡槽对准运动 + 振动执行流程
- [ ] 调通振动模式完整 BT 流程

### Phase 3：真实感知接入（3周）
- [ ] 完成手眼标定（双相机）
- [ ] **BundleSDF 离线建 mesh**：采集蓝莓参考视频 → 运行 `bundlesdf/run_nerf.py` → 生成 `blueberry.obj`
- [ ] 验证 mesh 质量（在 MeshLab/trimesh 中检查，顶点数合理，无穿模）
- [ ] 跑通 `FoundationPoseWrapper.detect_all()`，验证单帧输出 pose 在 RViz 中对齐
- [ ] 验证 TF 链：`camera_wrist_optical_frame` → `base_link` 坐标转换误差 ≤5mm
- [ ] 振动模式：对检测到的多颗蓝莓位置做 PCA，在 RViz 中可视化枝条方向箭头

### Phase 4：真机联调（2周）
- [ ] 吸盘模式：低速全流程，调整吸附阈值
- [ ] 振动模式：验证枝条入槽精度，调整振动时长和接触位置
- [ ] 两种模式换装切换测试
- [ ] 连续批量采摘测试

---

## 8. 依赖清单

| 依赖包 | 用途 | 安装方式 |
|--------|------|----------|
| `moveit2` | 运动规划 | `apt` |
| `ros2_control` | 轨迹 + GPIO 执行 | `apt` |
| `gpio_controllers` | 气泵/电机继电器控制 | `apt` |
| `behaviortree_cpp_v4` | 任务协调 | `apt` / 源码 |
| `OrbbecSDK_ROS2` | Orbbec Dabai 手腕相机驱动（RGB + 结构光深度） | 源码：github.com/orbbec/OrbbecSDK_ROS2 |
| `v4l2_camera` | 固定 USB 单目相机驱动 | `apt` |
| `easy_handeye2` | 手眼标定 | 源码 |
| `FoundationPose` | 6D 位姿估计（两种模式共用） | 源码：`/home/ziwei/piper_x_dev/FoundationPose` |
| `BundleSDF` | 离线生成蓝莓 mesh（含于 FoundationPose 源码的 `bundlesdf/` 子目录） | 同上 |
| `PyTorch` + `CUDA` | FoundationPose / BundleSDF 推理后端 | pip + conda |
| `nvdiffrast` | GPU 光栅化渲染（FoundationPose 必须） | 源码编译 |
| `pytorch3d` | 3D 几何操作（FoundationPose 必须） | 源码编译 |
| `trimesh` | mesh 加载与操作 | pip |
| `groot2` | BT 可视化调试 | 二进制 |
| `gazebo_ros` | 仿真 | `apt` |