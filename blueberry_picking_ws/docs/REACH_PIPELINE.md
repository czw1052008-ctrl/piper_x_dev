# 蓝莓触达管线（Reach Pipeline）

真机主路径：**键盘遥操** + **固定单目锁目标** + **手腕 RGB-D 精细触达** + **人工确认后 reset**。  
不做气泵/GPIO 采摘。任务与感知走 **topic**，不新增业务 service。

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
| `/perception/global/berries` | `picking_msgs/DetectedBerryArray` | 全局候选 ~2–5 Hz |
| `/perception/target_lock` | `picking_msgs/DetectedBerry` | FSM 锁定的采集对象 |
| `/perception/fine/berries` | `picking_msgs/DetectedBerryArray` | 手腕 YOLO+depth ~2–5 Hz |
| `/reach/plan` | `picking_msgs/SuctionGraspPlan` | pre / contact / retreat |
| `/reach/status` | `std_msgs/String` | 状态名 |
| `/reach/cmd` | `std_msgs/String` | `start` / `confirm_reset` / `abort` / `clear_lock` |
| `/reach/reached` | `std_msgs/Bool` | 触达完成边沿 |

## 状态机

```
IDLE → LOCKING → ALIGNING → REFINING → PLANNING → APPROACHING → REACHED → WAIT_CONFIRM → RESETTING → IDLE
```

- `start`：从 IDLE 进入 LOCKING  
- `ALIGNING`：**agent/启发式估关节目标 → 位置控制 → hold → 粗方位判定**（`set_joints` / `coarse_ok`）；详见 [ALIGN_HANDOFF_2026-08-04.md](ALIGN_HANDOFF_2026-08-04.md)。`run_align_qa.sh` 默认 file+agent。旧「写死整臂步长」已废弃。  
- `REFINING`：手腕 fine 3D；超时则回退用全局锁定位姿规划（粗）  
- `confirm_reset`：WAIT_CONFIRM → 关节回 `[0,0,0,0,0,0]`  
- `abort` / `clear_lock`：中止并清锁  

运动期间请停 teleop，避免与 MoveIt 抢控制。FSM 用 **异步 MoveIt**（`MultiThreadedExecutor` + 非阻塞 poll），避免 timer 内 `spin_until_future_complete` 死锁。

### 阶段冒烟

```bash
# 0 硬件 / 1 global / 2 fine / 3 规划单测 / 4 FSM dry-run；截图在 log/real_robot/stage_N/
bash scripts/reach_stage_smoke.sh 0
bash scripts/reach_stage_smoke.sh all
```

## 相机分工

1. **固定单目**：全局检测，选定并钉住采集对象（`/perception/target_lock`）。  
2. **手腕 RGB-D**：深度中值 → base 系 3D；失败时 mono 尺寸先验。主导触达坐标。  
3. 固定相机 `fixed_camera_to_base.yaml` / `FIXED_CAM_*` 若为占位，全局位姿可能漂；触达仍以手腕精定位为准。

### 固定相机外参（卷尺粗标，2026-08-04）

`FIXED_CAM_TX/TY/TZ=(0.60,0.33,0.48)`；朝向按「斜下看工作区/植株」look-at 生成四元数（见 `fixed_camera_to_base.yaml`）。  
重启固定相机 launch 后生效：`tf2_echo base_link camera_fixed_optical_frame`。若粗对准仍偏，微调平移或 look-at 目标点。

## 启动

```bash
# 配置
cp config/real_robot.env.example config/real_robot.env
# 编辑 PIPER_X_DEV、CAN、FIXED_CAMERA_DEVICE、ORBBEC_USB_PORT

source /opt/ros/humble/setup.bash
source ../agx_arm_ros/install/setup.bash
source ../OrbbecSDK_ROS2_main/install/setup.bash   # Dabai DC1 = OpenNI / SDK v1
export LD_LIBRARY_PATH="../OrbbecSDK_ROS2_main/install/orbbec_camera/lib:${LD_LIBRARY_PATH}"
source install/setup.bash
# Keep SciPy/apt compatible (avoid user-site NumPy 2.x breaking agx_arm_ctrl)
export PYTHONNOUSERSITE=1

# 键盘遥操（可叠加相机/YOLO 可视化）
bash scripts/real_robot_teleop.sh --viz
# 仅开窗口不遥操：
bash scripts/run_teleop_viz.sh
# 窗口：固定 HSV、手腕 YOLO、手腕深度伪彩
# topic：/perception/global/detection_viz
#         /perception/fine/detection_viz
#         /perception/wrist/depth_viz

# 触达管线（臂 + 双相机 + 感知 + FSM）
bash scripts/run_real_reach.sh
# 另开终端：
ros2 topic pub --once /reach/cmd std_msgs/String "{data: start}"
# 触达完成后：
ros2 topic pub --once /reach/cmd std_msgs/String "{data: confirm_reset}"
```

## 验收清单

- [x] `can0` UP  
- [x] `/camera_wrist/color|depth` 有数据（SDK **v1** / `OrbbecSDK_ROS2_main`，约 18–27 Hz）  
- [x] `/camera_fixed/image_raw` 有数据  
- [x] `link6 → camera_wrist_color_optical_frame` 静态 TF（bringup `camera_tf`；`CAMERA_MOUNT_TY=-0.08`）  
- [x] 工作空间可编译：`picking_msgs/perception/bringup/grasp`  
- [x] dry-run：`/reach/cmd start` → `LOCKING→ALIGNING→REFINING→PLANNING→APPROACHING→REACHED→WAIT_CONFIRM` → `confirm_reset` → `IDLE`  
- [x] 真机运动（2026-08-04）：异步 MoveIt；`align/pre/contact/home` 均 OK → `WAIT_CONFIRM` → `IDLE`  
- [ ] 键盘 teleop（`bash scripts/real_robot_teleop.sh --viz`）— 可选验收  
- [ ] 手腕 FOV 对准植株后精测距（当前 ALIGNING 仅位置约束，腕姿可能未朝向果簇；YOLO 因 numpy/matplotlib 冲突回退 HSV）  

### Orbbec 已知问题（2026-08）

DaBai DC1 是 **双 USB 口**：彩色 `2bc5:0557`（UVC）+ 深度 `2bc5:0657`（Vendor/OpenNI）。两路都会出现在 `lsusb` / SDK 端口列表里。

| SDK | 枚举两路 | 匹配成设备 |
|-----|----------|------------|
| **v2.8.x**（当前 `OrbbecSDK_ROS2`） | 是 | **否** → `No matched usb device found!` |
| **v1.10.x**（`OrbbecSDK_ROS2_main`） | 是 | **是** → `New openni device matched` / `DaBai DC1 PID 0x0657` |

结论：DC1 走 **OpenNI 协议**，应用 **SDK v1 + `OrbbecSDK_ROS2_main`**，不是 v2 UVC 路径。  
`libob_usb.so` / `liblive555.so` 已从 [OrbbecSDK v1.10.27](https://github.com/orbbec/OrbbecSDK/releases) 补进 `OrbbecSDK_ROS2_main/install/orbbec_camera/lib/`；bringup 会把该路径加入 `LD_LIBRARY_PATH`。`OrbbecSDK_ROS2`（v2）仅作归档对照，**不要**再当真机主路径。

辅助现象：深度口常无序列号 → `Failed to query USB device serial number`（描述符 index 0），不阻止 v1 匹配。  
当前若仍挂在 USB2 hub（`lsusb -t` 显示 480M），优先改插 **Bus 02 / 5000M+** 根口。

### 机械臂 / NumPy /「必须断电」根因

`agx_arm_ctrl` 依赖 apt SciPy；若加载到 **NumPy 2.x**（常见于 `~/.local`）会直接崩。Bringup 已设 `PYTHONNOUSERSITE=1`；本机请保持系统侧 `numpy==1.24.x`。

**为何有时只能给臂断电再上电才恢复（2026-08 复现结论）：**

1. **触发**：整卡 USB host（`xhci_hcd` unbind/bind）或 GS-USB 被拽掉时，臂往往已处于 **joint enable**。主站突然消失 ≠ 干净 `disable()`。
2. **主机侧**：`gs_usb` 出现 `failed to xmit URB … -ENOENT`，随后 `Error -71` / `failed to set bittiming: -EPROTO` → `ip link set can0 up` 报 Protocol error，软复位不够，需拔插 candleLight 或只复位该 USB 口。
3. **臂侧**：使能中的关节驱动在 CAN 中断后进入故障锁存，不再回状态帧；`enable()` / `get_firmware()` 一直超时。软件重使能不能清锁存，**只有臂电源复位**才能清。
4. **避免**：动 USB/Orbbec 前先 `real_robot_shutdown`（并确认已 disable）；**禁止**在臂使能时整卡 xHCI reset；Orbbec 只复位其端口。

诊断口令：`ip -s link show can0`、`sudo dmesg | grep gs_usb`、`timeout 3 candump can0`。

## 明确不做

GPIO 吸盘、振动真机、BT 真机、FoundationPose 主路径、重写 `agx_arm_ros`。
