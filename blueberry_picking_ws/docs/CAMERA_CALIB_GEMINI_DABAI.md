# Gemini 305 腕 + DaBai 全局 — 外参重标定

硬件（本机已写入 `config/real_robot.env`）：

| 角色 | 设备 | 序列号 | USB | SDK |
|------|------|--------|-----|-----|
| 腕 | Gemini 305 | `CV2L761000W1` | `2-6`（USB3 @ 5000M） | `OrbbecSDK_ROS2` |
| 全局 | DaBai DC1 | `CC1N162021Z` | `1-8.4`（USB2） | `OrbbecSDK_ROS2_main` |

冒烟：`bash scripts/smoke_dual_orbbec.sh`

## 已写入

| 项 | 值 |
|----|-----|
| 全局 DaBai → `base_link` | **2026-09-01 FK 绿骨架贴合**：`tx,ty,tz = -0.72, 0.28, 0.20`；quat `0.5,-0.5,0.5,-0.5`（光轴大致朝 +X）。验收：`/planning/viz/fixed_fk_skeleton` |
| 腕 Gemini → `link6` 平移 | `0, -0.07, +0.04` |
| 腕光学俯仰 Rx | **-32.95°**（2026-08-11：`tip_z=0.148`, `d=0.19`） |
| 吸盘中心 → `link6` | `0, 0, 0.07`（`ARM_TCP_OFFSET` / `PICK_TCP_MOUNT_Z` / `suction_eef`） |

改 TCP / `CAMERA_MOUNT_*` 后需 **重启臂栈或至少重发静态 TF**（`tcp_offset` 只在 launch 时生效；腕 TF 已在标定后重发）。

## 腕光轴 Rx（已完成 2026-08-11）

现场：果 152033 对准主点；`tip_z=0.148`，`d=0.19` → **Rx = -32.95°**。

复现：

```bash
python3 scripts/fit_wrist_optical_pitch.py \
  --tip-to-berry-z 0.148 --cam-center-to-berry-z 0.19 \
  --tx 0 --ty -0.07 --tz 0.04 --apply
```

验证：`ros2 run tf2_ros tf2_echo link6 camera_wrist_color_optical_frame`
