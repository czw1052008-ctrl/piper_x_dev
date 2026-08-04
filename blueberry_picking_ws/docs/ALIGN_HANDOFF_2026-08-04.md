# ALIGN 探索交接（2026-08-04）

> 给**下一次对话**的无缝入口。先读本文件，再按「下一步」继续；勿从旧的「写死整臂步长」方案接着改。

## 一句话结论

**ALIGNING 已改为：agent 看固定单目估关节目标 → 位置控制到位后 hold → agent 判粗方位 `coarse_ok` → 进腕部精细。**  
真机 session `20260804_202637` 已跑通一轮：`set_joints`（j1≈29°/j5≈-43°）→ hold（yaw_err≈0.4°）→ `coarse_ok` → `WAIT_CONFIRM`（align-only）。腕部画面已出现浆果/叶片。

## 当前协议（权威）

### 决策动作

| action | 含义 |
|--------|------|
| `set_joints` | 位置控制。必带 `joints_deg`（绝对角，优先）或 `delta_deg` |
| `coarse_ok` / `done` | 粗对准 OK → 非 align-only 进 `REFINING`；align-only 进 `WAIT_CONFIRM` |

**禁止**再靠写死 `whole_arm_left/right ±35°` 来回摆；命名动作仅作遗留兼容，启发式也会改成「瞄 `plant_yaw`」。

### 两阶段

1. **phase=command**：看 `align_XX_request_*.png`（fixed 主、wrist 辅）→ 估量 → 写决策  
2. 到位 **hold**（不自动 rollback）→ 出 `align_XX_judge_*.png`  
3. **phase=judge**：粗方位对 → `coarse_ok`；不对 → 再写一版 `set_joints`

### 写决策示例

```bash
cd blueberry_picking_ws

# command：绝对角
python3 scripts/write_align_decision.py \
  --session-dir log/real_robot/qa/<SESSION> \
  --step-idx 0 --phase command --action set_joints \
  --joints-deg 'joint1=29,joint2=-12,joint3=15,joint5=-43' \
  --reason 'fixed: plant right of tip; aim plant_yaw + pitch down' \
  --provider agent

# judge：粗对准通过
python3 scripts/write_align_decision.py \
  --session-dir log/real_robot/qa/<SESSION> \
  --step-idx 0 --phase judge --action coarse_ok \
  --reason 'held: wrist faces plant; wrist RGB has foliage' \
  --provider agent
```

决策文件：`qa/<session>/align_decision.json`（FSM 消费后改名为 `.used_<step>_<phase>`）。

## 关键文件

| 路径 | 角色 |
|------|------|
| `scripts/align_judge.py` | `estimate_joint_targets_deg` / `decide_action(phase=…)` / `apply_joint_command` |
| `scripts/write_align_decision.py` | agent 写决策 CLI |
| `scripts/align_vlm_decider.py` | file 模式旁路：`heuristic` / `agent`（等文件）/ `qwen`（未接） |
| `scripts/reach_fsm_node.py` | ALIGN：command→traj→hold→judge；`--align-judge-mode file\|heuristic` |
| `scripts/run_align_qa.sh` | align-only QA；默认 `ALIGN_JUDGE_MODE=file` `ALIGN_VLM_PROVIDER=agent` |
| `scripts/snap_reach_qa.sh` | 拍 fixed/wrist/viz |
| `test/test_align_judge.py` | 单测（`pytest test/test_align_judge.py`） |
| `.cursor/skills/piper-reach/SKILL.md` | 真机硬规则（Orbbec v1、禁整卡 xHCI reset 等） |

## 怎么跑一轮（真机）

```bash
# 1) 宿主机确认 can0（勿在 Cursor 沙箱里判断 can0）
ip -brief link show can0   # 应为 UP

# 2) bringup
bash scripts/real_robot_bringup.sh --fixed-cam --reach-perception --no-wait

# 3) 使能
source /opt/ros/humble/setup.bash
source ../agx_arm_ros/install/setup.bash && source install/setup.bash
ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}"
ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}"

# 4) align QA（file + agent）
export ALIGN_JUDGE_MODE=file ALIGN_VLM_PROVIDER=agent
bash scripts/run_align_qa.sh

# 5) 看最新 session，按 phase 写决策
ls -1dt log/real_robot/qa/[0-9]*/ | head -1
# … write_align_decision.py …

# 6) 回 home
ros2 topic pub --once /reach/cmd std_msgs/msg/String "{data: confirm_reset}"
```

观测字段（request/judge JSON）：`joint*_deg`、`plant_yaw_deg`、`yaw_error_deg`、`fine_visible`、`fixed_dx_px` 等。**以图像为准**，数值只是辅助。

## 今日踩过的坑（明天别再踩）

### 1. 「can0 not found」经常是假报

- **宿主机**上 `can0` 可能一直 UP；Cursor **沙箱网络命名空间**里看不到 can0，复杂脚本会误报。  
- 用短命令在 allowlist/宿主机查：`ip -brief link show can0`。  
- **FSM `ERROR` ≠ 卸掉 can0**。真坏 can0 常见：GS-USB 掉枚举、BUS-OFF/无 ACK、双 `agx_arm_ctrl` 抢总线、整卡 xHCI reset。  
- 用户侧常「重启解决」——区分：**假报（沙箱）** vs **真 USB/CAN 故障**。

### 2. 写死步长 → 过冲左右摆（已废弃）

- 旧 `whole_arm_right/left ±35°`：图右语义对，但过冲后启发式立刻反向。  
- 已改为估计绝对目标 + hold + agent 再判。

### 3. 左右语义标定（仍有效）

- 本机固定单目：植株在臂尖**图像右侧** ↔ 需要 **增大 joint1**（朝 plant_yaw）。  
- 历史误用：`fixed_dx_px` 极性不可靠；`yaw_left` 命名与口语「大臂右转」易混——现用 `set_joints` + 绝对角。

### 4. Gate A 验收

- **看图**，不要信算法自报。  
- 通过：单目粗朝向植株；腕部最好已见植株/叶片（不必 YOLO `fine_visible`）。  
- 假阳性：腕部 neon/墙/窗被 YOLO 当浆果——`--align-fine-visible-conf` 默认 **0.55**；并有 `--fine-max-age-s` 防陈旧 fine。

### 5. j2/j3 命令了反馈仍为 0

- session `20260804_202637`：`joints_deg` 含 j2=-12 / j3=15，反馈仍 0；j1/j5 正常。  
- **明天应查**：`follow_joint_trajectory` 是否真正下发六轴、控制器是否忽略部分轴、观测是否漏记。粗对准目前可不阻塞。

### 6. 运维注意

- **单实例** `reach_fsm_node`；勿 `pkill -f` 匹配到当前 shell 命令行。  
- bringup/CAN 相关操作尽量在**非沙箱**环境；需要时对 kill/bringup 用完整权限。  
- Orbbec 用 `OrbbecSDK_ROS2_main`（SDK v1）；`LD_LIBRARY_PATH` 含 `install/orbbec_camera/lib`。

## 成功会话摘要

| Session | 结果 |
|---------|------|
| `20260804_193151` 等 | 早期启发式/小步：有运动，腕部未朝植株 |
| `20260804_194711` | file+agent 旧动作：方向讨论，Gate A 未过 |
| `20260804_200431` | `whole_arm_*` 方向对但过冲振荡 |
| **`20260804_202637`** | **新协议成功**：set_joints → hold → coarse_ok → WAIT_CONFIRM |

QA 图：`log/real_robot/qa/20260804_202637/`（request / before / after / judge）。

## 明天优先做什么

1. **查 j2/j3 未动**（轨迹点 / `arm_controller` / feedback）。  
2. **非 align-only**：`coarse_ok` → `REFINING`（腕部 fine）真机冒烟。  
3. 可选：接真实 Qwen2.5-VL 到 `provider=qwen`；或加强 agent 估角（少依赖 `plant_yaw`）。  
4. 可选：把「宿主机 vs 沙箱 can0」写进 bringup 提示或 skill，减少误报重启。

## 单测

```bash
cd blueberry_picking_ws
python3 -m pytest test/test_align_judge.py -q
```

## 对话入口提示（粘贴给新 Agent）

```
继续 Piper X ALIGN：读 blueberry_picking_ws/docs/ALIGN_HANDOFF_2026-08-04.md。
协议已是 set_joints 位置控制 + hold + coarse_ok，不要恢复写死 whole_arm 步长。
下一步优先：查 j2/j3 未动，再跑非 align-only 进 REFINING。
```
