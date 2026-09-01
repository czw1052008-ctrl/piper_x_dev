# 采摘循环管线（Pick Cycle）

PBVS 主线路径上的**多果采摘编排**，替代原 `run_real_reach.sh` 单次触达流程。

## 架构

```
run_pick_system.sh
  ├── real_robot_bringup（臂 + 双 Orbbec + global/fine 检测）
  ├── reach_fsm_node.py（--use-pbvs --pbvs-mode single，单果触达）
  └── pick_cycle_fsm_node.py（多果循环编排）
        └── suction_keyboard_sim.py（独立终端：s/e/y/f）
```

### 状态机

**吸盘（现状）：**
```
IDLE → CLUSTER_PLAN → CLUSTER_ALIGN_MOVE
  → [每颗果] RESTORE_ENTRY → TOUCH(PBVS) → CONFIRM → RETRACT(+n) → SUCTION_OFF → RESTORE_ENTRY
  → MISSION_DONE → HOME
```

**两指夹爪 `agx_gripper_v1`（目标序列，异于吸附）：**
```
… → RESTORE_ENTRY
  → PRE_OPEN（微开 pre_grasp_width）
  → TOUCH / PRESS_IN（tip 沿 approach 越过果心 press_in_m）
  → GRASP_CLOSE（合至 grasp_width，轻夹）
  → CONFIRM → RETRACT（保持夹紧）→ DROP_OPEN → RESTORE_ENTRY
```

| 阶段 | 吸盘 | 夹爪 |
|------|------|------|
| 簇对齐位 | `refine_entry_pose.json` | 同左 |
| 果实队列 | 全局相机近→远 | 同左（需固定相机簇） |
| 接触前 | — | `gripper_cmd.py --open` |
| 触达 | tip 贴果，无 press | tip **压入** `press_in_m` |
| 锁定 | 键盘 `s` 开吸 | `gripper_cmd.py --close` |
| 扯果 | +n 退 `retract_m` | 保持夹紧后退 |
| 落果 | 键盘 `e` 断吸 | OPEN 放果 |

参数见 `config/end_effector_profiles.yaml` → `agx_gripper_v1`。


### Topic

| Topic | 说明 |
|-------|------|
| `/pick/cmd` | `start_mission` / `retry_fruit` / `abort` |
| `/pick/status` | 采摘循环状态 |
| `/pick/suction_state` | `Bool`，键盘模拟吸附 |
| `/pick/touch_confirm` | `ok` / `fail` |
| `/reach/cmd` | 内层 FSM：`start_refine` / `next_fruit` / `confirm_reset` |

---

## 使用方式

### 前置条件

1. 已配置 `config/real_robot.env`（CAN、相机、外参等）
2. 已有作业簇对齐位：`log/real_robot/refine_entry_pose.json`  
   （当前由 agent ALIGN / `refine_entry_pose.py save` 写出；后续接 VLM 自动规划）
3. 硬件就绪：臂使能、双 Orbbec 出图、全局/腕部检测正常

快速检查：

```bash
ros2 topic hz /camera_wrist/color/image_raw
ros2 topic hz /camera_fixed/color/image_raw
ros2 topic hz /perception/global/berries
ros2 topic hz /perception/fine/berries
```

### 一键启动（两个终端）

**终端 1 — 启动整套栈 + 采摘编排：**

```bash
cd blueberry_picking_ws
bash scripts/run_pick_system.sh
```

脚本会自动拉起：臂 bringup、双相机感知、`reach_fsm`（PBVS 模式）、`pick_cycle_fsm`。

**终端 2 — 键盘模拟吸附 / 触达确认：**

```bash
cd blueberry_picking_ws
export PYTHONPATH="$(pwd)/scripts:${PYTHONPATH:-}"
python3 scripts/suction_keyboard_sim.py
```

键盘映射（终端需保持焦点）：

| 键 | 作用 |
|----|------|
| `s` | 吸附开（触达开始时按） |
| `e` | 吸附关（回撤到落果位后按） |
| `y` | 触达成功确认 |
| `f` | 触达失败（abort + 录制数据） |
| `q` | 退出键盘节点 |

### 开始采摘任务

```bash
ros2 topic pub --once /pick/cmd std_msgs/String "{data: start_mission}"
```

监控状态：

```bash
# 外层采摘循环
ros2 topic echo /pick/status

# 内层 PBVS 触达
ros2 topic echo /reach/status
```

### 操作员流程（每颗果）

单果循环：`entry → touch → retract → 落果断吸 → entry（下一颗）`

1. 系统自动 **restore entry**（回到作业簇对齐位）
2. 系统发送 **start_refine**，PBVS 4s oneshot 触达开始  
   → **触达开始同时按 `s`**（开吸，硬件未到位时用键盘模拟）
3. 触达完成（`/reach/status` → `WAIT_CONFIRM`）后：  
   → 判断触达是否成功，按 **`y`**（成功）或 **`f`**（失败）
4. 确认成功后，系统沿法向 **+n 回撤 50mm**（保持吸附）
5. 到达落果安全位后，按 **`e`**（关吸，果实落入下方篓子）
6. 系统自动回 **entry**，开始下一颗果

整簇采完后，系统自动 `confirm_reset` 回机械臂 home。

### 失败处理

触达失败、扯果失败、吸附超时等会进入 `ERROR_ABORT`，失败过程数据写入：

```
log/real_robot/pick_failures/<mission>_fruit<NN>_<timestamp>/
  failure.json          # 失败阶段与原因
  reach_qa/             # 对应 reach_fsm QA session 快照（若有）
```

人工分析清楚后，重试当前果：

```bash
ros2 topic pub --once /pick/cmd std_msgs/String "{data: retry_fruit}"
```

中止整次任务：

```bash
ros2 topic pub --once /pick/cmd std_msgs/String "{data: abort}"
```

### 测试模式：跳过簇对齐，直接从果实任务开始

用于验证 `entry → touch → retract → entry` 闭环，无需全局相机簇规划。

**要求：** 已有 `log/real_robot/refine_entry_pose.json`，机械臂在 entry 附近。

```bash
bash scripts/run_pick_system.sh --test-config config/pick_test_fruit_task.json
```

测试配置示例（`config/pick_test_fruit_task.json`）：

```json
{
  "skip_to": "fruit_task",
  "entry_pose": "log/real_robot/refine_entry_pose.json",
  "fruit_count": 1,
  "start_fruit_index": 0
}
```

启动后同样开键盘终端，发 `start_mission` 或等待 FSM 自动进入果实任务（test 模式会直接进入 `FRUIT_RESTORE_ENTRY`）。

### 其他启动参数

```bash
# dry-run（reach_fsm 不真动臂，调试用）
bash scripts/run_pick_system.sh --dry-run

# 指定 real_robot.env
bash scripts/run_pick_system.sh --config config/real_robot.env
```

`run_real_reach.sh` 已弃用，内部转调 `run_pick_system.sh`。

### 下班安全关机

```bash
cd blueberry_picking_ws
bash scripts/_safe_poweroff.sh
```

等价：abort → confirm_reset → home → 下使能 → 杀栈。**禁止**臂使能时整卡 xHCI reset。

---

## 模块

| 文件 | 作用 |
|------|------|
| `scripts/run_pick_system.sh` | 一键部署入口 |
| `scripts/pick_cycle_fsm_node.py` | 顶层编排 |
| `scripts/pick_motion.py` | restore / reach cmd 封装 |
| `scripts/pick_retract.py` | +n 回撤 50mm |
| `scripts/fruit_queue.py` | 全局果队列排序 |
| `scripts/cluster_align_planner.py` | 簇选择 + JSON 对齐位 |
| `scripts/pick_session_recorder.py` | 失败录制 |
| `scripts/suction_keyboard_sim.py` | 键盘模拟 |
| `config/pick_test_fruit_task.json` | 果实任务测试跳步配置 |

## 与 reach_fsm 的衔接

新增 `/reach/cmd next_fruit`：从 `WAIT_CONFIRM` 回到 `IDLE` **不回 home**，供多果循环使用。整簇结束后 `confirm_reset` 回 home。

相关文档：[REACH_PIPELINE.md](REACH_PIPELINE.md)（PBVS 触达细节、QA 离线复盘）
