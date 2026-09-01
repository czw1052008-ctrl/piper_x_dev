# 学习型轨迹规划器（Pick Trajectory Planner）

统一替换 ALIGN / 簇对齐 / 触达路径上的 **VLM / 启发式 / 单步 BC**，与 [`PICK_CYCLE.md`](PICK_CYCLE.md) 编排层对接。

**设计原则（已定稿）：**

- 模型**唯一输出**：tool 帧 **6D 绝对位姿轨迹**（position + approach_axis），所有 FSM 阶段格式相同。
- **1 s replan**，**4 s horizon**，16 路点（dt=0.25 s），每周期只执行前 5 点（0~1 s）。
- **选果、阶段切换**由规则 + 外层 FSM 完成；`fruit_id`、`fsm_state`、`tool_profile` 为模型**输入**。
- **关节角、样条、限速、跟踪**由传统执行层（IK + 控制）完成，模型不输出关节。

---

## 1. 系统位置

```
感知 (YOLO + 深度 + 外参)
  → SceneGraph 组装
  → fruit_queue（规则选 fruit_id）
  → pick_cycle_fsm / reach_fsm（fsm_state）
  → trajectory_planner_node（Transformer，1 Hz replan）
  → trajectory_executor_node（IK → 关节样条 → 臂控）
```

与现有 PBVS 关系：

- **P0–P2**：executor 与 `piper_position_ik` 对接；PBVS 仅作示教 teacher / 对比 baseline。
- **P3+**：pick_cycle 簇对齐、触达均走统一 planner；PBVS 可退役或仅作安全监控。

---

## 2. Tool 帧 6D 约定

### 2.1 坐标系

| 项 | 约定 |
|----|------|
| 父帧 | `base_link` |
| 参考点 **position** | **tool tip**（吸盘杯心 / 触达点），非 link6 原点 |
| **approach_axis** | tool 帧 **+Z** 在 `base_link` 下的单位向量（指向接近/触达方向） |
| 维度 | 6：`[x, y, z, ax, ay, az]`，`‖approach_axis‖ = 1` |

与现有 `cup_axis_ik` 语义一致：位置约束 link6（或 EE），方向约束 cup 轴对准目标；模型直接在 **tool tip + tool +Z** 表达，换末端时只改标定。

### 2.2 与 link6 / 标定关系

```
p_tip = p_link6 + R_link6 · offset_tip_link6
a_tool_base = R_link6 · R_link6_tool · [0,0,1]ᵀ
```

`offset_tip_link6`、`R_link6_tool` 来自 `config/end_effector_profiles.yaml`（见 §5）。  
训练标签：示教 `joint_states` → FK → 上式得到 16×6 绝对路点。

### 2.3 5-DOF 臂说明

Piper X：**j6 解锁 ±165°**，完整 **j1–j6**。IK 以 `q_now` 为 seed 求最近解，**失败即 ERROR**（无启发式回退）。  
绕 approach 轴的自旋由 IK 零空间消化，模型不输出。

---

## 3. 模型 I/O

### 3.1 输入 `/planning/planner_input`

由 orchestrator（`trajectory_planner_node` 内或独立 assembler）每 replan 组装一次。

```yaml
header:
  stamp: t0
  frame_id: base_link

scene:
  clusters[]:          # 最多 8
    id: int
    position: [x,y,z]  # base_link, m
    bbox_uv: [x1,y1,x2,y2]  # 固定相机，可选
    confidence: float
  berries[]:           # 最多 10；无则空
    id: int             # track_id 或稳定 index
    position: [x,y,z]   # base_link 表面/中心点
    bbox_uv: [u0,v0,u1,v1]
    image_uv: [u,v]
    center_depth_m: float   # 相机光轴深度，优先 RGB-D
    surface_normal: [nx,ny,nz]  # base_link 单位法向；朝外/朝相机
    normal_valid: bool
    confidence: float
    visible_wrist: bool
    visible_global: bool
    pick_role: enum     # PENDING=待采集 | ACTIVE=采集目标 | DONE=已采
    # diag: depth_mode, z_depth_m, z_mono_m, track_source
  obstacles[]:         # DEPRECATED — use occupancy_local
  occupancy_local:     # 32³ ESDF crop around tip/fruit
    origin_xyz / voxel_m / size_xyz / labels / esdf

ego:                   # 机械臂+末端（非障碍）；仿真 free 用补集
  q_rad: [6]
  qd_rad: [6]
  tip_xyz: [3]
  approach_axis: [3]
  ee_radius_m / link_xyz[18] / link_radius_m
  workspace_center / workspace_radius_m / workspace_z_min  # 上半球
  workspace_aabb: [xmin,xmax,ymin,ymax,zmin,zmax]          # 半球外接盒（兼容）

task_context:
  fsm_state: enum      # 见 §6
  cluster_id: int      # LOCK 后 ≥0；-1 无效
  fruit_id: int        # fruit_queue 规则选出；与 ACTIVE berry.id 一致；-1 未采果

end_effector:
  tool_type_id: int    # profile 枚举
  profile_name: string # 如 suction_cup_v1
  tip_offset_link6: [3]
  # 可扩展：approach_standoff_m, cup_radius_m, T_link6_tool_rpy, ...
```

**选果**：[`fruit_queue.py`](../scripts/fruit_queue.py) — `filter_berries_in_cluster` + `sort_berries_near_to_far`；FSM 写入 `fruit_id`，模型不输出选果。  
Assembler 将 `fruit_id` 对应 berry 标为 **`pick_role=ACTIVE`（采集）**，其余为 **`PENDING`（待采集）**（最多 10 颗进模型）。模型 / PBVS teacher **只接近 ACTIVE** 那颗。

### 3.2 输出 `/planning/tool_trajectory_4s`

```yaml
header:
  stamp: t0              # 轨迹起点
  frame_id: base_link
seq: uint64

horizon_s: 4.0
dt_s: 0.25
execute_until_index: 4   # executor 只跟踪 waypoint[0..4]

reference: tool_tip
waypoints[16]:
  - position: [x, y, z]           # m，绝对
    approach_axis: [ax, ay, az]   # 单位向量，绝对
    time_from_start: k * 0.25     # k = 0..15
```

**replan 周期：1.0 s** — 每 1 s 发布新轨迹；`waypoint[5..15]` 丢弃。

---

## 4. 执行层（trajectory_executor）

### 4.1 单周期流程

```
1. 订阅 tool_trajectory_4s，取 w0..w4
2. w0 对齐：若 FK(q_now) 与 w0 偏差大，用 q_now 作 IK seed，首点替换为实际 tip pose 或短 blend
3. 对每个 tool 6D 路点：
     tool_pose_ik(position, approach_axis, seed=q_prev, tip_offset)  # 6-DOF
     → q[6]；失败 → 上报 /planning/executor_status ERROR
4. 5 组关节角 → 三次样条 / minimum-jerk，满足 qd_max / qdd_max
5. 1 s 内跟踪；到时等待下一 replan
```

### 4.2 IK 接口（待实现 `tool_pose_ik`）

基于 [`piper_position_ik.py`](../scripts/piper_position_ik.py) 扩展，与 `cup_axis_ik` 同构，但**方向目标来自模型 `approach_axis`**，而非 `berry - tip`：

```python
def tool_pose_ik(
    target_tip_xyz: Sequence[float],
    approach_axis_base: Sequence[float],  # 单位向量，tool +Z
    seed_q: Sequence[float],
    *,
    tip_offset_link6: Sequence[float],
    ...
) -> Optional[List[float]]:
    """5-DOF: link6 位置 + cup 轴 ∥ approach_axis；tip 在 target_tip_xyz。"""
```

触达阶段若仍有 `fruit_id`，可将 fruit position 仅用于 **IK 辅助验证**（轴与 tip→berry 夹角门控），不作为 planner 输出。

### 4.3 安全门控

| 条件 | 动作 |
|------|------|
| IK 任一路点失败 | FSM → ERROR |
| `‖Δq‖` 或 tip 跳变超阈值 | ERROR |
| 关节超限 | clamp + WARN；连续失败 → ERROR |

---

## 5. 末端夹具配置

`config/end_effector_profiles.yaml`：

```yaml
suction_cup_v1:
  tool_type_id: 0
  mode: suction
  tip_offset_link6: [0.0, 0.01883, 0.06152]
  approach_standoff_m: 0.003
  press_in_m: 0.0          # 吸盘不压入

agx_gripper_v1:
  tool_type_id: 1
  mode: gripper
  tip_offset_link6: [-0.024259, 0.017894, 0.132617]  # 2026-09-01 GT 标定
  approach_standoff_m: 0.008
  pre_grasp_width_m: 0.055 # 接触前微开
  grasp_width_m: 0.012     # 轻夹卡住果实
  grasp_force_n: 1.5
  press_in_m: 0.008        # 沿 approach 越过接触点再合爪
  retract_m: 0.050
```

启动参数 `--end-effector agx_gripper_v1`；换末端只改 yaml + 标定，**不改模型结构**（`tool_type_id` embedding 区分品类）。

### 5.0 模型学什么 vs 规则做什么

| 层 | 职责 |
|----|------|
| **模型** | 合拢（或吸盘）状态下 **tip → 接触点** 的接近轨迹（`tool_trajectory_4s`） |
| **规则** | 多夹具差异：夹爪 OPEN / 压入 / CLOSE / RETRACT；吸盘开吸 / 断吸等 |

当前 P0 验收：**先跑通 close 态指尖碰到接触点**；open→压入→close **暂不接**。

冒烟：`scripts/tip_touch_smoke.py`（需 `trajectory_executor --drive-arm`）。

完整夹爪规则序列（后接，非模型）：

| 步骤 | 吸盘 `suction` | 夹爪 `gripper`（规则） |
|------|----------------|------------------------|
| 接近 | tip → 果表面（模型） | tip → 果表面（模型，**合拢**） |
| 接触后 | 开吸 | OPEN → 压入 `press_in_m` → CLOSE |
| 回撤 | +n 退 `retract_m`，断吸 | 夹紧退 `retract_m`，落果位 OPEN |

手动开合：`python3 scripts/gripper_cmd.py --open|--close`。

### 5.1 自由空间：Occupancy + ESDF（业界语义）

**工作空间** = `base_link` **上半球**（\(R=0.85\,\mathrm{m}\)，\(z\ge 0.02\)），不是业务 AABB。  
**双目晚融合**：固定 DaBai + 腕部 Gemini 各自 TF→同一 log-odds 体（OctoMap/Voxblox 风格）；腕部近距加权更高。  
**不**单独语义分割桌/盆——其余深度命中即接触面 OCC。

```text
1. FK 胶囊链 → EGO 掩膜（深度自滤）
2. 固定+腕部 depth → 共享 log-odds（hit/miss，带权，可衰减）
3. ACTIVE 果 → BERRY；其余果 → OCC；簇不参与标签
4. ESDF(p) = 到最近 OCCUPIED 的欧氏距离
free(p) ⇔ in_hemisphere ∧ label∈{FREE,BERRY?} ∧ ESDF≥margin
```

| 类别 | 体素标签 | 说明 |
|------|----------|------|
| 深度命中（植株/盆/桌面/背景/非 ACTIVE 果） | OCCUPIED | 双目融合 |
| 射线清空 | FREE | |
| 未见 / 球外 | UNKNOWN | 保守：不可进 |
| 机械臂+末端 | EGO | URDF FK 胶囊 + tip 标定 |
| ACTIVE 果实 | BERRY | 可触达 |

**话题 / 模型输入：**

| Topic / 字段 | 内容 |
|--------------|------|
| `/perception/occupancy_esdf` | **结构化真值**：全半球 labels+ESDF（分析/bag/模型） |
| `/perception/occupancy_local` → `planner_input.occupancy_local` | tip/果附近 32³ crop |
| `/planning/viz/occupancy_cloud/occupied` | 3D 点云：不可通行 OCC（红） |
| `/planning/viz/occupancy_cloud/free` | 3D 点云：可通行 FREE（绿） |
| `/planning/viz/occupancy_cloud/ego` | 3D 点云：臂体 EGO（黄） |
| `/planning/viz/occupancy_cloud/berry` | 3D 点云：ACTIVE 触达 BERRY（品红） |
| `/planning/viz/occupancy_markers` | 半球线框 + ACTIVE 球 + base 轴 |
| `/planning/viz/occupancy_overlay` | 固定 RGB 叠点（辅助，非主视图） |

**3D 查看：** `bash scripts/view_occupancy_3d.sh`（RViz，`Fixed Frame=base_link`）。  
节点：[`scripts/occupancy_map_node.py`](../scripts/occupancy_map_node.py)。  
仿真：[`planner_sim_env.is_free`](../scripts/planner_sim_env.py) 查 ESDF/体素。

---

## 6. FSM 与阶段切换（规则，非模型）

### 6.1 `fsm_state` 枚举（planner 输入）

| 值 | 含义 | 典型 scene |
|----|------|------------|
| `CLUSTER_ALIGN` | 粗对齐簇 | 有 cluster，腕部果少/无 |
| `APPROACH_FRUIT` | 触达当前果 | cluster + fruit_id + 腕部可见 |
| `RETRACT` | 回撤 | fruit_id 仍有效，standoff 增大 |
| `HOLD` | 保持 / 等待人工 | 可选 |
| `IDLE` | 不 replan | planner 可停 |

### 6.2 进入采果循环（果与簇关联）

满足**全部**时，外层 FSM 从 `CLUSTER_ALIGN` → 采果循环，并写入第一个 `fruit_id`：

1. `cluster_id` 已锁定；
2. global + wrist 中，≥1 颗果的 `base_link` 位置落在簇中心 `radius_m` 内（默认 0.25 m，同 `filter_berries_in_cluster`）；
3. 腕部检测 `fine_visible` 或 berry `visible_wrist == true`。

### 6.3 与现有节点映射

| 现有 | 迁移后 |
|------|--------|
| `reach_fsm` ALIGN / PLAN | 订阅 executor 完成 + `fsm_state=CLUSTER_ALIGN` |
| `refine_entry_pose.json` | P2 前仍作 CLUSTER_READY 快照；长期由轨迹首点替代 |
| `pick_cycle_fsm` TOUCH | `fsm_state=APPROACH_FRUIT` + replan 触达 |
| `vlm_align_client` / `bc_align_policy` | 由 planner 替换 |

---

## 7. 模型结构（自研，不迁移 Plannn2）

### 7.1 规格

| 项 | 值 |
|----|-----|
| 输入 | 结构化 token（无原始图像进模型） |
| 骨干 | Transformer，d=256, L=4, heads=8，约 3–6M 参 |
| 输出 | 唯一主头：`TrajHead` → `[16, 6]`（绝对 tool 6D） |
| 训练 | 阶段 A：**BC**（PBVS teacher）；阶段 B：**离线仿真 RL**（§7.4） |

### 7.2 Token 布局

```
ClusterToken × ≤8
BerryToken   × ≤10  （mask 无效槽）
OccupancyEncoder  # 必选：32³ labels+ESDF crop → CNN/MLP → OccupancyToken
EgoToken     × 1     q + tip + links + workspace_aabb
TaskToken    × 1     embed(fsm_state, cluster_id, fruit_id)
ToolToken    × 1     embed(tool_type_id) + MLP(calib vector)

→ Transformer → 16 个 causal query slots → TrajHead
```

**无 `occupancy_local` 不得训、不得推**（`input_flags` 含 `no_occupancy` 则跳过该帧）。旧 ObstacleToken 球已废弃。

### 7.3 训练标签（BC 冷启动）

- 来源：成功 episode 的 bag：**`planner_input`（含 `occupancy_local`）** @ t0 + PBVS/executor 的 `tool_trajectory_4s` 或 `joint_states`；
- 另录全图 `/perception/occupancy_esdf` 供 sim/审计（模型主吃 local crop）；
- 对齐 `t0`，重采样 16 点 @ 0.25 s；
- FK → tool tip + tool +Z → 6D 绝对标签；
- 数据增强：同一 episode 随机多个 `t0`（模拟 replan）；
- **不要求人手示教**；PBVS + executor 即自动 teacher；
- **过滤**：`occupancy_local` 空 / `no_occupancy` 帧不进 BC。

### 7.4 为何需要模型：PBVS 上限 + 离线仿真 RL

**PBVS = 冷启动 teacher，不是终局。**

| 能力 | PBVS | TrajHead |
|------|------|----------|
| 单果局部闭环 | 强 | 可复现 |
| 全局 replan、ALIGN+触达统一 I/O | 弱 | 强 |
| 最短路径 / 多障碍权衡 | 弱 | 可优化 |
| 感知噪声、多果排队 | 脆弱 | 可学 |
| 新摆位泛化 | 需重调 | 数据驱动 |

```
阶段 A  真机 PBVS/executor 录 bag → BC 初始化
阶段 B  离线 sim RL：planner_input 作环境真值 → 静态占据 + FK rollout
阶段 C  真机验收；失败 case 可选回流仿真
```

**离线仿真（工程可控）：**

```text
s_t = planner_input(t0)          # clusters, berries, occupancy_local, ego — 真值
a_t = policy(s_t) → traj[0:5]  # 1 s 窗口
rollout: FK tip 折线 + 障碍球碰撞 + tool_pose_ik
r_t = -w1·dist(goal) - w2·path - w3·collision - w4·ik_fail + success
```

静态场景、无照片级渲染、无完整物理引擎。RL 以 **离线**（IQL/CQL + BC 正则）为主，真机不做在线探索。

---

## 8. Topic 一览

### 8.1 规划 / 执行主链（须进 bag）

| Topic | 类型 | 说明 |
|-------|------|------|
| `/planning/planner_input` | `PlannerInput` | **模型完整输入**（含 `occupancy_local` 32³） |
| `/planning/tool_trajectory_4s` | `ToolTrajectory4s` | 模型输出：tool 6D × 16 |
| `/planning/ik_joint_trajectory` | `IkJointTrajectory` | **IK 解算结果**：与 tool 轨迹对齐的关节路点 + 每点 `ik_ok` / tip FK |
| `/planning/joint_cmd` | `trajectory_msgs/JointTrajectory` | **实际下发**样条后的关节指令（通常仅 w0..w4 / 1 s 窗口） |
| `/planning/executor_status` | `ExecutorStatus` | `ok` / `ik_fail` / `jump_reject` / 跟踪残差 |
| `/feedback/joint_states` | `sensor_msgs/JointState` | 真机反馈（对比 IK / cmd） |
| `/joint_states` | 同上（relay 别名） | 可选，与 feedback 二选一即可 |

Debug 对照链：

```
tool_trajectory_4s  →  ik_joint_trajectory  →  joint_cmd  →  feedback/joint_states
     (模型)                (IK+FK残差)           (样条下发)         (实际)
```

### 8.2 感知 / 编排（须进 bag）

| Topic | 类型 | 说明 |
|-------|------|------|
| `/perception/scene_graph` | `PerceptionScene` | **统一感知输出**：`clusters[]` + `berries[]`（稳定 track id） |
| `/perception/occupancy_esdf` | `OccupancyEsdf` | **全图占据+ESDF**（sim/审计；须进 bag） |
| `/perception/occupancy_local` | `OccupancyLocal` | **模型用 32³ crop**（亦嵌在 `planner_input`） |
| `/perception/global/berries` | `DetectedBerryArray` | 原始簇（debug，对照 scene_graph） |
| `/perception/fine/berries` | `DetectedBerryArray` | 原始腕部果（debug） |
| `/planning/task_context` | `PlannerTaskContext` | fsm / cluster_id / fruit_id |
| `/pick/status` | `String` | pick_cycle 状态 |
| `/reach/status` | `String` | reach FSM 状态 |
| `/reach/cmd` | `String` | 内层 reach 命令 |

**Tracking：**  
- 簇 / 果：世界系 **3D 关联为主**（`prefer_xyz`）  
- **漏检 coast**：短时仍发布上次 `base_link` 位姿、**同一 track_id**；`track_source=coast`（果同时 `depth_mode=coast`）；`flags` bit4=`has_coast`  
- 再检出且 3D 距离门内 → 接回原 id，`track_source=live`  
- bag / viz / sidecar 均带 `track_source`，便于区分检出更新 vs coast  

Assembler **优先**读 `/perception/scene_graph`，无则回退 global/fine。

### 8.3 可视化（可选进 bag）

| Topic | 说明 |
|-------|------|
| `/planning/viz/fixed` `/wrist` `/hud` `/status` | 叠图与摘要 |
| `/planning/viz/occupancy_slice` | 占据俯视切片 QA |

`record_planner_bag.sh` / `--record-bag` 默认录 **8.1 + 8.2（含 occupancy）+ viz**。

---

## 9. 实施阶段

| 阶段 | 交付 | 验收 |
|------|------|------|
| **P0** | **输入数据链**：msg、`planner_input_assembler`、缺口修复（见 [PICK_TRAJECTORY_PLANNER_DATA.md](PICK_TRAJECTORY_PLANNER_DATA.md)） | `/planning/planner_input` @ 1 Hz，字段审计表无阻塞项 |
| **P0b** | `tool_pose_ik` 雏形、executor、PBVS→`tool_trajectory_4s` | 臂能跟 1 s 窗口关节样条；bag 有 ik/joint_cmd |
| **P1** | 障碍填充 + PBVS bag BC 初值 + **离线静态 sim RL v0** | `obstacles[]` 进 input；仿真里短路径+避障优于 PBVS baseline |
| **P2** | `APPROACH_FRUIT` 同一模型 + pick_cycle 闭环 | 触达 `fruit_id`；RL 奖励含 approach/触达；真机采 1 颗 |
| **P3** | 去 PBVS 主路径、launch 统一 | `run_pick_system.sh` 默认 planner |

**P0 已落地文件**

| 路径 | 说明 |
|------|------|
| [`docs/PICK_TRAJECTORY_PLANNER_DATA.md`](PICK_TRAJECTORY_PLANNER_DATA.md) | 数据流图、字段审计、改码清单 |
| [`scripts/planner_input_assembler.py`](../scripts/planner_input_assembler.py) | 1 Hz 组装 `/planning/planner_input` |
| [`scripts/planner_input_recorder.py`](../scripts/planner_input_recorder.py) | 可视化；可选 `--record-bag` |
| [`scripts/record_planner_bag.sh`](../scripts/record_planner_bag.sh) | **训练用** rosbag 录 `/planning/planner_input` |
| [`scripts/end_effector_profile.py`](../scripts/end_effector_profile.py) | 末端 profile 加载 |
| [`config/end_effector_profiles.yaml`](../config/end_effector_profiles.yaml) | 默认吸盘标定 |
| `picking_msgs/msg/PlannerInput.msg` 等 | 结构化 I/O |

```bash
cd blueberry_picking_ws
colcon build --packages-select picking_msgs picking_perception --symlink-install
source install/setup.bash
export PYTHONPATH="$(pwd)/scripts:${PYTHONPATH}"

# 1) 输入链
python3 scripts/planner_input_assembler.py --end-effector suction_cup_v1

# 2) P0b 执行链（PBVS 作 teacher 轨迹源）
bash scripts/run_trajectory_executor.sh --drive-arm &
python3 scripts/reach_fsm_node.py --use-pbvs --pbvs-mode single --pbvs-via-executor ...

# 影子模式（只发 joint_cmd，不驱臂）：
bash scripts/run_trajectory_executor.sh

ros2 topic echo /planning/ik_joint_trajectory --once
ros2 topic echo /planning/executor_status --once
```

---

## 10. 数据采集要点

- 脚本：扩展 [`align_data_collector.py`](../scripts/align_data_collector.py) → `trajectory_episode_recorder.py`；
- 每 replan 周期存：`planner_input`（**必含 `occupancy_local`**）+ `tool_trajectory_4s` / `joint_states` + 派生 6D 标签；
- 同步录 `/perception/occupancy_esdf`（全图）便于离线 sim 与 QA；
- **BC 数据**：成功 PBVS+executor episode（≥100 bag 片段），非人手示教；
- **RL 数据**：同上 bag 作离线 replay buffer；仿真增广随机 `t0`、障碍半径扰动；
- coarse / fine 用 `fsm_state` 区分；失败 episode 不进 BC，可进 RL 负样本；
- 与 §5.1 同步：录 bag 时逐步写入 `obstacles[]`（先 viz 验收再进训练）。

---

## 11. 相关文件

| 路径 | 角色 |
|------|------|
| [`scripts/trajectory_executor_node.py`](../scripts/trajectory_executor_node.py) | `tool_trajectory_4s` → IK → `/planning/joint_cmd` |
| [`scripts/run_trajectory_executor.sh`](../scripts/run_trajectory_executor.sh) | 启动 executor（`--drive-arm` 真机） |
| [`scripts/tool_trajectory_utils.py`](../scripts/tool_trajectory_utils.py) | PBVS/teacher 构建 4s tool 轨迹 |
| [`scripts/occupancy_map.py`](../scripts/occupancy_map.py) | OccupancyVolume + ESDF |
| [`scripts/occupancy_map_node.py`](../scripts/occupancy_map_node.py) | `/perception/occupancy_esdf` + local crop |
| [`scripts/obstacle_extractor_node.py`](../scripts/obstacle_extractor_node.py) | **DEPRECATED** 球体 |
| [`scripts/planner_sim_env.py`](../scripts/planner_sim_env.py) | 离线静态 sim（occupancy ESDF） |
| [`scripts/planner_rl_smoke.py`](../scripts/planner_rl_smoke.py) | RL reward loop 冒烟 |
| [`scripts/fruit_queue.py`](../scripts/fruit_queue.py) | `fruit_id` 规则 |
| [`scripts/pick_cycle_fsm_node.py`](../scripts/pick_cycle_fsm_node.py) | 外层编排，`fsm_state` 来源 |
| [`scripts/reach_fsm_node.py`](../scripts/reach_fsm_node.py) | 内层触达，逐步改为 executor 驱动 |
| [`docs/PICK_CYCLE.md`](PICK_CYCLE.md) | 采摘循环总览 |

---

## 变更记录

| 日期 | 内容 |
|------|------|
| 2026-09-01 | 半球 workspace + 双目 log-odds 晚融合；EGO 胶囊；仅 ACTIVE=BERRY |
| 2026-09-01 | 固定相机外参 FK 骨架贴合：`t=(-0.72,0.28,0.20)` |
| 2026-09-01 | Occupancy+ESDF freespace（固定 depth 射线）；废弃障碍球主路径 |
| 2026-09-01 | obstacles≤12 球 + ego(tip/links/AABB)；free=补集查询 |
| 2026-09-01 | berries≤10 + `pick_role` ACTIVE/PENDING；fine 默认不全量 pin 成 1 |
| 2026-09-01 | §5.1 障碍语义；§7.4 PBVS 冷启动 + 离线 sim RL；P1/P2 去「人手示教」 |
| 2026-09-01 | 漏检世界系 coast：短时保 id + 再检出 3D 接回 |
| 2026-09-01 | 固定/腕部 pose 统一 RGB-D；取消 silent mono invent |
| 2026-09-01 | 统一 `/perception/scene_graph`；簇/果 realtime tracking；bag 录原始感知 |
| 2026-09-01 | bag 增加 IK 关节轨迹 / joint_cmd / executor_status 录制约定 |
| 2026-09-01 | berries：bbox/深度/法向；recorder+viz；P0 数据闭环 |
| 2026-08-31 | P0：输入数据框架文档 + PlannerInput msg + assembler |
| 2026-08-31 | 初稿：tool 6D 输出、1 s replan、绝对位姿、FSM/fruit_id/tool 作输入 |
