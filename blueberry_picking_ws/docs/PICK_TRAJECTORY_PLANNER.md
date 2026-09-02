# 学习型轨迹规划器（Pick Trajectory Planner）

统一替换 ALIGN / 簇对齐 / 触达路径上的 **VLM / 启发式 / 单步 BC**，与 [`PICK_CYCLE.md`](PICK_CYCLE.md) 编排层对接。

**设计原则（已定稿）：**

- 模型**唯一输出**：tool 帧 **6D 绝对位姿轨迹**（position + approach_axis），所有 FSM 阶段格式相同。
- **1 s replan**，**4 s horizon**，16 路点（dt=0.25 s），每周期只执行前 5 点（0~1 s）。
- **选果、阶段切换**由规则 + 外层 FSM 完成；`fruit_id`、`fsm_state`、`tool_profile` 为模型**输入**。
- **关节角、样条、限速、跟踪**由传统执行层（IK + 控制）完成，模型不输出关节。

---

## 0. 现行方案（2026-09-02 冻结 · 动代码前）

本节覆盖此前所有占用栅格 / YOLO 选果 / agent 锁区 / 「连通域大球」混写。**目标系统以本节为准**；§1 之后未改完的 occupancy、YOLO、`CLUSTER_ALIGN` 视为遗留兼容。Teacher 触达在 **M3 部署前**仍走 [`REACH_PIPELINE.md`](REACH_PIPELINE.md) 的 PBVS；接触段 tracking 必须单独过 **T1/T2**，「碰到了」不算感知过关。

### 0.1 一句话

固定 DaBai + 腕部 Gemini 做 RGB-D。冻 DINOv3 按实例写成 SceneObject。assembler 汇总成世界表，再从 ACTIVE 的 3D 算出接近轴与夹爪开合。规划器吃结构化 token，输出 tool 6D。

### 0.2 分层

```text
┌─────────────────────────────────────────────────────────┐
│ 传感器  固定 DaBai RGB-D     腕部 Gemini 305 RGB-D     │
└─────────────────┬───────────────────┬─────────────────┘
                  ▼                   ▼
         DINOv3 语义+实例（同一套头，两路相机）
                  │
                  ▼
         SceneObject[]  每实例：class, polygon_uv,
                         center/r/contact/normal, geometry, attributes
                  │
         融合（P3）：两路全量进世界；空间重合并点云；不重合都保留
                  │
    ┌─────────────┼──────────────┬──────────────┐
    ▼             ▼              ▼              ▼
 berries[]    obstacles[]      residual     ego = URDF FK
 (单颗果)     枝+硬物几何      挖除后深度    （图像 ego 不抬升）
    │             │              │              │
    ▼             ▼              ▼              ▼
 fruit_queue  ObstacleToken   软障碍       EgoToken
 写 fruit_id
    │
    ▼
 GraspRequest ← ACTIVE 3D + 邻域 + 夹爪极限
    │
 FSM（规则）
   SELECT_FRUIT → APPROACH_STANDOFF → APPROACH_FRUIT → RETRACT
    │
    ▼
 Transformer TrajHead → tool 6D × 16  →  IK 样条 → 臂
```

叶子不标一类。未落入 berry/branch/rigid/ego 的**有效深度**做成 **residual 软障碍**（仿真扣分、真机轻擦不算失败）。`ego` 分割只用于挖掉自身像素/监督，规划器避碰用 FK。

### 0.3 技术选型（选什么、为什么、不选什么）

| 层 | 选定 | 原因 | 明确不选 |
|----|------|------|----------|
| 场景表示 | **实例 SceneObject**（2D polygon + RGB-D 几何 + 属性） | 规划/选果/RViz 要的是物体，不是像素，也不是 2 cm 体素外皮 | 半球 occupancy 主路径；整簇 `fit_sphere` |
| 语义骨干 | **冻 DINOv3 ViT-S** + 分割头 | 10 张已能分四类；权重小；与标注「多边形 mask」一致 | 再训检测器；DINOv2；为枝/电钻伪造 CAD 跑 FoundationPose |
| 单果实例 | **同一骨干上的实例解码**，不挂 YOLO | 标注已是每颗多边形；粘连是监督被压成语义类，不是骨干能力不足 | 新主路径再加 berry YOLO 框 |
| 实例落地顺序 | **先几何切分，再实例头** | 见 §0.4 | 一上来 Mask2Former / 大数据实例分割 |
| 果的 3D | 切开后：球心 + 半径/椭球 + 可见面接触点 + 法向 + 邻域 | 浆果近球；接触用可见面；**接近轴与开合从这张 3D 表规划** | 粘连连通域一个球当 `fruit_id`；用 2D 框宽当开合；yaml 常数当开合指令 |
| 枝的 3D | 连通域 → 骨架折线 + 半径，或胶囊链；可选沿骨架切片 polygon | 细枝要轴向，不要冠层团 | occupancy 膨胀半径当枝 |
| 硬物 3D | 连通域凸包或 OBB；盆沿可用 z 切片 | 桌/盆/电钻是块状 | 无 CAD 的 6D 位姿估计 |
| 自身 | **FK 胶囊**；图像 class=ego 挖深度/监督 | 臂位姿比分割准 | 把吸杯 mask 抬升成障碍 |
| 双目 | 两路全量进世界；空间重合并；不重合都留 | 90° 互补轮廓要并起来；腕部独有检出不能丢 | 预设腕部只留锁定果；超门就把腕部几何扔掉；整幅只信一路 |
| 选果 | `fruit_queue` 规则（近、可达、未挡、未 DONE） | 模型不输出选果；全局已有方位 | agent 点区域（仅备份到 C1） |
| 接近 | **standoff lookat**（12–20 cm）再用腕部 | 全局 3D 不够当接触点 | 入口关节 `refine_entry_pose` 当对准该果 |
| 触达 | **TrajHead 一路出到接触点**；PBVS 只录 teacher bag | 腕部修正必须进每一拍 `planner_input`；关联失败则 HOLD 不碰 | 全局质心当接触；未 QA 改旧 REFINING 公式当部署主路 |
| 规划器 | 结构化 token Transformer，输出 tool 6D | 无图进模型；BC←PBVS，再离线 sim RL | 模型出关节；VLM 对齐 |

**单果实例为何不选 YOLO：** 检测框解决的是「哪有一块果」，DINOv3 语义已经给出 **果区（簇）**。缺的是簇里面的 **id**。YOLO 与分割双栈会抢同一监督。颗头叠在同一骨干上，圆只在 berry 语义门内生效。

**单果怎么从簇里出来：** 语义头继续出整块品红（簇，挡误检）。颗头出果心热力图，只与 berry 像素求交。圆心/半径来自 **贴果皮实例 mask** 的矩（质心 + 等面积圆作量膜），不是规则射线圆。**禁止** watershed 把语义团切成碎多边形当果皮。簇里没有颗的像素仍是果区，不是 `fruit_id`。

**夹爪不另开感知轨，但开合也不是 yaml 常数：** 现行末端 `agx_gripper_v1`。profile 只给 **机械极限**（最大开口、最小闭合、指厚、tip 标定）。**接近轴、jaw 开合方向、开口/闭合宽度、压入深度** 都从 P3 世界表里 ACTIVE 果的 3D + 邻果/近枝算出来（§0.4.1）。不要 YOLO 框，也不要把 `pre_grasp_width_m=55mm` 当成指令。

**占用栅格为何退出主路径：** 细枝在深度里只有几个像素，2 cm voxel 会胀成团，polygon 只是那团的皮。语义实例按 RGB 看见的枝/盆切开，再抬升，几何才和眼睛一致。

**DINOv3 给什么、几何抬升给什么、规则给什么（不要上 FoundationPose）：**

同一冻 DINOv3：**语义头**分果区/枝/硬/ego；**颗头**在果区里出果心；2D 轮廓来自实例 mask。3D 球/椭球+法向是 **P3 抬升**，不是 FoundationPose，也不再把簇切成 Voronoi。

| SceneObject 字段 | 谁出 | 不是谁 |
|------------------|------|--------|
| class（果/枝/硬/ego） | DINOv3 语义 | — |
| 单果 id、2D | 颗头果心 + 贴皮实例 mask（门=berry 语义） | YOLO 框；watershed 切团；FP |
| xyz、尺度、接触 | 实例 mask 内 RGB-D 反投影（§0.4.1：contact + r → center） | DINOv3 不产米制；YOLO 框当果径；可见点均值当球心 |
| 朝向 | **几何**：果=可见面法向；枝=骨架切向；硬物=OBB 主轴 | FoundationPose 6D |
| 接近轴 / 开合 | **GraspRequest**：assembler 从 ACTIVE 3D+邻域+极限算 | yaml 常数；2D 框宽 |
| 帧间 `fruit_id` / coast / 腕部覆盖 | **规则**（3D 门、live/coast） | DINOv3 是单帧，无时间 |
| ego 位姿 | FK | 分割 mask |

FoundationPose 适合「已知 CAD 的刚体 6D」。这里没有枝/电钻网格，浆果自旋无意义。深度相机 + 实例 mask 已经能出接触用的表面点与法向。


### 0.4 Checkpoint（两轨并行，接触感知单独过）

PBVS 若能稳定触达，teacher bag 可以先扩到 **≥100 条**（**BAG**），不必等 DINOv3 全部完成。但 **「碰到了」≠ tracking/感知准**：接触过程必须单独走 **T1/T2**，不过的 episode **不得进 BC/RL**。

```text
轨 P 场景建模（DINOv3）              轨 T+D 真机 teacher（可先跑）
P1 单果实例 + polygon                 D1 现 REACH/PBVS 能重复触达
P2 实例 GT 不压成类图                 T1 接触段 fruit_id 不跳、不抢邻果
P3 世界表 + 腕部锁定果融合           T2 接触段 xyz/法向精度（live，非 coast）
P4 运动中锁定果位置修正              BAG ≥100 成功 bag（过 T 才进训练集）
        \                            /
         C1 全局选果  S  standoff 后腕部看见该果
                    M1 BC 模仿到接触 → M2 离线 RL → M3 部署切主路
```

| ID | 轨 | 做什么 | 通过标准（可测） |
|----|----|--------|------------------|
| **P1** | 感知 | berry 按颗切开 + `polygon_uv`；重叠优先 3D 聚类 | 一簇 ≥2 轮廓贴单颗；无检测器；一团品红一大球 = 未过 |
| **P2** | 感知 | 训练保留每颗 polygon / instance map | 不再把全部果压成像素 1 |
| **P3** | 感知 | 双目全量世界表；重合并、不重合都留；assembler；GraspRequest | 腕部独有检出出现在世界表；重合实例切片变完整；不因 fruit_id/门限丢掉腕部几何 |
| **P4** | 运动中修正 | 接近时持续用腕部改写 **锁定果** xyz（live/coast、id 不跳） | 与 T1/T2 同一套：最后 15 cm live；id 切换=0 |
| **D1** | teacher | 现 `REACH_PIPELINE` PBVS 重复触达 | 同一流程可连跑成功；不改接触公式 |
| **T1** | **接触 tracking** | 从腕部锁定到接触：`fruit_id` 连续 | **一次接近中 id 切换 = 0**；无邻果抢锁；`track_source=coast` 不得在最后 15 cm 当 goal |
| **T2** | **接触感知精度** | 最后 15 cm 的 ACTIVE xyz/法向 | 最后 15 cm **live 帧 ≥ 80%**；帧间 xyz 跳变中位有记录（草案 < 8 mm，用 bag 校准）；接触瞬间 tip↔果皮残差进 sidecar；**不过 T2 的 bag 不进 BC** |
| **BAG** | 数据 | 成功 episode ≥100 | 每条含 tip 轨迹 + 同步 `planner_input`（字段随 P 轨变完整）；**仅 T1∧T2 通过者**入训练集 |
| **C1** | 任务 | `fruit_queue` 吃全局单果 | 不依赖腕部可见、不点区域 |
| **S** | 任务 | `APPROACH_STANDOFF` lookat | 到位后腕部检出 ACTIVE（过 P3 关联门） |
| **M1** | 模型 | TrajHead BC 模仿 PBVS 到接触 | 离线回放：过 T 的 bag 上 tip 误差不差于 teacher |
| **M2** | 模型 | 离线 RL（非静态 t0；SceneObject+residual） | sim 接触 goal = 腕部表面点；硬碰失败、residual 轻碰小罚 |
| **M3** | 部署 | TrajHead 主路；PBVS 退役 | 真机前仍离线 QA；`visible_wrist` 后加快 replan |

**并行规则：** D1 → 可立刻开 T 与 BAG（先用现有腕部 YOLO/depth 量 tracking）。P1–P4 做好后，**同一套 T1/T2 指标再跑一遍** DINOv3 实例（对照旧检测，不双栈部署）。M1 入口 = BAG≥100 且 T 通过，且 P3 融合字段已能写入 `planner_input`（没有腕部覆盖则 BC 学不到修正）。

#### 0.4.1 P3 世界表：结构 / 获得 / 汇总

P3 只产出 **事实**（每实例几何 + residual）。**接近方向、jaw 平面、开口/闭合、压入** 是规划量：assembler 从世界表算出 `GraspRequest`。模型和规则都只吃这张表，不要各开一套几何。

```text
固定 RGB-D / 腕部 RGB-D   （两路同一套 DINOv3）
        │
        ▼
 semantic {bg,berry,branch,rigid,ego}  +  berry_hm 果心
        │
        ▼
 实例图（P1/P2 已按颗切开；P3 不负责切簇）
        │
        ▼
 mask ∩ 有效深度 → K 反投影 → T_base_cam → base_link 点云
        │
        ├── berry / branch / rigid  → SceneObject（按类拟合）
        ├── ego mask                → 只挖深度，不抬升
        └── residual                → 四类挖掉后的有效深度
        │
        ▼
 WorldScene（两路全量；同类空间重合并点云；不重合都保留）
        │
        ▼
 assembler 投影 → planner_input.berries/obstacles/ego + GraspRequest
```

**两层，不要混：**

| 层 | 是什么 | 例 |
|----|--------|-----|
| 物体事实 | 这颗果在哪、多大、皮朝哪、旁边是谁 | `center` `r` `contact` `normal` 邻果 id |
| 抓取查询 | 这一拍打算怎么靠近、张多开、合多紧 | `approach_axis` `jaw_axis` `jaw_open_m` `jaw_close_m` `press_in_m` |
| 夹爪 profile | 硬件极限与标定 | `jaw_open_max` `jaw_close_min` `finger_thickness` `tip_offset` |

`bbox_uv` 只是 polygon 的包围盒，给 HUD。开合用 **3D 直径在 jaw 轴上的投影 + 邻域净空**，不用像面框宽。

##### 每类对象：字段与获得

**Berry（采摘单位，必须按颗）**

| 字段 | 获得 |
|------|------|
| `id` | 本帧 instance id；跨帧 `fruit_id` 用 3D 门（P4/T1） |
| `source_cam` `score` | 哪路相机；分割/热力图置信 |
| `centroid_uv` `polygon_uv` | 果心峰（berry 门内）；贴皮 mask 外轮廓 |
| `pts_xyz[]` | mask ∩ 有效深度；**像面**邻域深度一致才保留（局部 median/MAD），禁止用 `base_link` 高度帽/世界 z 去野点 |
| `n_pts` `xyz_std` `depth_valid` | 点云统计；点数过少则 `depth_valid=false` |
| `contact_xyz` | 可见面上离相机最近的一档点的中位，或过 `centroid_uv` 的深度射线 |
| `approach_normal` | 接触邻域 PCA 最小特征向量，翻成朝向该相机光心 |
| `r` | **不要**对半球点云做 mean 当球心。`r_img = sqrt(area_px/π)·z/fx`（z=接触深度）与该实例点云径向中位交叉。**禁止**夹到「蓝莓 6–18 mm」这种物种表 |
| `center_xyz` | `contact + approach_normal · r`（球心在皮后，不是可见点均值） |
| `axes` | 默认球；仅当 PCA 长短轴比明显才写椭球 |
| `occluded` | 果心像素射线先击中 branch/rigid |
| `neighbor_berry_ids` `gap_m` | 其他果心距离 < ~k·该果自身 `r`（相对尺度，不是 12 cm 邻域） |
| `pedicel_dir` | 若最近枝骨架点 < ~k·该果 `r`：`normalize(branch_pt − center)`（不是写死 4 cm） |

**Branch**

| 字段 | 获得 |
|------|------|
| 实例 | `sem==branch` 连通域（一般一根/一块，不必按果那样切颗） |
| `pts_xyz` | 同 berry 反投影 |
| `skeleton[]` | 点云 PCA 主轴折线（细枝）；分叉多再骨架细化 |
| `radius_m` | 点到骨架的中位距离（该枝自身），不要夹 3–25 mm 物种表 |
| `geometry` | 相邻骨架点 → 胶囊链；可选沿切向 `z_slices` |

**Rigid**

| 字段 | 获得 |
|------|------|
| 实例 | `sem==rigid` 连通域 |
| `geometry` | PCA OBB 或凸包；盆沿可沿 `base_z` 切片多边形 |
| 用途 | 碰撞、选果「别从盆/桌后穿」 |

**Ego**

| 字段 | 获得 |
|------|------|
| 图像 | `sem==ego`：从深度/residual **挖掉**（腕壳、夹爪、可见连杆），禁止抬成障碍 |
| 规划 | URDF FK 胶囊；两指间距用 **当前指令/编码器开口**，与 `GraspRequest.jaw_open_m` 一致 |

**Residual（软障碍，不是一类物体）**

| 字段 | 获得 |
|------|------|
| mask | 有效深度 ∧ `sem==bg` ∧ ¬ego |
| 表示 | 按该残差团自身尺度降采样（或少量 PCA 团）；不定实例 id |
| 用途 | sim 轻碰小罚；真机轻擦叶不算失败。禁止再胀成整株 occupancy |

**Cluster（派生，不是第 5 语义类）**

近邻 berry 的 3D 团聚或 berry 语义连通块的 AABB。只给 standoff / 选簇，**不能**当 `fruit_id`。

##### 双目全量融合（静态 P3）

两路检出都进世界表，**不预先规定腕部只留锁定果、其余丢掉**。

- **重合**（同类；果：质心距 ≤ 各自点云稳健 XY 半径之和；刚体/枝/残差：AABB 相交）：并点云再切片，`source=fused`。
- **不重合**：原样保留（`fixed` 或 `wrist`）。腕部看见、固定没看见的果/盆壁/枝都要留下。
- `fruit_id` / 接触门只给 ACTIVE 打标签，**不是建图过滤器**。运动中改锁定果 xyz 仍是 P4/T1。
- **飞点**：只在像面做局部深度一致；**禁止** `base_link` 高度帽（桌面 35 cm、花盆 20 cm、人 1.8 m 一律禁止）。稀疏高层用「相对该物体最密层」丢层，不是世界 z 阈值。

同图核对：`SceneSlices` 应同时有两路几何；`WristSlices` 是腕部原始抬升对照。

**过 P3 = 静态环境感知过关。** 下一步是 **P4 / T1：运动中持续改写锁定果 xyz**（live，最后 15 cm 不许 coast）。

##### 禁止场景尺度魔法数（P3 建图）

建图只允许三类数：**(1) 传感器近远裁**（来自驱动/`camera_info`，不是桌高）；**(2) 该实例自己的点云/mask 统计**（半径、AABB、层密度）；**(3) 算力/渲染帽**（最大切片数、多边形顶点数）。**禁止**把「这间实验室的桌子/花盆/蓝莓有多高」写进几何。

| 已否决（不要恢复） | 错在哪 | 替换 |
|--------------------|--------|------|
| 刚体 `z ∈ [median−8cm, median+35cm]` | 假定桌面刚体高度 | 像面 speckle + 相对层密度 |
| 世界 z MAD 打刚体/果 | 地面主导会杀掉立面 | 同上；世界坐标不做离群帽 |
| `fuse_gate_m=6cm`、`neighborhood_m=12cm` | 建图过滤器；腕部只修锁定果 | 两路全量；重合用实例自身半径/AABB |
| 按类 OVERLAP_PAD 2/5/8 cm | 按物种拍的米制门 | 果：`r_xy(a)+r_xy(b)`；其余 AABB |
| 果心距 2 cm、`BERRY_MAX_R=20mm`、`extent_split=35mm` | 假定蓝莓直径 | P1 切开用该 blob 自身尺度（2D watershed + 该团 extent / 自身 r） |
| 枝半径夹 3–25 mm、果 r 夹 6–18 mm | 物种表 | 该实例点云中位；不要全局夹 |
| `associate` 4–6 cm（P3 建图） | 把接触关联门拿来丢几何 | P3 不用这扇门。P4 关联同样用实例半径重叠，不用厘米常数 |
| occupancy `plant_radius=12cm`、`wrist_near_z=0.35` | 旧 occupancy 场景假设 | occupancy 已退出主路径，禁止抄回 P3 |

可保留但必须标明身份（不是场景语义）：

- `DEPTH_MIN/MAX`：传感器近远裁，应从驱动读，过渡默认不是「花盆高度」。
- `DZ_M` / `CELL_M` / `MAX_SLICES`：切片可视化分辨率 + 算力；大物体已被 `span/max_slices` 拉长。
- `MIN_PX` / `MIN_PTS_SLICE`：像面连通域噪点地板，与米制高度无关。
- RViz 球/箭头尺度：纯显示。
- 夹爪 `jaw_open_max` 等：硬件 profile，不是场景。
- standoff 退 12–20 cm：任务站位，不是建图过滤器。

##### 汇总（assembler → `planner_input`）

每拍一张世界表，再投影（token 定长，完整几何留 RViz / 碰撞）：

| 槽 | 从哪来 | 进模型的紧凑量 |
|----|--------|----------------|
| `berries[]` ≤10 | 单果 SceneObject，近→远，workspace 内，`depth_valid`，未 DONE | `id, center, r, contact, normal, occluded, pick_role, source_cam, xyz_std` |
| `obstacles[]` | branch 胶囊 + rigid OBB（**果不进障碍**，果走 berries） | `kind, pose, scale/axis` |
| residual | 挖除后点云 | sim 用；真机 token 可短 |
| `ego` | FK | `q, tip, approach_axis, link 胶囊`（指缝=当前开口） |
| `task_context` | FSM | `fsm_state, fruit_id` |
| **`GraspRequest`** | **ACTIVE 果事实 + 邻域 + profile 极限** | 见下；**随该果 3D 变，不是常数** |

##### 抓取查询（从事实规划，不是物体固有）

ACTIVE 那颗 + 其邻果/近枝 + 夹爪极限 → 一条查询。规则可以直接执行；TrajHead 目前仍出 tip 6D，但必须在观测里看见这些量，才能学接近方向。jaw 开合可先走规则，与轨迹同步。

| 量 | 怎么算 |
|----|--------|
| `contact_xyz` | 抄 ACTIVE 事实；全局质心禁用 |
| `approach_axis` | 初值：指向接触（或 `−normal`）。绕接近轴搜 **jaw roll**，使两指扫掠不穿邻果/枝/rigid；尽量 ⊥ `pedicel_dir`。腕部可见后用腕部法向重算 |
| `jaw_axis` | ⊥ `approach`；取邻果间隙最大的开合方向 |
| `jaw_open_m` | `clamp(2r + 2·finger_clearance, jaw_open_min, jaw_open_max)`；若 `open/2` 大于沿 `jaw_axis` 的邻果净空 → 换 roll 或放弃这颗 |
| `jaw_close_m` | `clamp(k·2r, jaw_close_min, jaw_close_max)`，k≈0.75–0.90 |
| `press_in_m` | ≈ `r` 减去指尖厚度，沿 approach 过赤道再合；**不是**写死 8 mm |
| `standoff_xyz` | 接触点沿 `−approach` 退 **该果 `r` 的若干倍**（任务站位，让腕部看见 ACTIVE）。禁止写成实验室桌距 12–20 cm |

P3 未过（世界表缺 r/contact/邻域）= 不得把 OPEN→压入→CLOSE 当主路。

**现在：** 规则圆/watershed **停用**。语义簇仍出品红门；单果要 **贴果皮的实例 mask**。待标队列：`datasets/scene_seg/to_label/`（先腕部）。**P3 抬升已接：** mask∩深度 → `/perception/scene_objects` 分层多边形，RViz `/planning/viz/scene_slices`（非球）。GraspRequest 仍未接。

旧 ID 对照：A=P1，A2=P2，B=P3，C=C1，D=S，E≈M1–M3，F⊂P3/P4。

### 0.5 两套状态机（不要混）

| | **真机现在**（M3 前 = teacher） | **目标** |
|--|---------------------------|----------|
| 选目标 | agent 在固定图上点区域 | `SELECT_FRUIT`：全局单果队列 |
| 粗到位 | `ALIGNING` 入口关节 / `refine_entry_pose` | `APPROACH_STANDOFF`：对该果 lookat |
| 精触达 | `REFINING` 腕部 YOLO + PBVS（**仅 teacher**） | `APPROACH_FRUIT`：腕部覆盖 xyz 后 TrajHead 顶到接触 |
| 感知 | YOLO 簇/果 + 占用栅格可调试 | DINOv3 SceneObject；占用仅审计 |

### 0.6 废止（文档后文若再出现，以本节为准）

- 主路径：2 cm occupancy / ESDF / occupancy polygon 当枝条
- 新主路径：berry YOLO 检测器、HSV 假果
- 语义连通域整簇一次 `fit_sphere` 当作一颗果
- 为枝/电钻造 CAD 跑 FoundationPose
- 图像 ego 当 ObstacleToken
- 全局质心当接触点
- 用 yaml 常数（`pre_grasp_width_m` / `grasp_width_m` / 固定 `press_in_m`）当开合/压入指令
- 用 2D `bbox_uv` 宽度当夹爪开口
- 未离线 QA 改 REFINING / 接触 IK

标注格式里的 **YOLO-seg txt** 只是「一行一个多边形」的存储，**不是**运行时 YOLO 检测器。

展开（schema、缺口表、可视化）：**§12**。Planner I/O 与执行层：**§2–§8**（其中 occupancy topic 为遗留 bag）。

### 0.7 落地前评审（小样本 BC → RL → 触达）

**已拍板（2026-09-02）：**

1. **接触职责：** TrajHead 从远处一路输出到接触点。PBVS / 现有 REFINING **只当 teacher 录 bag**，不是部署主路。
2. **叶子：** 不新标一类。语义挖掉果/枝/硬物/ego 之后，剩余有效深度 = **residual 软障碍**（仿真轻碰扣小分、重碰失败；真机轻擦叶不算失败）。

**总评：方向仍对，但「一路到接触 + 小样本 RL」比切 standoff 更紧。** 可行前提：腕部修正进入每一拍观测、关联失败禁止接触、RL 世界与部署同为 SceneObject+residual，而不是冻结的全局质心 + occupancy。

#### 对 RL 目标：环境信息够不够？

| 进 token 的 | 够不够 | 说明 |
|-------------|--------|------|
| ACTIVE 果 xyz + 法向 + `pick_role` | 接触级不够，除非已被腕部改写 | 必须带 `source_cam`、`visible_wrist`、`xyz_std` |
| 枝胶囊 / 硬物 OBB | 硬避碰够 | polygon/切片给 RViz 与精细碰撞；token 定长即可 |
| residual 软障碍 | **要补** | 冠层/叶不进四类；用挖除后深度，勿把整株 2 cm occupancy 当枝条 |
| ego FK | 够 | |
| 遮挡 | 缺 | `occluded`：枝后近果不要当第一目标 |
| 1 Hz 无图 | 远处够，近距紧 | `visible_wrist` 后应变快 replan（建议 5–10 Hz）或缩短 execute 窗口；否则 1 s 开环会顶穿 |

SceneObject 给人和碰撞看完整几何；**RL 只优化进 `planner_input` 的投影**。腕部修正后的 xyz 必须写进 `berries[]`。

#### 腕部能不能把目标位修准？（一路到接触时的硬条件）

**能修到接触可用，关联错误比深度噪声更致命。**

- **时序：** standoff 前用固定相机厘米级目标学快路径；`visible_wrist` 后 **覆盖** ACTIVE 的 xyz/法向（不与全局平均）。之后 TrajHead 的 goal 只能是这条腕部位。
- **关联：** 腕部实例接回 `fruit_id` 用 **该果自身点云半径重叠**（与 P3 `detections_overlap` 同一套），失败 → `HOLD` / 重新 standoff，**禁止接触**。禁止再写 4–6 cm 物种门。
- **外参：** 同一颗果固定 vs 腕部 xyz 先做对齐验收；TF 偏了会越修越偏。
- **RL 不得静态 t0：** 远距=带噪声的全局果位；tip 进入该果尺度若干倍后换成腕部观测。碰撞 = SceneObject + residual，主碰撞不用 occupancy。
- Teacher 是 PBVS 成功轨迹；RL 奖励的 goal 必须是 **腕部表面点**（沿法向），不是全局球心。

腕部近距、单果 mask 中位数可以到毫米～亚厘米；固定相机不能当接触点。

#### 小样本

- 10 张分割过拟合 ≠ 策略预训练。BC 样本是成功触达 bag。一路到接触时，**失败关联的 episode 不得进 BC**。
- 感知不稳则 token 是错世界。分割要在多株/光照上先稳，再大规模 RL。
- IQL/CQL+BC 正则补不上关联门和 TF 误差。

#### 方案应补的约束（仍先不动代码）

1. Token：`source_cam`、`visible_wrist`、`xyz_std`、`occluded`；residual 软障碍进 sim。
2. `visible_wrist` 后提高 replan 或缩短窗口。
3. 关联失败 HOLD，不碰。
4. P1：重叠果优先 **3D 聚类**（果径），2D watershed 辅助。
5. `planner_sim_env` 主碰撞 = SceneObject + residual。
6. P4 / T2：双目同一果 xyz 偏差必须有记录；接触段 live/coast 分列。

**结论：** 从 **P1** 和 **D1/T** 两头开工。100 条 bag 有价值，但必须过接触 tracking/精度门再进 BC。

---

## 1. 系统位置

```
感知 (DINOv3 语义 + 实例)
  → 结构化 SceneObject（多边形 / xyz / 朝向 / 属性）
  → 双目抬升 / 腕部优先融合
  → SceneGraph（berries[] 单颗）+ obstacles（枝/硬物几何）
  → fruit_queue（规则选 fruit_id）
  → pick_cycle_fsm / reach_fsm（fsm_state）
  → trajectory_planner_node（Transformer，1 Hz replan）
  → trajectory_executor_node（IK → 关节样条 → 臂控）
```

目标采摘循环与建模：**§0（冻结基线）**，展开 **§12**。

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
    # P3 目标（msg 待加）：r_m, contact_xyz, occluded, xyz_std, source_cam
    # diag: depth_mode, z_depth_m, z_mono_m, track_source
  obstacles[]:         # branch 胶囊、rigid OBB；果不进此槽
  occupancy_local:     # 可选遗留 ESDF crop
  occupancy_layers:    # DEPRECATED 2cm voxel BEV

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
  # P3 目标 GraspRequest（规划量，随 ACTIVE 3D 变；msg 待加）
  # grasp.contact_xyz, approach_axis, jaw_axis
  # grasp.jaw_open_m, jaw_close_m, press_in_m, standoff_xyz

end_effector:
  tool_type_id: int    # profile 枚举
  profile_name: string # 如 agx_gripper_v1
  tip_offset_link6: [3]
  # 极限不是指令：jaw_open_max/min, jaw_close_min, finger_thickness_m
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
  # 以下为机械极限 / 旧默认，不是每颗果的开合指令（指令=GraspRequest）
  pre_grasp_width_m: 0.055 # → jaw_open_max 量级
  grasp_width_m: 0.012     # → jaw_close 下限量级
  grasp_force_n: 1.5
  press_in_m: 0.008        # 旧默认；P3 改为 ≈r
  retract_m: 0.050
```

启动参数 `--end-effector agx_gripper_v1`；换末端只改 yaml + 标定，**不改模型结构**（`tool_type_id` embedding 区分品类）。

### 5.0 模型学什么 vs 规则做什么

| 层 | 职责 |
|----|------|
| **模型** | 合拢（或吸盘）状态下 **tip → 接触点** 的接近轨迹（`tool_trajectory_4s`） |
| **规则** | 多夹具差异：夹爪 OPEN / 压入 / CLOSE / RETRACT；吸盘开吸 / 断吸等 |

当前 P0 验收：**先跑通 close 态指尖碰到接触点**；open→压入→close **暂不接**（等 **§0.4.1** 世界表能给出该果的 r/contact/邻域，并算出 GraspRequest）。

冒烟：`scripts/tip_touch_smoke.py`（需 `trajectory_executor --drive-arm`）。

yaml 里现有 `pre_grasp_width_m` / `grasp_width_m` / `press_in_m` 视为 **极限或旧默认**，不是每颗果的指令。开口/闭合/压入按 §0.4.1 从 3D 算，再 clamp 到 profile。

完整夹爪规则序列（后接，非模型；宽度来自 GraspRequest）：

| 步骤 | 吸盘 `suction` | 夹爪 `gripper`（规则） |
|------|----------------|------------------------|
| 接近 | tip → 果表面（模型） | tip → 果表面（模型，**合拢**） |
| 接触后 | 开吸 | OPEN → 压入 `press_in_m` → CLOSE |
| 回撤 | +n 退 `retract_m`，断吸 | 夹紧退 `retract_m`，落果位 OPEN |

手动开合：`python3 scripts/gripper_cmd.py --open|--close`。

### 5.1 自由空间：语义实例障碍（果 / 枝 / 硬物）

**不再**用半球 2 cm voxel occupancy 作为主路径。RGB 能看见的细枝在深度栅格里会被膨胀成一团，polygon 只是那团的外皮。

主路径：

1. DINOv3 输出像素语义 `berry / branch / rigid / ego`，并解码成 **按实例切开的 SceneObject**（见 §12）。`ego` 不进障碍。
2. 每个实例先落成**结构化几何**（2D 多边形 + RGB-D 点云 / 凸包 / z 切片），再派生规划器用的紧凑量（质心、朝向、尺度）。**禁止**把整簇浆果拟合成一个球——那是当前 `fit_sphere` 的有损压缩，不是目标表示。
3. 紧凑派生（给 token / 碰撞，**不是**场景的唯一样式）：
   - berry：**每颗**质心 + 半径/椭球 + 可见面法向
   - branch：骨架折线 + 半径，或胶囊链
   - rigid：凸包顶点或 OBB；可选 z 切片多边形
   - ego：分割只作监督；规划器 ego 仍用 URDF FK
4. 发布结构化对象 + `/perception/obstacles`；RViz 先画 polygon/hull，再叠紧凑框。
5. 规划器输入 = 这些对象的紧凑量；自由空间 = 障碍几何补集。选果吃 **单果 SceneObject**。

采摘推进、对象 schema、缺口与 checkpoint：**§12**。

| 来源 | kind | RViz |
|------|------|------|
| 果实例 | KIND_BERRY=0 | 品红 contour + 单果椭球 |
| 枝条实例 | KIND_BRANCH=1 | 绿骨架 / 切片多边形 |
| 硬物实例 | KIND_RIGID=2 | 红 hull / OBB |
| 吸杯/夹爪 mask | 不进 obstacles | — |
| URDF FK | ego（非 obstacle） | 自滤 |

```text
固定/腕部 RGB → scene_seg_node
  → semantic + instance_id + polygon_uv
深度 + TF → 结构化抬升
  → SceneObject[]（xyz / 朝向 / hull 或 z_slices / 属性）
  → 派生紧凑量 → /perception/obstacles + MarkerArray
assembler → planner_input（berries[] + obstacles[]）
```

**话题：**

| Topic | 内容 |
|-------|------|
| `/perception/scene_seg/semantic` | uint8：0 bg / 1 berry / 2 branch / 3 rigid / 4 ego |
| `/perception/scene_seg/instances` | mono16 实例 id（须按颗，禁止粘连整簇） |
| `/perception/scene_objects` | **目标** SceneObject[]：polygon + xyz + 朝向 + 属性（§12.1） |
| `/perception/obstacles` | 紧凑派生：`SceneObstacle[]` 位姿+尺度 |
| `/planning/viz/semantic_obstacles` | RViz：contour / hull / 切片，再叠胶囊/OBB |
| `/planning/planner_input.obstacles` | 模型障碍 token |

**3D 查看：** `bash scripts/view_semantic_obstacles.sh`。  
采集标注：`datasets/scene_seg/README.md`；训练：`python3 scripts/train_dinov3_seg.py`。  
节点：[`scene_seg_node.py`](../src/picking_perception/picking_perception/scene_seg_node.py)、[`semantic_obstacle_node.py`](../scripts/semantic_obstacle_node.py)。  
仿真：[`planner_sim_env.is_free`](../scripts/planner_sim_env.py) 对 rigid/branch 基元拒碰；berry 挖洞。旧 occupancy_local 仅在无 obstacles 时回退。

旧 voxel 节点 `occupancy_map_node` 可手动开调试，**bringup 默认不启动**。

---

## 6. FSM 与阶段切换（规则，非模型）

**目标推进**（语义建模切换后）见 **§0.5 / §12.5**。  
下面 6.1–6.3 是 planner 输入枚举与现状映射；真机 `reach_fsm` 在 **M3 前**仍走 [`REACH_PIPELINE.md`](REACH_PIPELINE.md) 的 LOCKING→ALIGNING→REFINING 作为 teacher。

### 6.1 `fsm_state` 枚举（planner 输入）

| 值 | 含义 | 典型 scene |
|----|------|------------|
| `SELECT_FRUIT` | 全局单果队列，写 `fruit_id` | 固定相机已有 ≥1 颗 3D 果 |
| `APPROACH_STANDOFF` | IK 到 ACTIVE 附近，让腕部看见该果 | 果位粗、腕部可能还看不见 |
| `APPROACH_FRUIT` | 腕部精锁定 + 触达 | ACTIVE + `visible_wrist` |
| `RETRACT` | 回撤 | fruit_id 仍有效，standoff 增大 |
| `HOLD` | 保持 / 等待人工 | 可选 |
| `IDLE` | 不 replan | planner 可停 |
| `CLUSTER_ALIGN` | **遗留**粗对齐簇 | 仅兼容旧 bag / 未切 **S** 前 |

### 6.2 进入采果循环（果与簇关联）

**新规则（§12.3）**：不要求腕部先看见果。固定相机给出单果 `base_link` 位姿即可 `SELECT_FRUIT` → `APPROACH_STANDOFF`。腕部可见是 **standoff 完成门控**，不是选果前提。

遗留规则（未切 **C1** 前仍可用）：

1. `cluster_id` 已锁定；
2. 果的 `base_link` 落在簇中心 `radius_m` 内（默认 0.25 m）；
3. `visible_wrist`（旧路径把「腕部看见」当成进入 TOUCH 的条件）。

### 6.3 与现有节点映射

| 现有 | 迁移后 |
|------|--------|
| `reach_fsm` LOCKING（agent 区域） | `SELECT_FRUIT`：`fruit_queue` 近到远 + 可达，**不**再等人点区域 |
| `reach_fsm` ALIGNING / `refine_entry_pose` | `APPROACH_STANDOFF`：对 ACTIVE 解算到 standoff，看向该果 |
| `reach_fsm` REFINING / PBVS | `APPROACH_FRUIT`：腕部优先更新局部 scene 后触达 |
| `pick_cycle_fsm` TOUCH | 同上 `APPROACH_FRUIT` |
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
ObstacleToken  # 必选：posed berry/branch/rigid（kind+pose+scale）；见 scripts/obstacle_tokens.py
EgoToken     × 1     q + tip + links + workspace_aabb
TaskToken    × 1     embed(fsm_state, cluster_id, fruit_id)
ToolToken    × 1     embed(tool_type_id) + MLP(calib vector)

→ Transformer → 16 个 causal query slots → TrajHead
```

**无 `obstacles[]` 不得训、不得推**（`input_flags` 含 `no_obstacles` 则跳过该帧）。旧 occupancy polygon 与 Obstacle 球不再作为主编码。

### 7.3 训练标签（BC 冷启动）

- 来源：成功 episode 的 bag：**`planner_input`（含 `obstacles[]`）** @ t0 + PBVS/executor 的 `tool_trajectory_4s` 或 `joint_states`；
- 另录全图 `/perception/occupancy_esdf` 供 sim/审计；
- 对齐 `t0`，重采样 16 点 @ 0.25 s；
- FK → tool tip + tool +Z → 6D 绝对标签；
- 数据增强：同一 episode 随机多个 `t0`（模拟 replan）；
- **不要求人手示教**；PBVS + executor 即自动 teacher；
- **过滤**：`obstacles` 空 / `no_obstacles` 帧不进 BC。旧 occupancy bag 不能直接训。

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
s_t = planner_input(t0)          # clusters, berries, obstacles, ego — 真值
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
| `/planning/planner_input` | `PlannerInput` | **模型完整输入**（含 `obstacles[]`） |
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
| `/perception/occupancy_esdf` | `OccupancyEsdf` | 全图占据+ESDF（sim/审计；须进 bag） |
| `/perception/occupancy_polygons` | `OccupancyPolygons` | **模型用 2cm 层 polygon**（亦嵌在 `planner_input`） |
| `/perception/occupancy_local` | `OccupancyLocal` | 32³ ESDF crop（sim） |
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
- 每 replan 周期存：`planner_input`（**必含 `obstacles[]`**）+ `tool_trajectory_4s` / `joint_states` + 派生 6D 标签；
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
| [`scripts/semantic_primitives.py`](../scripts/semantic_primitives.py) | mask→球/胶囊/OBB |
| [`scripts/semantic_obstacle_node.py`](../scripts/semantic_obstacle_node.py) | `/perception/obstacles` + RViz |
| [`scripts/obstacle_tokens.py`](../scripts/obstacle_tokens.py) | ObstacleToken 编码 |
| [`src/picking_perception/.../scene_seg_node.py`](../src/picking_perception/picking_perception/scene_seg_node.py) | DINOv3 语义/实例图 |
| [`scripts/occupancy_map.py`](../scripts/occupancy_map.py) | 遗留 OccupancyVolume（调试） |
| [`scripts/occupancy_map_node.py`](../scripts/occupancy_map_node.py) | 遗留 voxel；bringup 默认不开 |
| [`scripts/obstacle_extractor_node.py`](../scripts/obstacle_extractor_node.py) | **DEPRECATED** 球体 |
| [`scripts/planner_sim_env.py`](../scripts/planner_sim_env.py) | 离线静态 sim（occupancy ESDF） |
| [`scripts/planner_rl_smoke.py`](../scripts/planner_rl_smoke.py) | RL reward loop 冒烟 |
| [`scripts/fruit_queue.py`](../scripts/fruit_queue.py) | `fruit_id` 规则（须吃单果 SceneObject，见 §12） |
| [`scripts/infer_scene_seg.py`](../scripts/infer_scene_seg.py) | DINOv3 离线推理预览 |
| [`scripts/pick_cycle_fsm_node.py`](../scripts/pick_cycle_fsm_node.py) | 外层编排，`fsm_state` 来源 |
| [`scripts/reach_fsm_node.py`](../scripts/reach_fsm_node.py) | 内层触达，逐步改为 executor 驱动 |
| [`docs/PICK_CYCLE.md`](PICK_CYCLE.md) | 采摘循环总览 |

---

## 12. DINOv3 上线：结构化场景对象（§0 的展开）

基线与选型以 **§0** 为准。本节补 schema、当前代码缺口与可视化细节。

### 12.1 产品定义：每个可见物体一条结构化记录

对固定相机、腕部各跑一遍分割，融合后得到 `base_link` 下的对象表。字段如何获得、怎么汇总成 `GraspRequest`：**§0.4.1**。这里只列记录形状。

```text
WorldScene
  berries[]   Branch[]   Rigid[]   residual   ego(FK)
  GraspRequest   # 规划量，只针对 fruit_id=ACTIVE

SceneObject  （berry / branch / rigid 共用头，geometry 按类）
  id, class ∈ {berry, branch, rigid}     # ego 不进此表
  score, source_cam ∈ {fixed, wrist}
  polygon_uv[]  bbox_uv                  # bbox 仅 HUD
  centroid_uv                            # berry：果心峰
  pts_xyz[]     n_pts, xyz_std, depth_valid
  geometry:
    berry  → center_xyz, r, axes?, contact_xyz, approach_normal
             occluded, neighbor_berry_ids, gap_m, pedicel_dir
    branch → skeleton[] + radius → 胶囊链；可选 z_slices
    rigid  → OBB 或 convex_hull；可选 z_slices
  attributes  pick_role, area_px, ...

GraspRequest  （不是物体；assembler 从 ACTIVE+邻域+profile 极限算）
  contact_xyz, approach_axis, jaw_axis
  jaw_open_m, jaw_close_m, press_in_m, standoff_xyz
```

**z 切片（切片逻辑）**：把实例点云按 `z`（或沿枝骨架弧长）分箱，每层对 XY 做凸包/轮廓，得到一层一层的 polygon。这是仓储/农业表型里常用的 2.5D 表示，适合细长枝和盆沿，比单层 occupancy 外皮干净。切片是**对象的几何字段**，不是再开一套 voxel 主路径。

规划器 token 是这张表的**投影**（位置、朝向、尺度、class），不是场景的唯一样式。RViz 必须能画出 polygon / hull / 切片，而不是只画一个球。

### 12.2 「抬升后变成大球」是什么（当前代码，不是目标）

DINOv3 **并不输出球**。现在的链路是：

```text
语义图 (berry==1 的所有像素)
  → instances_from_semantic：同类 8-连通域给一个 id
  → 该 id 的深度点全部反投影
  → fit_sphere：质心 + 点到质心中位半径（再 clamp 6–40 mm）
```

贴在一起的浆果在语义图里是同一类、而且彼此连通 → **一个 id** → **一次** `fit_sphere` → 看起来像「一簇一个大球」。球是我们对点云做的**有损压缩**，用来塞进现有 `SceneObstacle`（`position + scale`）。枝被压成胶囊、硬物被压成 OBB，同理。

目标表示是 §12.1 的 SceneObject。紧凑球/胶囊/OBB 只作为 token 派生，且必须在**已经切开的实例**上做：一颗浆果用小球合理；一簇用一个球不合理。

### 12.3 单果分割：走 DINOv3 实例，不引入 YOLO

标注规范已经是「每颗一个多边形」（`datasets/scene_seg/README.md`）。YOLO-seg **txt 里本来就是逐实例多边形**。训练时 `yolo_seg_to_mask` / `*_mask.png` 把所有果填成像素值 `1`，实例 id 被丢掉；头是 5 类语义 argmax。这是**监督被压成语义**，不是 DINOv3 不能分颗。

同类相邻像素，语义头在数学上就分不开「第几颗」。要单果，必须让模型或后处理输出 **instance id**。目标方案（按优先）：

| 优先级 | 做法 | 要什么 |
|--------|------|--------|
| **主路径** | 实例监督 + 解码 | GT 保留每颗 polygon / instance map；头输出 instance（Mask 式 decoder，或 DINO patch embedding 聚类，或果心热力图+半径）。**运行时无 YOLO** |
| **过渡** | 只从 berry 语义 mask 做几何切分 | 已知果径 8–18 mm + 深度 → 像面期望半径；距离变换 + watershed / 腐蚀断开。仍无检测器 |
| 不做 | 再挂 berry YOLO 框当种子 | 与 DINOv3 上线目标重复；遗留 `reach_fsm` 触达可暂留，**不进**新规划器主路径 |

枝 / 硬物：连通域通常够用（一根枝、一块盆）。果必须按实例，因为采摘单位是颗。

### 12.4 四类如何进规划器；建模还缺什么

四类语义**够当 class 通道**。喂模型的是 SceneObject 投影：

```
BerryToken     ← center / r / contact / normal / occluded / pick_role
GraspRequest   ← ACTIVE 的 approach / jaw_axis / jaw_open / jaw_close / press_in
ObstacleToken  ← branch 胶囊链 + rigid hull/OBB/切片
EgoToken       ← q + tip + link 胶囊（FK；指缝=当前开口）
TaskToken      ← fsm_state + fruit_id
```

| 信息 | 现在 | 缺什么 |
|------|------|--------|
| 四类语义图 | 10 张过拟合预览；节点默认只订 **固定** 相机 | 腕部 `scene_seg_node`；真人标注规模；val |
| 实例 id | 语义连通域（果会粘连） | 实例 GT + 解码，或 watershed 过渡 |
| 2D polygon | 标注侧有；推理未发布 | 每个 instance 的 contour → topic / 字段 |
| 3D xyz / 朝向 | `fit_sphere/capsule/OBB` 有损（簇球） | **P3 §0.4.1**：切开后单果 center/r/contact/normal + 邻域；枝骨架；硬物 hull |
| 抓取查询 | 无；yaml 常数开口 | `GraspRequest` 从 ACTIVE 3D 算接近轴与开合 |
| 枝 / 硬物 3D | 代码能抬升 | 未用真人 GT 权重在线；腕部近距未接 |
| ego | assembler FK ✅ | 运行时用 TF 挖吸杯深度；图像 ego 不进 obstacles |
| 双目交叉覆盖 | 旧：腕部优先只覆盖 ACTIVE 邻域 8–12 cm | **§0.4.1**：两路全量；重合并；不重合都留；fruit_id 不是建图过滤器 |
| `fruit_queue` | 代码有，origin=`base_link` | 输入必须是单果 SceneObject；加可达 / 别穿 rigid |
| `APPROACH_STANDOFF` | **无**（ALIGN 仍是入口关节） | 对 ACTIVE 质心 lookat |
| 精锁定 + 触达 | REFINING / PBVS ✅ | 门控改看腕部 berry 实例 UV，不看 YOLO |
| occupancy ESDF | 退出主路径 | bag/sim 可留审计 |

叶子保持背景，不进对象表。

### 12.5 目标推进策略（状态机要改）

全局 SceneObject 已能给出果实方位和（切开后的）哪一颗；腕部因姿态可能暂时看不到，**不阻塞**粗建模和选果。交叉覆盖以腕部为准。补约束：

1. 选果 + 粗接近只用固定相机单果对象；不要求 `visible_wrist`。
2. standoff 把腕部摆到能看见 ACTIVE（沿 `−approach` 退到该果尺度若干倍，杯轴 lookat）。任务站位，不是建图门。
3. 静态世界表按 §0.4.1 双目全量融合。运动中只改写 **锁定果** xyz（P4），不要把 occupancy/邻域门抄回来。
4. 排序：近到远 + workspace + 深度有效 + 不在 rigid 后 + 未 DONE。
5. **全局质心不得当接触点**；接触只信腕部实例的深度/法向。

```text
IDLE
  → SELECT_FRUIT        固定相机 berry SceneObject 队列 → fruit_id
  → APPROACH_STANDOFF   IK/planner 到 ACTIVE 前 standoff（全局质心只作粗目标）
  → 门控：腕部 berry 实例与 ACTIVE 按自身半径重叠（P4 关联，不是 6 cm 门）
  → 运动中修正：只改写锁定果 xyz/法向（P4）；世界表其余物体仍按 P3 全量融合
  → APPROACH_FRUIT      精锁定 + 触达
  → WAIT_CONFIRM / RETRACT → 下一颗
```

未到 **M3** 前，真机触达仍走 `REACH_PIPELINE.md` 作 teacher（可暂留腕部检测作对照）。接触段必须过 **T1/T2**。稳定后选果是规则，不是 agent 点区域。

### 12.6 可视化（建模验收）

| 层 | 要求 |
|----|------|
| 语义叠图 | 品红果 / 绿枝 / 红硬 / 青 ego；**每颗果轮廓单独描边** |
| 2D polygon | 每实例一条 contour，标 `id` |
| 3D | 果：单果球（center 在皮后）+ 接触点 + 法向箭头；枝：骨架/胶囊；硬物：hull/OBB；可叠 GraspRequest 的 approach/jaw 轴 |
| 臂 | FK 胶囊，与青 mask 大致重合 |
| HUD | `fsm_state` `fruit_id` `visible_wrist` |

一团品红 + 一个大球 / 一条粗胶囊包整簇 = **P1** 未过。接触段 id 乱跳或最后 15 cm 靠 coast = **T1/T2** 未过。

### 12.7 Checkpoint 摘要

以 **§0.4** 为准（P / T / D / C / S / M）。此处不重复旧 A–F。

| 现在卡在 | 可并行 |
|----------|--------|
| **P2** 实例 GT（热力图切开簇） | **D1** 确认 PBVS 连跑；立刻做 **T1/T2** 接触段统计（可先用现有腕部检测） |

**明确不做：** 为枝/电钻伪造 CAD；occupancy 冒充枝条；粘连 `fit_sphere` 当 `fruit_id`；主路径再加 YOLO 检测器；**未过 T 的 bag 进 BC**；未 QA 改接触 IK。

---

## 变更记录

| 日期 | 内容 |
|------|------|
| 2026-09-02 | **禁止场景尺度魔法数**：删刚体 35 cm 高度帽、6/12 cm 融合门；飞点改像面；重合用实例半径/AABB；§12 旧「腕部邻域覆盖」作废 |
| 2026-09-02 | 停用规则圆；待标簇图 `to_label/`；单果改为贴皮实例 mask 再训 |
| 2026-09-02 | §0.4 checkpoint 重排：P 建模 ∥ D1/BAG；**T1/T2 接触 tracking 与感知精度独立门**；≥100 bag 仅过 T 进 BC |
| 2026-09-02 | **§0.7 落地前评审**：环境投影进 token、腕部修正条件、静态 RL 缺口；职责切到 standoff |
| 2026-09-02 | §0.7 拍板：TrajHead 一路到接触；叶子=residual 软障碍；腕部覆盖+关联门+非静态 RL |
| 2026-09-02 | **§0 现行方案冻结**：架构 / 选型原因 / 落地 CP-A…F；废止 occupancy 主路径与主路径 YOLO |
| 2026-09-02 | §12 纠正：DINOv3 产出 SceneObject（polygon/xyz/朝向/切片），单果走实例解码，主路径不引入 YOLO |
| 2026-09-02 | §12 单果切开 + 四类 token 缺口 + 目标 FSM（选果→standoff→腕部优先触达）与 CP-A…F |
| 2026-09-02 | §5.1 改语义实例障碍（DINOv3 果/枝/硬物 + 3D 基元）；voxel occupancy 退出主路径 |
| 2026-09-02 | 植株=含果的占据连通块（非半径/非YOLO掩膜）；无果独立物体=HARD |
| 2026-09-02 | 只认果(YOLO)/植株(簇球)/EGO(FK)；其余深度=无类别体积；去掉桌/盆语义 |
| 2026-09-02 | EGO 自滤补腕相机/夹爪壳体（避免固定 depth 把 Gemini 标成 OCC） |
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
