# Refactor Spec: 3D Scene Perception (Semantic Feature-Driven)

Branch: TBD (建议从 `pbvs-vlm-reach-v2` 派生)
Status: Draft — 待实施
Author context: 由主机 A 上的 Claude 依据 pbvs-vlm-reach-v2 上一次可见状态撰写；实际实施发生在主机 B。

---

## 0. 阅读须知（给实施方 AI）

本文档是**方向性设计规范**，不是逐字节实现。动手前你必须做的事：

1. **先核对代码现状**。文档中提到的文件名 / 类名 / 消息字段名基于一次静态扫描，**你手上的代码更新**，路径可能已重命名或迁移。用 `grep`/`find` 校对，不要照文档写死路径盲改。
2. **每个 Phase 是独立 commit 单元**。Phase 内部尽量按小步提交，方便回滚。
3. **消息结构改动需要显式列出影响面**（发布者 / 订阅者 / launch / yaml），确认后再动。
4. **遇到与文档冲突的现状**（例如某文件已被删、某字段已改名），以现状为准，并把冲突记录在 Phase commit message 里，供上游同步。
5. **不要引入本文档 Non-Goals 列出的方向**（见第 8 节），避免 scope 扩散。

---

## 1. 背景 & 目标

### 场景
- 实验室蓝莓采摘机器人
- 双相机：DaBai (fixed, 全局粗定位) + Gemini 305 (wrist, 精细感知)，两者均为 RGB-D
- 下游消费者：
  1. 场景可视化 / 离线分析
  2. RL 轨迹规划的环境建模

### 输出契约（不变）
`List[SlicedObject]`，每个对象含：
```
{
  id: int (跨帧稳定),
  class_id: int (0=berry, 1=branch, 2=rigid obstacle, ...),
  attributes: {centroid_xyz, volume, visibility, ...},
  slices: [{z_min, z_max, polygon_xy}, ...]
}
```

### 当前 pipeline
```
DINOv3 语义分割
  → 2D 实例分割 (watershed, berry_instances.py)
  → 深度反投影
  → Z-slice + polygon 提取 (z_slice_geometry.py)
  → BoT-SORT 2D 跟踪 + 3D Kalman Filter (berry_kf_tracker.py)
```

### 现有痛点（需要解决的核心问题）

1. **深度融合冗余**：`depth_fusion.py` 里 RGB-D / Depth-Anything-V2 / mono 三路 fallback + `depth_mode` 字段。实际项目已明确**信任 RGB-D 相机深度**，其他两路是历史包袱。
2. **2D 实例分割瓶颈**：`berry_instances.py` 的 watershed 是固定阈值 2D 逻辑，遮挡 / 连片场景不稳，田间数据丰富时无法泛化。
3. **几何后处理天花板低**：`z_slice_geometry.py` 依赖固定阈值参数，参数漂移敏感。
4. **Tracking 跨维度**：BoT-SORT 在 2D 图像空间跟踪 + 3D KF 更新状态，坐标切换多，遮挡后 track_id 频繁跳变。

### 约束（硬约束，不可绕开）

| 约束 | 含义 |
|------|------|
| 只能标注 2D RGB 语义分割 | 无 3D 标注、无 polygon 标注、无 3D bbox 标注 |
| 相信 RGB-D 深度 | 不再引入 mono / 单目深度模型作为兜底 |
| 标注规模：当前 <200，中期 >5000 | 决定 backbone 从"全冻结"到"部分解冻"的时间线 |
| 保持 `SlicedObject` 消息结构 | 下游 RViz / RL / VLM 不受影响 |

---

## 2. 核心设计决策（含理由）

### 决策 1：不做端到端 "RGB-D → 结构化 polygon" 融合模型

**理由**：没有 3D 标注 → 模型学不到"如何用 depth"的监督信号，只会学到"看 depth 猜 mask"。深度是几何真值，应走反投影，不该走网络。

### 决策 2：只在两点上引入学习——2D 语义分割 + 3D 实例聚类

- **2D 语义分割**：你能标注，DINOv3 head 用 supervised 学。
- **3D 实例聚类**：**无监督**，用 DINOv3 patch feature 相似度 + 空间连通性驱动，不需要额外标注。
- 其他环节（反投影 / TSDF 融合 / polygon 提取）保持确定性几何算法。

### 决策 3：实例分割从 2D 迁到 3D 空间

**理由**：现有 pipeline 最大瓶颈就是"2D watershed → 反投影"这条路径，2D 遮挡一发生，后面全崩。3D 空间做实例分割天然规避 2D 遮挡歧义。

**新流程**：
```
稠密点云 (xyz + label + DINOv3 feat)
  → 类内 3D 聚类 (Phase 1: DBSCAN; Phase 2: spectral / OpenMask3D 风格)
  → per-instance 点云
  → Z-slice + alpha shape → polygon
```

### 决策 4：Tracking 从 "2D BoT-SORT + 3D KF" → "3D obj-to-obj feature-aware Hungarian"

**理由**：在 3D 空间做 obj 级 ReID，DINOv3 patch feature 天然是好的 ReID 描述子，比 KF 的 constant-position 假设更 robust（采摘场景相机常在动，KF 假设失效）。

---

## 3. 分层架构

```
┌─────────────────────────────────────────────────────┐
│ [学习层]  DINOv3 backbone + 5 类分割 head           │
│           Phase 1: <200 张，全冻结 backbone         │
│           Phase 2: >1000 张，last 2–4 blocks 解冻   │
└─────────────────────────────────────────────────────┘
                     ↓ per-pixel semantic + per-patch feat
┌─────────────────────────────────────────────────────┐
│ [反投影层]  RGB-D → 稠密点云                        │
│             每点: (xyz, rgb, label, feat_pca32)     │
└─────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ [融合层]  多帧 TSDF                                 │
│           Phase 1: Open3D VoxelBlockGrid            │
│           Phase 2: 可选 NVBlox                      │
└─────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ [3D 实例分割层]  ← 关键升级点                        │
│   按 class 分组 → 类内聚类                          │
│   Phase 1: DBSCAN, metric = α·euclid + β·cos(feat) │
│   Phase 2: spectral / graph-cut on feature affinity │
│   Phase 3: (可选) Mask3D / OneFormer3D end-to-end   │
└─────────────────────────────────────────────────────┘
                     ↓ per-instance 点云
┌─────────────────────────────────────────────────────┐
│ [Polygon 提取层]  Z-slice + alpha shape (Shapely)   │
│                   输出 SlicedObject                 │
└─────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ [Tracking 层]  3D obj ↔ 3D obj Hungarian            │
│                cost = α·centroid_dist + β·feat_dist │
└─────────────────────────────────────────────────────┘
                     ↓
              RViz / RL env / VLM
```

---

## 4. Phased Implementation

### Phase 0 — 清理冗余（1–3 天）

**目的**：删除历史包袱，为后续重构降噪。不引入新架构。

改动清单（**动手前先 grep 核对实际路径**）：

- [ ] `depth_fusion.py`
  - 删除 Depth Anything V2 分支及其 lazy-load 逻辑
  - 删除 mono / size-based 分支
  - 只保留 RGB-D median + valid ratio 校验
- [ ] `picking_msgs/msg/DetectedBerry.msg`
  - 删除冗余字段：`depth_mode / z_mono_m / z_depth_m / center_depth_m` 中重复的
  - 保留单一 `z_m` 或类似（视现状取一）
- [ ] `fine_detector_node.py` / `global_detector_node.py`
  - 删除 `depth_pose_source` 参数的 `'mono'` / `'fused_da2'` 分支
  - 简化为直接读 RGB-D 深度
- [ ] `config/foundation_pose.yaml`（若存在）
  - 删除 mono 相关参数（`berry_diameter_m` 若仅用于 mono 估计则删）
- [ ] `berry_kf_tracker.py`
  - 打 TODO 注释标注 Phase 2 会被替换（此阶段不删，PBVS 还依赖）
- [ ] 更新 launch / yaml 里失效引用

**验证**：
- 现有集成测试跑通
- RViz 现有可视化输出无变化
- 单帧 `DetectedBerry` 输出（除删除字段外）与 Phase 0 前一致

### Phase 1 — 3D 空间实例聚类（1–2 周）

**目的**：首次引入新架构。输出接口保持不变，下游无感知升级。

**新增文件**：

- [ ] `scripts/scene_3d_fusion.py`（或 `picking_perception/` 下）
  - Input: RGB + Depth + DINOv3 semantic map + DINOv3 patch feature map
  - 反投影为稠密点云 `(xyz, rgb, label, feat_pca32)`
  - 多帧 TSDF 融合：Open3D `VoxelBlockGrid`（推荐 Phase 1 起点，CPU/GPU 均可）
  - 输出 fused 点云 + voxel grid 句柄

- [ ] `scripts/instance_3d_cluster.py`
  - Input: 融合后的语义点云
  - 按 `class_id` 分组，类内 DBSCAN
  - 距离度量：`d = α · euclidean(xyz) + β · (1 - cos_sim(feat))`
  - 参数 per class 独立配置（berry 用小 eps，branch 用大 eps 长条形态）
  - 输出 `List[InstancePointCloud]`

**改造文件**：

- [ ] `z_slice_geometry.py`
  - Input 从 "2D mask + depth" 换成 "3D instance point cloud"
  - Polygon 提取从 marching squares 换成 alpha shape（Shapely 2D per-slice）
  - `SlicedObject` 消息结构**保持不变**
- [ ] `berry_instances.py`
  - 弃用主路径；或降级为 2D 调试可视化独立工具
- [ ] `dinov3_seg.py`
  - 额外暴露 patch feature 输出接口（384D → 用离线 PCA 压到 32D）
  - PCA 矩阵从校准脚本一次生成，保存为 `.npy` 作为静态 asset
- [ ] DINOv3 分割 head 用现有 <200 张数据微调（backbone 全冻结，只训 head）

**参数入口**（放到 config yaml）：
```yaml
scene_3d_fusion:
  voxel_size_m: 0.005          # 5mm, 蓝莓尺度
  tsdf_trunc_m: 0.02
instance_3d_cluster:
  berry:
    dbscan_eps: 0.008
    dbscan_min_samples: 20
    feat_weight: 0.3           # β / (α+β)
  branch:
    dbscan_eps: 0.03
    dbscan_min_samples: 50
    feat_weight: 0.5
polygon_extraction:
  z_slice_thickness_m:
    berry: 0.004
    branch: 0.006
  alpha_shape_alpha:
    berry: 200.0
    branch: 80.0
```

**验证**：
- 遮挡 / 连片测试集上，实例召回率相较 Phase 0 前有可测量提升（目标: +15% 以上）
- polygon 帧间抖动定性测试通过（同一静态场景，连续 30 帧的 polygon 顶点均值方差）
- 端到端延迟不超过 200ms/帧（wrist 视角，Gemini 305 30Hz 场景）

### Phase 2 — 特征驱动升级 + 3D Tracking（1–2 月，数据到 1000+ 张时启动）

**改动**：

- [ ] DINOv3 backbone 部分层解冻（last 2–4 blocks），带正则微调
- [ ] 3D 实例分割升级：DBSCAN → spectral clustering 或 graph cut on feature affinity
  - 参考 OpenMask3D (NeurIPS 2023) 的 mask proposal + feature 蒸馏思路
- [ ] 新增 `scripts/obj_tracker_3d.py`
  - 输入：当前帧 obj list + 上一帧 obj list（每个 obj 带 centroid + mean patch feat）
  - Hungarian 匹配，cost = `α · centroid_dist + β · (1 - cos_sim(feat))`
  - 消息级：`SlicedObject.id` 跨帧稳定
- [ ] 弃用 `berry_kf_tracker.py`（PBVS 侧同步切到新 tracker）

### Phase 3 — 3D-Native 端到端分割（可选，>5000 张时评估）

- 引入 Mask3D 或 OneFormer3D 做端到端 3D 实例分割
- 3D 伪标注：用 Phase 2 输出人工修正即可，成本远低于从零标 3D

---

## 5. 什么该学、什么不该学（速查表）

| 环节 | 学 vs 不学 | 学习方式 | Phase | 理由 |
|------|-----------|---------|-------|------|
| 2D 语义分割 | 学 | supervised | 1 (冻) / 2 (微调) | 你能标 |
| RGB-D 反投影 | 不学 | — | — | 几何真值 |
| 多帧 TSDF 融合 | 不学 | — | — | 无 3D 标注也用不上学习 |
| 3D 实例分组 | 学 | unsupervised (feat + spatial) | 1 → 2 → 3 | 现有瓶颈；DINOv3 feat 天然可用 |
| Z-slice polygon 提取 | 不学 | — | — | 无 polygon 标注；alpha shape 够用 |
| Obj tracking | 半学 | unsupervised feat ReID | 2 | DINOv3 feat 做 ReID |

---

## 6. 学术 / 开源参考

在实施前建议至少读 F3RM 和 OpenMask3D 两篇的方法节，理解无监督 3D 特征聚类的技术路径。

- **F3RM** (MIT, CoRL 2023) — DINO 特征蒸馏进 NeRF 用于机械臂 open-vocab 抓取。**与本项目场景 1:1 相关**。
- **OpenMask3D** (NeurIPS 2023) — 无 3D 标注的 open-vocab 3D 实例分割。Phase 2 的直接参考。
- **SAM3D / SAI3D** — SAM 2D mask 投影到 3D 后基于特征聚类。
- **Feature-3DGS** / **LangSplat** (CVPR 2024) — 特征挂 3D Gaussian，未来 Phase 可选替代 TSDF 表示。
- **Mask3D** (ICCV 2023) — Phase 3 的 SOTA baseline。
- **ConceptFusion** (RSS 2023) — 多模态特征融进 3D map。
- **NVBlox** (NVIDIA) — GPU TSDF 工业实现，Phase 2+ 可选。

---

## 7. 开放决策（需要在 Phase 1 启动前拍板）

- **TSDF 后端**：Open3D `VoxelBlockGrid` (CPU-first，易集成) vs NVBlox (GPU，快但依赖 CUDA + 部署复杂)
  - 推荐：Phase 1 先 Open3D，Phase 2 视性能瓶颈再评估
- **DINOv3 patch feature 压缩**：PCA (简单，离线一次校准) vs 学习的 projection head (需自监督 loss)
  - 推荐：Phase 1 用 PCA，Phase 2 视 3D 聚类质量再决定
- **Alpha shape 库**：Open3D vs Shapely (2D per-slice) vs 自研
  - 推荐：Shapely，稳定且社区活跃
- **全局粗定位路径去留**：目前 `global_detector_node` (DaBai) 仍在用于 LOCKING 阶段
  - Phase 1 保持共存（不动 fixed 相机路径），Phase 2 再评估是否统一到新架构

---

## 8. Non-Goals（防止 scope 扩散，实施方请严格遵守）

**本次重构不做的事**：

- ❌ 不引入 Depth Anything V2 / 任何单目深度模型（已明确信 RGB-D）
- ❌ 不做 6DoF grasp pose 端到端估计（FoundationPose 保持独立模块）
- ❌ 不重写 PBVS 控制层（`reach_fsm_node` 仅在消息接口对齐处做最小改动）
- ❌ 不引入 VLM 融合到几何 pipeline（VLM 保持在决策层，与感知解耦）
- ❌ 不改 `SlicedObject` 消息结构（下游 RL / VLM / RViz 稳定）
- ❌ 不做多相机融合 (fixed + wrist 联合)（Phase 2+ 再评估）
- ❌ 不引入端到端 "RGB-D → structured output" 网络（无 3D 标注，不合理）
- ❌ 不做深度自监督预训练（数据规模不支持）

---

## 9. 交付物 Checklist（每个 Phase 收尾时勾选）

**Phase 0**
- [ ] 冗余字段 / 分支已删除
- [ ] 全部集成测试通过
- [ ] git log 里 Phase 0 commit 独立可回滚

**Phase 1**
- [ ] `scene_3d_fusion.py` / `instance_3d_cluster.py` 就位，含单元测试
- [ ] `z_slice_geometry.py` 改造完成，输入接口切换到 3D 点云
- [ ] DINOv3 head 在现有 <200 张数据上完成一次微调
- [ ] 遮挡 / 连片对照测试集通过（相对 Phase 0 前实例召回率提升 ≥15%）
- [ ] 端到端延迟 ≤ 200ms/帧
- [ ] `SlicedObject` 消息内容与旧 pipeline 语义一致（下游无感知）

**Phase 2**
- [ ] 3D tracker 上线，`berry_kf_tracker.py` 弃用
- [ ] DINOv3 部分解冻微调完成
- [ ] Spectral / graph-cut 聚类替代 DBSCAN
- [ ] track_id 稳定性回归测试通过（同一 obj 跨 100 帧内 id 变化率 ≤5%）

**Phase 3**（可选）
- [ ] 3D 端到端分割评估报告（相较 Phase 2 spectral clustering 提升是否显著）
