# 固定单目簇检测 + 统一 YOLO 模型

用 **最少张数** 采集固定相机图 → **Qwen2.5-VL 伪标簇** → 人工抽检 → 与腕部数据 **合并训练一个 YOLO**。

---

## 能不能共用一个模型？

**可以，而且推荐。**

两个模型不是必须。之前分开是因为：

- 全局用 YOLOE 零样本（没标固定相机数据）
- 腕部用闭集微调（330 张腕图）

端侧部署一个 `yolo11n`（约 6MB）完全够跑双相机；两个模型也就 12MB，**浪费不大，但维护成本高**（两套权重、两套训练、行为不一致）。

**统一方案：一个模型、两个类别**

| class_id | 名称 | 用途 | 数据来源 |
|----------|------|------|----------|
| 0 | `berry` | 腕部单果触达 | 现有 `datasets/blueberry/` |
| 1 | `cluster` | 全局簇规划 | 新建 `datasets/fixed_mono/` |

推理时：

- `global_detector_node`：只采纳 `class=cluster`，放宽圆度/面积过滤
- `fine_detector_node`：只采纳 `class=berry`，保持现有参数

同一 `.pt` 文件加载两次，**过滤类别不同**即可。VLM 只在 **离线标数据** 时用，不上车。

---

## 最划算路径（约 18 张 + 人工抽检）

### 0. 准备

```bash
# 固定相机
bash scripts/real_robot_bringup.sh --fixed-cam --camera-only

# 本地 VLM（二选一）
ollama pull qwen2.5vl:7b

# 或 API：export DASHSCOPE_API_KEY=...
```

### 1. 交互式最小采集

按提示挪动 **植株位置** 和 **全局相机位姿**，每格按 Enter 拍一张（可 `s` 跳过）：

```bash
bash scripts/capture_fixed_mono_dataset.sh
```

输出：`datasets/fixed_mono/images/fm_*.png`（约 18 格 + 可选 3 张补充）

### 2. VLM 自动标「簇」框

**本地 Ollama：**

```bash
bash scripts/run_fixed_mono_label_pipeline.sh prelabel
```

**API（DashScope 等 OpenAI 兼容）：**

```bash
export DASHSCOPE_API_KEY=your_key
bash scripts/run_fixed_mono_label_pipeline.sh prelabel-api
```

输出：

- `datasets/fixed_mono/labels/*.txt`（YOLO 格式，`class_id=1`）
- `datasets/fixed_mono/previews/*_vlm.jpg`（可视化，先快速扫一遍）

单张试跑：

```bash
python3 scripts/teacher_prelabel_vlm.py --limit 1 --overwrite
```

### 3. 人工审核（必须）

```bash
bash scripts/run_fixed_mono_label_pipeline.sh annotate
```

重点改：漏框、框到叶片/盆、簇边界过大。比从零手标快很多。

### 4. 合并腕部数据 → 统一训练集

```bash
bash scripts/run_fixed_mono_label_pipeline.sh merge
```

生成 `datasets/blueberry_unified/data.yaml`（berry + cluster）。

### 5. 训练 & 部署

```bash
# 在 train 脚本里指定 data.yaml（或复制 data.yaml 路径给 ultralytics）
yolo detect train data=datasets/blueberry_unified/data.yaml model=yolo11n.pt epochs=80

# 更新配置后重启感知
python3 scripts/update_foundation_pose_yolo.py \
  --model runs/detect/blueberry-unified-1/weights/best.pt --conf 0.28
```

`foundation_pose.yaml` 中 **global 与 fine 指向同一 `best.pt`**，分别设置：

- global：`yolo_open_vocab: false`，后处理只保留 cluster
- fine：只保留 berry（现有逻辑）

> 注：类别过滤参数将在联调时接到 `yolo_berry_detector`；训练完成前可继续用 YOLOE 跑全局。

---

## 命令速查

| 步骤 | 命令 |
|------|------|
| 采集 | `bash scripts/run_fixed_mono_label_pipeline.sh capture` |
| VLM 伪标（本地） | `bash scripts/run_fixed_mono_label_pipeline.sh prelabel` |
| VLM 伪标（API） | `bash scripts/run_fixed_mono_label_pipeline.sh prelabel-api` |
| 人工审核 | `bash scripts/run_fixed_mono_label_pipeline.sh annotate` |
| 合并数据集 | `bash scripts/run_fixed_mono_label_pipeline.sh merge` |
| 采集+伪标 | `bash scripts/run_fixed_mono_label_pipeline.sh all` |

---

## 脚本说明

| 文件 | 作用 |
|------|------|
| `capture_fixed_mono_guide.py` | 18 格交互采集清单 |
| `teacher_prelabel_vlm.py` | Qwen2.5-VL 出簇 bbox |
| `prepare_yolo_unified_dataset.py` | 腕部+固定合并为 2-class |
| `run_fixed_mono_label_pipeline.sh` | 一条龙入口 |

---

## 预期数据量

| 阶段 | 固定相机张数 | 说明 |
|------|-------------|------|
| 验证 | 18 + VLM + 1h 审核 | 明显好于 YOLOE |
| 可用 | +30 难例迭代 | 够簇规划联调 |
| 与腕部联合 | +330 腕图 | 一个 unified 模型端侧部署 |

VLM **不参与实时推理**，只当 Teacher；上车只有 YOLO11n 一个文件。
