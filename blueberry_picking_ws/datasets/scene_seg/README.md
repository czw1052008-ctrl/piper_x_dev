# Scene segmentation 数据集（果 / 枝条 / 硬障碍 / 自身）

替代 2 cm voxel occupancy。监督必须是 **多边形 mask**，不是 berry YOLO 框。

## 类别

| YOLO-seg class | PNG 像素 | Overlay | 含义 | 画法 |
|----------------|----------|---------|------|------|
| 0 berry | 1 | 品红 | 单颗浆果 | 每颗一个多边形，贴果皮，不要包整簇 |
| 1 branch | 2 | 绿 | 枝条 / 细梗 | **沿线画细条**，不要包成冠层团 |
| 2 rigid | 3 | 红 | 花盆、桌面、电钻、显示器等硬物 | 实例外轮廓 |
| 3 ego | 4 | 青 | 吸杯、夹爪、腕部相机壳、可见连杆 | 贴外形；**不要**标成 rigid |

背景 = 0，不标。叶子不算一类，保持背景。

`ego` 只给分割头当监督。抬升时丢掉，**不进** `/perception/obstacles`。规划器自身避碰仍用关节 FK，不靠这张 mask。

## 目录

```
datasets/scene_seg/
  raw/                 # 同步 RGB-D（fixed+wrist）
  overlays/*.jpg       # 初标复核图（品红=果 绿=枝 红=硬物）
  train/images/*.png
  train/labels/*.txt   # YOLO-seg：每行 class x1 y1 x2 y2 ... (归一化 0–1)
  val/images/
  val/labels/
```

同名 `stem.png` ↔ `stem.txt`。也可用同名 `stem_mask.png`（uint8：0/1/2/3/4）和 `stem_inst.png`（uint16 果实例 id）。语义 PNG 表示 **果区（簇）**；逐颗果心来自 txt 质心。不要用 mask PNG 把多颗果压成一类再当颗 GT。

## 采集

多视角（会动腕部，采完回入口位）：

```bash
bash scripts/run_collect_scene_seg_views.sh
```

只抓当前帧、不驱臂：

```bash
bash scripts/run_capture_scene_seg.sh
# 终端里回车或按 s 保存一帧（固定+腕部 RGB+depth）
```

保存到 `datasets/scene_seg/raw/`。建议 80–150 张、两相机、远近都有。当前批次约 22 视角 × 2 相机。

## 标注

初标（自动，**请务必看 overlays 再改**）：

```bash
bash scripts/run_auto_label_scene_seg.sh
# 复核：datasets/scene_seg/overlays/
```

手动精修（推荐，不用画多边形）：

```bash
bash scripts/run_annotate_scene_seg.sh
```

- `1` 后点果心：SAM 贴这一颗像素（一下一颗）。每种颜色一个 id，连成一块就说明标并了
- 不要按 `y`（YOLO 椭圆贴不上果皮）
- `2` + 拖动画笔：沿线刷枝
- `3` + `s` / `f`：硬物
- `4`：吸杯/夹爪/可见连杆
- Shift+左键 / `e` 擦除，`x` 删这一颗，`u` 撤销，`d`/`a` 下一张

未标队列（优先腕部，固定相机可后做）：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONPATH=src/picking_perception \
  /home/user/miniconda3/envs/dreamzero/bin/python scripts/prep_cluster_label_queue.py
# 打开 datasets/scene_seg/to_label/index.html 看簇裁剪
bash scripts/run_annotate_scene_seg.sh --queue datasets/scene_seg/to_label/queue_wrist.txt
```

也可用 Labelme 多边形：`bash scripts/run_labelme_scene_seg.sh`

## 训练

冻仓库内 **DINOv3 ViT-S**（`models/dinov3-vits16-pretrain-lvd1689m/`），只训分割头。不再用 DINOv2。

```bash
cd blueberry_picking_ws
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONPATH=src/picking_perception \
  /home/user/miniconda3/envs/dreamzero/bin/python scripts/train_dinov3_seg.py \
  --data datasets/scene_seg --keep datasets/scene_seg/human_keep.txt \
  --out runs/seg/dinov3-p2-inst --epochs 80 --no-tiny --size 448 \
  --resume runs/seg/dinov3-overfit-10/best.pt
```

训练优先 `*.txt` 的果心，同时训语义 CE + 果心热力图。推理：**品红 = 簇（门）**；圆 = 簇里的颗。不要加载 `dinov3-scene-1`。

推理预览：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONPATH=src/picking_perception \
  /home/user/miniconda3/envs/dreamzero/bin/python scripts/infer_scene_seg.py \
  --ckpt runs/seg/dinov3-p2-inst/best.pt --keep datasets/scene_seg/human_keep.txt \
  --only-keep --out datasets/scene_seg/preview_p2_circles
```
