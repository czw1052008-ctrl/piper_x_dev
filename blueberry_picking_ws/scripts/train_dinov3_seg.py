#!/usr/bin/env python3
"""Train a frozen-DINOv3 (or TinySeg fallback) head on scene_seg YOLO-seg / PNG masks.

P2: prefer YOLO-seg txt (one polygon per fruit) and train a berry-center heatmap.
Never loads a previous seg head unless --resume is set. Backbone = local DINOv3.

  PYTHONPATH=src/picking_perception \\
    /home/user/miniconda3/envs/dreamzero/bin/python scripts/train_dinov3_seg.py \\
    --data datasets/scene_seg --keep datasets/scene_seg/human_keep.txt \\
    --out runs/seg/dinov3-p2-inst --epochs 80 --no-tiny --size 448 \\
    --resume runs/seg/dinov3-overfit-10/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))

from picking_perception.dinov3_seg import NUM_CLASSES  # noqa: E402
from picking_perception.instance_gt import (  # noqa: E402
    centers_from_inst,
    centers_from_yolo_det,
    heatmap_from_centers,
    maps_from_yolo_seg,
    resize_centers,
)

IM_SIZE = 224
HM_LOSS_W = 1.0
IGNORE = -1


def heatmap_focal_loss(pred_hm, target, *, pos_thr: float = 0.85):
    """CornerNet / CenterNet penalty-reduced focal loss on a Gaussian heatmap."""
    pred = pred_hm.clamp(1e-4, 1.0 - 1e-4)
    pos = target >= pos_thr
    neg_w = (1.0 - target).clamp(min=0.0).pow(4)
    loss_pos = -((1.0 - pred).square()) * pred.log()
    loss_neg = -(pred.square()) * neg_w * (1.0 - pred).log()
    n_pos = pos.sum().clamp(min=1.0)
    return (loss_pos[pos].sum() + loss_neg[~pos].sum()) / n_pos


def load_pair(img_path: Path, lab_dir: Path) -> tuple:
    """RGB, semantic uint8, berry-center heatmap. Prefer txt so instances are kept."""
    import cv2
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(img_path)
    h, w = bgr.shape[:2]
    stem = img_path.stem
    png = lab_dir / f'{stem}_mask.png'
    txt = lab_dir / f'{stem}.txt'
    inst_png = lab_dir / f'{stem}_inst.png'
    mask = np.zeros((h, w), dtype=np.uint8)
    centers: list = []
    if txt.is_file():
        mask, _inst, centers = maps_from_yolo_seg(txt, h, w)
    if png.is_file():
        m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape[:2] == (h, w):
            if int((mask > 0).sum()) == 0:
                mask = np.clip(m, 0, NUM_CLASSES - 1)
            else:
                other = np.clip(m, 0, NUM_CLASSES - 1)
                mask = np.where(other > 1, other, mask)
    if inst_png.is_file():
        im = cv2.imread(str(inst_png), cv2.IMREAD_UNCHANGED)
        if im is not None and im.shape[:2] == (h, w) and int((im > 0).sum()) > 0:
            centers = centers_from_inst(im.astype(np.uint16))
            mask[im > 0] = 1
    hm = heatmap_from_centers(h, w, centers)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, mask.astype(np.int16), hm, centers


def load_yolo_hm_pair(img_path: Path, txt: Path) -> tuple:
    """Detect boxes → heatmap only. Semantic pixels are IGNORE (no bg/branch/rigid/ego)."""
    import cv2
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(img_path)
    h, w = bgr.shape[:2]
    centers = centers_from_yolo_det(txt, h, w, berry_cls=(0,))
    mask = np.full((h, w), IGNORE, dtype=np.int16)
    hm = heatmap_from_centers(h, w, centers)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, mask, hm, centers


def _txt_has_berry_box(txt: Path) -> bool:
    if not txt.is_file():
        return False
    for line in txt.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        if int(float(parts[0])) != 0:
            continue
        bw, bh = float(parts[3]), float(parts[4])
        if 0.0 < bw * bh <= 0.03:
            return True
    return False


def list_yolo_det_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    cands: list[tuple[Path, Path]] = []
    if (root / 'labels').is_dir() and (root / 'images').is_dir():
        cands.append((root / 'images', root / 'labels'))
    if (root / 'train' / 'labels').is_dir() and (root / 'train' / 'images').is_dir():
        cands.append((root / 'train' / 'images', root / 'train' / 'labels'))
    for img_dir, lab_dir in cands:
        for txt in sorted(lab_dir.glob('*.txt')):
            if txt.stem in seen or not _txt_has_berry_box(txt):
                continue
            img = None
            for ext in ('.png', '.jpg', '.jpeg'):
                p = img_dir / f'{txt.stem}{ext}'
                if p.is_file():
                    img = p
                    break
            if img is None:
                continue
            seen.add(txt.stem)
            pairs.append((img, txt))
    return pairs


def _has_label(lab_dir: Path, stem: str) -> bool:
    return (lab_dir / f'{stem}.txt').is_file() or (lab_dir / f'{stem}_mask.png').is_file()


def _load_keep(path: str) -> set[str] | None:
    if not path:
        return None
    stems: set[str] = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            stems.add(Path(s).stem)
    return stems


class SceneSegDataset:
    def __init__(
        self,
        root: Path,
        split: str,
        size: int = IM_SIZE,
        keep: set[str] | None = None,
        labeled_only: bool = True,
        hflip: bool = False,
        yolo_det: Path | None = None,
    ) -> None:
        self.size = size
        self.hflip = hflip
        img_dir = root / split / 'images'
        self.lab_dir = root / split / 'labels'
        self.items: list[tuple[str, Path, Path | None]] = []
        if img_dir.is_dir():
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in ('.png', '.jpg', '.jpeg') or p.name == '.gitkeep':
                    continue
                if keep is not None and p.stem not in keep:
                    continue
                if labeled_only and not _has_label(self.lab_dir, p.stem):
                    continue
                self.items.append(('sem', p, None))
        self.n_sem = len(self.items)
        if yolo_det is not None and Path(yolo_det).is_dir():
            for img, txt in list_yolo_det_pairs(Path(yolo_det)):
                self.items.append(('hm', img, txt))
        self.n_hm = len(self.items) - self.n_sem
        self.paths = [it[1] for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import cv2
        import torch
        kind, img_path, txt = self.items[i]
        if kind == 'hm':
            rgb, mask, hm, centers = load_yolo_hm_pair(img_path, txt)
        else:
            rgb, mask, hm, centers = load_pair(img_path, self.lab_dir)
        src_hw = rgb.shape[:2]
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.int16), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if centers:
            hm = heatmap_from_centers(
                self.size, self.size, resize_centers(centers, src_hw=src_hw, dst_hw=(self.size, self.size)))
        else:
            hm = cv2.resize(hm, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if self.hflip and int(np.random.randint(0, 2)) == 1:
            rgb = rgb[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            hm = hm[:, ::-1].copy()
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        y = torch.from_numpy(mask.astype(np.int64))
        hmt = torch.from_numpy(hm.astype(np.float32))
        return x, y, hmt


def class_weights(ds: SceneSegDataset, n_cls: int):
    import torch
    counts = np.zeros(n_cls, dtype=np.float64)
    flip = ds.hflip
    ds.hflip = False
    for i in range(len(ds)):
        _, y, _hm = ds[i]
        yy = y.numpy().reshape(-1)
        yy = yy[yy >= 0]
        if yy.size:
            counts += np.bincount(yy, minlength=n_cls)[:n_cls]
    ds.hflip = flip
    freq = counts / max(counts.sum(), 1.0)
    w = 1.0 / np.log(1.02 + np.maximum(freq, 1e-8))
    w[counts == 0] = 0.05  # no GT for this class: do not dominate
    w = w / w.mean()
    print('[train] px%', {i: round(float(freq[i]), 4) for i in range(n_cls)})
    print('[train] w', {i: round(float(w[i]), 3) for i in range(n_cls)})
    empty = [i for i in range(n_cls) if counts[i] == 0]
    return torch.tensor(w, dtype=torch.float32), empty


def make_dummy(n: int, size: int = IM_SIZE):
    import torch
    xs, ys, hs = [], [], []
    for _ in range(n):
        x = torch.rand(3, size, size)
        y = torch.zeros(size, size, dtype=torch.long)
        y[20:40, 20:40] = 1
        y[80:90, 40:180] = 2
        y[150:200, 20:200] = 3
        y[10:30, 150:210] = 4
        hm = torch.zeros(size, size)
        hm[30, 30] = 1.0
        xs.append(x)
        ys.append(y)
        hs.append(hm)
    return torch.stack(xs), torch.stack(ys), torch.stack(hs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'datasets' / 'scene_seg'))
    parser.add_argument('--out', default=str(ROOT / 'runs' / 'seg' / 'dinov3-p2-inst'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--size', type=int, default=IM_SIZE)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-dino', action='store_true', help='force TinySeg CNN')
    parser.add_argument('--no-tiny', action='store_true', help='skip TinySeg distillation copy')
    parser.add_argument('--keep', default='', help='txt of image stems to train on')
    parser.add_argument('--resume', default='',
                        help='load existing head (default: random head + frozen DINOv3)')
    parser.add_argument('--yolo-det', default=str(ROOT / 'datasets' / 'blueberry'),
                        help='YOLO detect dataset (box GT → heatmap only; class 1 cluster skipped)')
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from picking_perception.dinov3_seg import (
        DEFAULT_MODEL_ID,
        TinySegNet,
        build_seg_model,
        load_checkpoint,
        unpack_seg,
        wrap_for_train,
    )

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    if args.resume:
        print(f'[train] resume head from {args.resume}')
        net, kind = load_checkpoint(args.resume, device='cpu')
        net = net.to(device)
        inner = getattr(net, 'inner', None)
    else:
        inner = build_seg_model(prefer_dino=not args.no_dino, model_id=args.backbone)
        net = wrap_for_train(inner).to(device)
        kind = getattr(inner, 'kind', 'tiny')
        print('[train] random 1x1 head, frozen DINOv3 (not loading dinov3-scene-1)')
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weight = None
    empty_cls: list[int] = []
    if args.dummy:
        x, y, hm = make_dummy(max(args.batch, 4), args.size)
        loader = [(x, y, hm)]
        print(f'[train] dummy tensors kind={kind} device={device}')
    else:
        keep = _load_keep(args.keep)
        ds = SceneSegDataset(
            Path(args.data), 'train', size=args.size, keep=keep,
            labeled_only=True, hflip=True, yolo_det=Path(args.yolo_det) if args.yolo_det else None)
        if len(ds) == 0:
            print('[train] no labeled images; use --dummy or add *.txt / *_mask.png')
            sys.exit(2)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=min(args.batch, len(ds)), shuffle=True, drop_last=False)
        n_centers = []
        flip = ds.hflip
        ds.hflip = False
        for sample_kind, p, txt in ds.items:
            if sample_kind == 'hm':
                _rgb, _m, _h, c = load_yolo_hm_pair(p, txt)
            else:
                _rgb, _m, _h, c = load_pair(p, ds.lab_dir)
            n_centers.append(len(c))
        ds.hflip = flip
        print(f'[train] n={len(ds)} sem={ds.n_sem} yolo_hm={ds.n_hm} kind={kind} device={device}')
        print(f'[train] berry centers/image mean={float(np.mean(n_centers) if n_centers else 0):.1f} '
              f'(scene_seg + YOLO boxes; cluster class ignored)')
        weight, empty_cls = class_weights(ds, NUM_CLASSES)
        weight = weight.to(device)
        if empty_cls:
            print(f'[train] no GT for classes {empty_cls}, penalize those logits')

    net.train()
    last = 0.0
    last_ce = 0.0
    last_hm = 0.0
    for epoch in range(args.epochs):
        for batch in loader:
            xb, yb, hmb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            hmb = hmb.to(device)
            logits, hm_pred = unpack_seg(net(xb))
            valid = (yb >= 0)
            if bool(valid.any()):
                loss_ce = F.cross_entropy(logits, yb, weight=weight, ignore_index=IGNORE)
            else:
                loss_ce = logits.new_zeros(())
            if empty_cls:
                loss_ce = loss_ce + 0.15 * logits[:, empty_cls].mean()
            loss_hm = logits.new_zeros(())
            if hm_pred is not None:
                pred_hm = torch.sigmoid(hm_pred[:, 0])
                loss_hm = heatmap_focal_loss(pred_hm, hmb)
            loss = loss_ce + HM_LOSS_W * loss_hm
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
            last_ce = float(loss_ce.item())
            last_hm = float(loss_hm.item()) if hm_pred is not None else 0.0
            last_hm_max = float(pred_hm.max().item()) if hm_pred is not None else 0.0
        print(
            f'[train] epoch {epoch + 1}/{args.epochs} '
            f'loss={last:.4f} ce={last_ce:.4f} hm={last_hm:.4f} hm_max={last_hm_max:.3f}')

    ckpt = {
        'kind': kind,
        'num_classes': NUM_CLASSES,
        'model_id': getattr(inner, 'model_id', DEFAULT_MODEL_ID) if inner is not None else DEFAULT_MODEL_ID,
        'size': args.size,
        'has_berry_hm': kind == 'dinov3',
    }
    if kind == 'dinov3' and hasattr(net, 'inner'):
        ckpt['head'] = net.inner.head.state_dict()
        ckpt['berry_hm'] = net.inner.berry_hm.state_dict()
        net.has_berry_hm = True
        net.inner.has_berry_hm = True
    else:
        ckpt['state_dict'] = net.state_dict()
        ckpt['kind'] = 'tiny'
        ckpt['has_berry_hm'] = False
    if not args.dummy and not args.no_tiny:
        tiny = TinySegNet(num_classes=NUM_CLASSES).net.to(device)
        t_opt = torch.optim.Adam(tiny.parameters(), lr=args.lr)
        tiny.train()
        t_last = 0.0
        for epoch in range(min(args.epochs, 15)):
            for batch in loader:
                xb, yb = batch[0], batch[1]
                xb = xb.to(device)
                yb = yb.to(device)
                loss = F.cross_entropy(tiny(xb), yb, weight=weight)
                t_opt.zero_grad()
                loss.backward()
                t_opt.step()
                t_last = float(loss.item())
            print(f'[train] tiny epoch {epoch + 1} loss={t_last:.4f}')
        ckpt['tiny_state_dict'] = {k: v.cpu() for k, v in tiny.state_dict().items()}
    path = out / 'best.pt'
    torch.save(ckpt, path)
    print(f'[train] wrote {path}')


if __name__ == '__main__':
    main()
