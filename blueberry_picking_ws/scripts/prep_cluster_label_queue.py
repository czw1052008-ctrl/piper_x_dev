#!/usr/bin/env python3
"""Build a cluster-crop labeling queue. Does not invent fruit circles.

Uses the existing semantic head only as a hint: crop around the berry-class blob
so you can paint each fruit tightly. Global/fixed views may be coarse; wrist first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))


def _stems_keep(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if s and not s.startswith('#'):
            out.add(Path(s).stem)
    return out


def _list_train_images(data: Path) -> list[Path]:
    d = data / 'train' / 'images'
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and p.name != '.gitkeep')


def _berry_bbox(sem: np.ndarray, pad: float = 0.22):
    ys, xs = np.where(sem == 1)
    if ys.size < 8:
        return None
    h, w = sem.shape[:2]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bh, bw = y1 - y0, x1 - x0
    py = int(max(8, pad * bh))
    px = int(max(8, pad * bw))
    y0 = max(0, y0 - py)
    x0 = max(0, x0 - px)
    y1 = min(h, y1 + py)
    x1 = min(w, x1 + px)
    return x0, y0, x1, y1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'datasets' / 'scene_seg'))
    parser.add_argument('--ckpt', default='', help='semantic ckpt for cluster crop hint')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out) if args.out else data / 'to_label'
    crops = out / 'crops'
    overlays = out / 'cluster_hint'
    crops.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    keep = _stems_keep(data / 'human_keep.txt')
    images = _list_train_images(data)
    unlabeled = [p for p in images if p.stem not in keep]
    wrist = [p for p in unlabeled if 'wrist' in p.stem]
    fixed = [p for p in unlabeled if 'wrist' not in p.stem]

    import torch
    from picking_perception.dinov3_seg import colorize_semantic, load_checkpoint, unpack_seg

    ckpt = args.ckpt or str(ROOT / 'runs' / 'seg' / 'dinov3-overfit-10' / 'best.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, _kind = load_checkpoint(ckpt, device=device)
    net.to(device)
    net.eval()
    size = int(torch.load(ckpt, map_location='cpu').get('size', 448))

    def infer_sem(bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        inp = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            logits, _hm = unpack_seg(net(x))
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    cards = []
    for path in unlabeled:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        sem = infer_sem(bgr)
        box = _berry_bbox(sem)
        cam = 'wrist' if 'wrist' in path.stem else 'fixed'
        pri = 0 if cam == 'wrist' else 1
        crop_rel = ''
        if box is not None:
            x0, y0, x1, y1 = box
            crop = bgr[y0:y1, x0:x1]
            hint = cv2.addWeighted(crop, 0.65, colorize_semantic(sem[y0:y1, x0:x1]), 0.35, 0)
            cv2.imwrite(str(crops / f'{path.stem}.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            cv2.imwrite(str(overlays / f'{path.stem}.jpg'), hint, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            crop_rel = f'crops/{path.stem}.jpg'
        n_berry_px = int((sem == 1).sum())
        cards.append((pri, cam, path.stem, crop_rel, n_berry_px))

    wrist_stems = [c[2] for c in sorted(cards) if c[1] == 'wrist']
    fixed_stems = [c[2] for c in sorted(cards) if c[1] == 'fixed']
    (out / 'queue_wrist.txt').write_text('\n'.join(wrist_stems) + '\n', encoding='utf-8')
    (out / 'queue_fixed.txt').write_text('\n'.join(fixed_stems) + '\n', encoding='utf-8')
    (out / 'queue_all_unlabeled.txt').write_text(
        '\n'.join(wrist_stems + fixed_stems) + '\n', encoding='utf-8')

    rows = []
    for _pri, cam, stem, crop_rel, n_px in sorted(cards):
        hint = f'cluster_hint/{stem}.jpg' if crop_rel else ''
        img = f'<img src="{crop_rel}" height="220">' if crop_rel else '(no berry pixels)'
        hint_img = f'<img src="{hint}" height="220">' if hint else ''
        rows.append(
            f'<tr><td>{cam}</td><td><code>{stem}</code><br>berry_px={n_px}</td>'
            f'<td>{img}</td><td>{hint_img}</td></tr>')
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>scene_seg cluster label queue</title></head><body>
<h1>逐颗贴果皮标注（{len(unlabeled)} 张未进 human_keep）</h1>
<p>品红半透明 = 现成语义<strong>簇</strong>提示，不是果。请在原图标注器里对<strong>每一颗</strong>点一次，膜要贴果皮。
圆心/半径事后从你的 mask 算，不要手画圆。</p>
<ol>
<li>优先腕部：<code>bash scripts/run_annotate_scene_seg.sh --queue datasets/scene_seg/to_label/queue_wrist.txt</code></li>
<li>固定相机可后做（全局不稳没关系）。</li>
<li>按 <code>1</code>，点果心：SAM 贴像素。不要按 <code>y</code> 用 YOLO 椭圆。</li>
</ol>
<p>keep 已标 10 张不在本队列。腕部 {len(wrist_stems)} 张，固定 {len(fixed_stems)} 张。</p>
<table border="1" cellpadding="6">{''.join(rows)}</table>
</body></html>
"""
    (out / 'index.html').write_text(html, encoding='utf-8')
    print(f'[to_label] unlabeled={len(unlabeled)} wrist={len(wrist_stems)} fixed={len(fixed_stems)}')
    print(f'[to_label] wrote {out / "index.html"}')
    print(f'[to_label] annotate: bash scripts/run_annotate_scene_seg.sh --queue {out / "queue_wrist.txt"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
