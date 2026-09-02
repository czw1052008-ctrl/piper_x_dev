#!/usr/bin/env python3
"""Overlay DINOv3-seg predictions. Does not write labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'picking_perception'))


def _images(data: Path) -> list[Path]:
    paths: list[Path] = []
    for split in ('train', 'val'):
        d = data / split / 'images'
        if d.is_dir():
            paths.extend(sorted(
                p for p in d.iterdir()
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg')))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data', default=str(ROOT / 'datasets' / 'scene_seg'))
    parser.add_argument('--out', default=str(ROOT / 'datasets' / 'scene_seg' / 'preview_p2_circles'))
    parser.add_argument('--keep', default='', help='if set, skip these stems (already labeled)')
    parser.add_argument('--only-keep', action='store_true', help='infer keep list only (sanity)')
    parser.add_argument('--size', type=int, default=0)
    args = parser.parse_args()

    import torch
    from picking_perception.dinov3_seg import (
        colorize_semantic, load_checkpoint, maps_from_logits, unpack_seg,
    )
    from picking_perception.berry_instances import (
        contours_from_instances,
        overlay_instance_contours,
    )
    from picking_perception.instance_gt import peaks_from_heatmap

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, kind = load_checkpoint(args.ckpt, device=device)
    net.to(device)
    net.eval()
    size = int(args.size)
    if size <= 0:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        size = int(ckpt.get('size', 224))
    use_hm = bool(getattr(net, 'has_berry_hm', False))
    print(f'[infer] kind={kind} size={size} device={device} berry_hm={use_hm}')

    keep: set[str] = set()
    if args.keep:
        for line in Path(args.keep).read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if s and not s.startswith('#'):
                keep.add(Path(s).stem)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in _images(Path(args.data)):
        if keep:
            if args.only_keep and path.stem not in keep:
                continue
            if not args.only_keep and path.stem in keep:
                continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        x = x.to(device)
        with torch.no_grad():
            logits, hm = unpack_seg(net(x))
        pred, inst, hm_np, _circles = maps_from_logits(
            logits, hm, (h, w), use_heatmap=use_hm)
        contours = contours_from_instances(pred, inst)
        tint = colorize_semantic(pred)
        vis = cv2.addWeighted(bgr, 0.62, tint, 0.38, 0)
        vis = overlay_instance_contours(vis, inst, contours, rgb=False)
        n_pk = 0
        if hm_np is not None:
            berry = pred == 1
            for px, py, sc in peaks_from_heatmap(hm_np, min_score=0.22, min_dist_px=6):
                if not berry[py, px]:
                    continue
                cv2.drawMarker(vis, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
                n_pk += 1
        n_berry = int((pred == 1).sum() > 0)
        lab = path.parent.parent / 'labels' / f'{path.stem}.txt'
        n_gt = ''
        if lab.is_file():
            n_gt = sum(1 for line in lab.read_text().splitlines() if line.strip().startswith('0 '))
            n_gt = f' gt={n_gt}'
        counts = {k: int((pred == i).sum()) for i, k in enumerate(
            ('bg', 'berry', 'branch', 'rigid', 'ego')) if i > 0}
        cv2.putText(
            vis, f'{path.stem}  peaks={n_pk}{n_gt}  {counts}', (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f'{path.stem}.jpg'), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        n += 1
    print(f'[infer] wrote {n} → {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
