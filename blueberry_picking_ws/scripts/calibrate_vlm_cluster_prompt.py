#!/usr/bin/env python3
"""Derive VLM cluster-prompt calibration from human-labeled fixed_mono dataset.

Writes datasets/fixed_mono/vlm_calibration.json used by teacher_prelabel_vlm.py.

Usage:
  python3 scripts/calibrate_vlm_cluster_prompt.py
  python3 scripts/calibrate_vlm_cluster_prompt.py --images datasets/fixed_mono/images
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List


def _analyze(images_dir: Path, labels_dir: Path) -> Dict[str, Any]:
    per_image: List[Dict[str, Any]] = []
    all_boxes: List[Dict[str, float]] = []
    ref_w, ref_h = 640, 480

    for lab in sorted(labels_dir.glob('*.txt')):
        stem = lab.stem
        img_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            p = images_dir / f'{stem}{ext}'
            if p.is_file():
                img_path = p
                break
        if img_path is None:
            continue

        import cv2
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        ref_w, ref_h = w, h

        lines = [ln.strip() for ln in lab.read_text(encoding='utf-8').splitlines() if ln.strip()]
        boxes = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            pw, ph = bw * w, bh * h
            boxes.append({
                'class_id': cls_id,
                'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh,
                'pw': pw, 'ph': ph,
                'area_frac': bw * bh,
                'aspect': bh / max(bw, 1e-9),
            })
        per_image.append({'stem': stem, 'n_clusters': len(boxes), 'boxes': boxes})
        all_boxes.extend(boxes)

    if not all_boxes:
        raise SystemExit(f'no labeled boxes in {labels_dir}')

    counts = [p['n_clusters'] for p in per_image]

    def _stats(key: str) -> Dict[str, float]:
        vals = [b[key] for b in all_boxes]
        return {
            'min': min(vals),
            'max': max(vals),
            'mean': st.mean(vals),
            'median': st.median(vals),
        }

    return {
        'source': 'human_labels',
        'n_images': len(per_image),
        'n_clusters_total': len(all_boxes),
        'image_size': [ref_w, ref_h],
        'clusters_per_image': {
            'min': min(counts),
            'max': max(counts),
            'mean': st.mean(counts),
            'median': st.median(counts),
            'histogram': {str(k): counts.count(k) for k in sorted(set(counts))},
        },
        'box_norm': {
            'bw': _stats('bw'),
            'bh': _stats('bh'),
            'area_frac': _stats('area_frac'),
            'aspect_bh_over_bw': _stats('aspect'),
        },
        'box_px': {
            'pw': _stats('pw'),
            'ph': _stats('ph'),
        },
        'per_image': per_image,
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=root / 'datasets' / 'fixed_mono' / 'images')
    parser.add_argument('--labels', type=Path, default=root / 'datasets' / 'fixed_mono' / 'labels')
    parser.add_argument('--out', type=Path, default=root / 'datasets' / 'fixed_mono' / 'vlm_calibration.json')
    args = parser.parse_args()

    cal = _analyze(args.images, args.labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(cal, f, indent=2, ensure_ascii=True)

    cpi = cal['clusters_per_image']
    bn = cal['box_norm']
    bp = cal['box_px']
    print(f'[calibrate] images={cal["n_images"]} clusters={cal["n_clusters_total"]}')
    print(f'  clusters/image: {cpi["min"]}–{cpi["max"]} (median {cpi["median"]})')
    print(f'  box px (median): {bp["pw"]["median"]:.0f} x {bp["ph"]["median"]:.0f}')
    print(f'  area_frac (median): {bn["area_frac"]["median"]:.4f}')
    print(f'  wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
