#!/usr/bin/env python3
"""Interactive minimal capture for fixed (global mono) camera — cluster labeling.

Move the plant and/or fixed camera between slots; press Enter to save one frame.
Designed for ~18–24 diverse frames (cheaper than blind interval capture).

Usage:
  python3 scripts/capture_fixed_mono_guide.py
  python3 scripts/capture_fixed_mono_guide.py --out datasets/fixed_mono/images
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

# Reuse batch capture ROS helpers
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, '..'))
sys.path.insert(0, SCRIPTS_DIR)

from capture_annotation_batch import (  # noqa: E402
    BatchCapture,
    _preflight,
    _resolve_start_index,
    _save_png_rgb,
)

import rclpy  # noqa: E402


@dataclass(frozen=True)
class CaptureSlot:
    slot_id: str
    hint: str


# Minimal diversity grid for cluster detection (user moves plant + camera).
CAPTURE_SLOTS: List[CaptureSlot] = [
    CaptureSlot('c_far_high', '植株居中 | 相机偏高、稍远 — 整簇入画'),
    CaptureSlot('c_far_low', '植株居中 | 相机偏低、稍远 — 斜看果簇'),
    CaptureSlot('c_near_high', '植株居中 | 相机偏高、稍近'),
    CaptureSlot('c_near_low', '植株居中 | 相机偏低、稍近'),
    CaptureSlot('l_far', '植株偏左（画面左侧）| 中等距离'),
    CaptureSlot('r_far', '植株偏右（画面右侧）| 中等距离'),
    CaptureSlot('l_near', '植株偏左 | 较近'),
    CaptureSlot('r_near', '植株偏右 | 较近'),
    CaptureSlot('front_left', '植株前左角 | 中等距离'),
    CaptureSlot('front_right', '植株前右角 | 中等距离'),
    CaptureSlot('back_center', '植株靠后居中 | 中等距离'),
    CaptureSlot('high_tilt', '相机大俯角 — 俯视多簇'),
    CaptureSlot('low_tilt', '相机小俯角 — 侧看簇'),
    CaptureSlot('close_cluster', '某一簇占画面 30–50%'),
    CaptureSlot('multi_cluster', '画面中同时 2–3 个簇'),
    CaptureSlot('edge_cluster', '主簇靠近画面边缘'),
    CaptureSlot('shadow', '一侧有阴影/反光（难例）'),
    CaptureSlot('extra', '任意补充难例（可重复按 Enter 多张）'),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--color-topic', default='/camera_fixed/color/image_raw')
    parser.add_argument('--out-dir', default=os.path.join(ROOT, 'datasets', 'fixed_mono', 'images'))
    parser.add_argument('--prefix', default='fm_')
    parser.add_argument('--startup-timeout', type=float, default=60.0)
    parser.add_argument('--extra-count', type=int, default=3,
                        help='How many optional extra captures after the checklist')
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not _preflight(args.color_topic, args.startup_timeout):
        return 1

    rclpy.init()
    node = BatchCapture(args.color_topic)
    try:
        print(f'[fixed_mono] Waiting for {args.color_topic} ...')
        if not node.wait_rgb(args.startup_timeout):
            print('ERROR: no RGB — start fixed camera first:', file=sys.stderr)
            print('  bash scripts/real_robot_bringup.sh --fixed-cam --camera-only', file=sys.stderr)
            return 1

        start_idx = _resolve_start_index(out_dir, args.prefix, 0)
        saved = 0
        slot_idx = 0
        extra_left = int(args.extra_count)

        print('\n' + '=' * 72)
        print('  固定单目 · 最小采集（每格一张，挪动植株/相机后按 Enter）')
        print('  Enter=保存  s=跳过本格  q=结束')
        print('=' * 72 + '\n')

        while slot_idx < len(CAPTURE_SLOTS) or extra_left > 0:
            if slot_idx < len(CAPTURE_SLOTS):
                slot = CAPTURE_SLOTS[slot_idx]
                title = f'[{slot_idx + 1}/{len(CAPTURE_SLOTS)}] {slot.slot_id}'
                hint = slot.hint
            else:
                title = f'[extra {args.extra_count - extra_left + 1}/{args.extra_count}]'
                hint = '补充难例'
                slot = CaptureSlot('extra', hint)

            print(f'\n--- {title} ---')
            print(f'  {hint}')
            print('  Enter=拍  s=跳过  q=退出')

            key = input('> ').strip().lower()
            if key in ('q', 'quit'):
                break
            if key in ('s', 'skip'):
                slot_idx += 1
                continue

            if not node.wait_new_rgb(None, 3.0):
                node.spin_for(0.5)
            rgb = node.grab_rgb()
            fname = f'{args.prefix}{slot.slot_id}_{start_idx + saved:03d}.png'
            path = os.path.join(out_dir, fname)
            _save_png_rgb(path, rgb)
            saved += 1
            print(f'  saved → {path}')

            if slot_idx < len(CAPTURE_SLOTS):
                slot_idx += 1
            else:
                extra_left -= 1

        print(f'\n[fixed_mono] Done: {saved} images → {out_dir}')
        print('Next: bash scripts/run_fixed_mono_label_pipeline.sh')
        return 0 if saved > 0 else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
