#!/usr/bin/env python3
"""Write fixed DaBai eye-to-hand extrinsics into env + calibration yaml.

Usage (explicit quat):
  python3 scripts/apply_fixed_cam_extrinsic.py \\
    --tx 0.55 --ty 0.20 --tz 0.50 \\
    --qx ... --qy ... --qz ... --qw ...

Usage (tape + look-at target in base_link; optical +Z toward target):
  python3 scripts/apply_fixed_cam_extrinsic.py \\
    --tx 0.55 --ty 0.20 --tz 0.50 \\
    --look-at 0.35,0.0,0.25
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cup_axis_plan import quat_cup_toward_berry  # noqa: E402


def _parse_xyz(s: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 3:
        raise SystemExit(f'expected x,y,z got {s!r}')
    return float(parts[0]), float(parts[1]), float(parts[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tx', type=float, required=True)
    ap.add_argument('--ty', type=float, required=True)
    ap.add_argument('--tz', type=float, required=True)
    ap.add_argument('--qx', type=float, default=None)
    ap.add_argument('--qy', type=float, default=None)
    ap.add_argument('--qz', type=float, default=None)
    ap.add_argument('--qw', type=float, default=None)
    ap.add_argument(
        '--look-at', default=None,
        help='base_link point x,y,z that optical +Z should aim at')
    ap.add_argument(
        '--frame', default='camera_fixed_color_optical_frame',
        help='Child optical frame id')
    args = ap.parse_args()

    if args.look_at is not None:
        lx, ly, lz = _parse_xyz(args.look_at)
        ax, ay, az = lx - args.tx, ly - args.ty, lz - args.tz
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-6:
            raise SystemExit('look-at coincides with camera origin')
        qx, qy, qz, qw = quat_cup_toward_berry(ax / n, ay / n, az / n)
        print(f'look-at ({lx},{ly},{lz}) → quat xyzw='
              f'[{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]')
    elif None not in (args.qx, args.qy, args.qz, args.qw):
        qx, qy, qz, qw = args.qx, args.qy, args.qz, args.qw
    else:
        raise SystemExit('Provide --look-at x,y,z OR all of --qx --qy --qz --qw')

    env = ROOT / 'config' / 'real_robot.env'
    text = env.read_text(encoding='utf-8')
    reps = {
        'FIXED_CAMERA_FRAME': args.frame,
        'FIXED_CAM_TX': f'{args.tx}',
        'FIXED_CAM_TY': f'{args.ty}',
        'FIXED_CAM_TZ': f'{args.tz}',
        'FIXED_CAM_QX': f'{qx:.4f}',
        'FIXED_CAM_QY': f'{qy:.4f}',
        'FIXED_CAM_QZ': f'{qz:.4f}',
        'FIXED_CAM_QW': f'{qw:.4f}',
    }
    for k, v in reps.items():
        text = re.sub(rf'(?m)^{k}=.*$', f'{k}={v}', text)
    env.write_text(text, encoding='utf-8')

    yaml = ROOT / 'src' / 'picking_description' / 'calibration' / 'fixed_camera_to_base.yaml'
    yaml.write_text(
        '# Eye-to-hand: DaBai global RGB-D optical → base_link\n'
        '# Synced with config/real_robot.env FIXED_CAM_*\n'
        '# Updated via scripts/apply_fixed_cam_extrinsic.py\n'
        'frame_id: base_link\n'
        f'child_frame_id: {args.frame}\n'
        f'translation: [{args.tx}, {args.ty}, {args.tz}]\n'
        f'rotation: [{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]  # xyzw\n',
        encoding='utf-8',
    )
    print(f'wrote {env}')
    print(f'wrote {yaml}')


if __name__ == '__main__':
    main()
