#!/usr/bin/env python3
"""Fit wrist optical pitch (Rx about link6 X) from tip–berry GT.

Given tip→berry ΔZ and cam_optical_center→berry ΔZ in link6 (or base with
known T), solve Rx so optical +Z aligns with the observed geometry.

Example (legacy DaBai session 20260807_171055):
  tip_to_berry_z_m=0.025, cam_center_to_berry_z_m=0.055, ty=-0.08, tz=-0.04
  → Rx ≈ -23.5 deg

Usage:
  python3 scripts/fit_wrist_optical_pitch.py \\
    --tip-to-berry-z 0.025 --cam-center-to-berry-z 0.055 \\
    --ty -0.08 --tz -0.04 --apply
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fit_rx_rad(
    tip_to_berry_z: float,
    cam_center_to_berry_z: float,
    ty: float,
    tz: float,
) -> float:
    """Return Rx (rad) for optical pitched about link6 +X.

    Model: cam origin in link6 = (0, ty, tz) with R = Rx(a).
    Optical +Z in link6 = (0, -sin(a), cos(a)).
    Berry along optical ray at distance d_cam from cam center, with tip at
    link6 origin + tip_z along +Z (simplified: tip≈link6 origin for ΔZ fit).

    We match: berry_z_link6 ≈ tip_to_berry_z
            and |berry - cam| projected ≈ cam_center_to_berry_z along optical.

    Practical closed form used on 20260807: the cam→berry optical depth d and
    tip→berry ΔZ constrain pitch via
        tip_z ≈ tz + d * cos(a)
        (and lateral ty + d * (-sin(a)) ≈ 0 for berry on tip axis)
    → a = atan2(ty, tip_to_berry_z - tz) when berry is on tip axis,
      with d = cam_center_to_berry_z.
    """
    # Prefer lateral+depth consistency when berry is near tip axis:
    # ty - d*sin(a) ≈ 0  →  sin(a) = ty/d
    # tz + d*cos(a) ≈ tip_z → cos(a) = (tip_z - tz)/d
    d = float(cam_center_to_berry_z)
    if abs(d) < 1e-6:
        raise SystemExit('cam_center_to_berry_z must be non-zero')
    s = float(ty) / d
    c = (float(tip_to_berry_z) - float(tz)) / d
    n = math.hypot(s, c)
    if n < 1e-9:
        raise SystemExit('degenerate geometry')
    s /= n
    c /= n
    return math.atan2(s, c)


def _rewrite_env(path: Path, rpy: str, tx: float, ty: float, tz: float) -> None:
    text = path.read_text(encoding='utf-8')
    text = re.sub(r'(?m)^CAMERA_MOUNT_TX=.*$', f'CAMERA_MOUNT_TX={tx}', text)
    text = re.sub(r'(?m)^CAMERA_MOUNT_TY=.*$', f'CAMERA_MOUNT_TY={ty}', text)
    text = re.sub(r'(?m)^CAMERA_MOUNT_TZ=.*$', f'CAMERA_MOUNT_TZ={tz}', text)
    text = re.sub(r'(?m)^CAMERA_MOUNT_RPY=.*$', f'CAMERA_MOUNT_RPY={rpy}', text)
    path.write_text(text, encoding='utf-8')


def _rewrite_yaml(path: Path, rx: float, tx: float, ty: float, tz: float) -> None:
    q = math.sin(rx / 2.0)
    text = (
        '# Eye-in-hand: Piper X flange (link6) -> Gemini 305 optical frame.\n'
        f'# Fit Rx={math.degrees(rx):.2f}deg via scripts/fit_wrist_optical_pitch.py\n'
        'frame_id: link6\n'
        'child_frame_id: camera_wrist_link\n'
        f'translation: [{tx}, {ty}, {tz}]\n'
        f'rotation: [{q:.6f}, 0.0, 0.0, {math.cos(rx / 2.0):.6f}]\n'
        f'rpy_rad: [{rx:.6f}, 0.0, 0.0]\n'
        f'rpy_deg: [{math.degrees(rx):.2f}, 0.0, 0.0]\n'
    )
    path.write_text(text, encoding='utf-8')


def _rewrite_xacro(path: Path, rx: float, tx: float, ty: float, tz: float) -> None:
    text = path.read_text(encoding='utf-8')
    # Prefer the flange→camera_wrist_link joint (not optical child joints).
    pat = re.compile(
        r'(<joint name="camera_wrist_joint"[^>]*>.*?<origin xyz=")[^"]+(" rpy=")[^"]+("/>)',
        re.DOTALL,
    )
    repl = rf'\g<1>{tx} {ty} {tz}\g<2>{rx:.6f} 0 0\g<3>'
    new, n = pat.subn(repl, text, count=1)
    if n != 1:
        raise SystemExit(f'camera_wrist_joint origin not found in {path}')
    path.write_text(new, encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tip-to-berry-z', type=float, default=None,
                    help='link6 ΔZ tip→berry (m); with --cam-center-to-berry-z for geometric solve')
    ap.add_argument('--cam-center-to-berry-z', type=float, default=None,
                    help='optical depth cam→berry (m) when berry near image center')
    ap.add_argument('--rx-deg', type=float, default=None,
                    help='Direct Rx in degrees (preferred when you already fit offline)')
    ap.add_argument('--tx', type=float, default=0.0)
    ap.add_argument('--ty', type=float, default=-0.08)
    ap.add_argument('--tz', type=float, default=-0.04)
    ap.add_argument('--apply', action='store_true', help='Write env/yaml/xacro')
    args = ap.parse_args()

    if args.rx_deg is not None:
        rx = math.radians(float(args.rx_deg))
    elif args.tip_to_berry_z is not None and args.cam_center_to_berry_z is not None:
        rx = fit_rx_rad(
            args.tip_to_berry_z, args.cam_center_to_berry_z, args.ty, args.tz)
    else:
        raise SystemExit('Provide --rx-deg OR both --tip-to-berry-z and --cam-center-to-berry-z')

    rpy = f'{rx:.6f},0.0,0.0'
    print(f'Rx_rad={rx:.6f}  Rx_deg={math.degrees(rx):.2f}')
    print(f'CAMERA_MOUNT_RPY={rpy}')
    print(f'translation=[{args.tx}, {args.ty}, {args.tz}]')

    if not args.apply:
        return
    _rewrite_env(ROOT / 'config' / 'real_robot.env', rpy, args.tx, args.ty, args.tz)
    _rewrite_yaml(
        ROOT / 'src' / 'picking_description' / 'calibration' / 'wrist_camera_to_ee.yaml',
        rx, args.tx, args.ty, args.tz)
    _rewrite_xacro(
        ROOT / 'src' / 'picking_description' / 'urdf' / 'wrist_camera.urdf.xacro',
        rx, args.tx, args.ty, args.tz)
    print('applied to real_robot.env, wrist_camera_to_ee.yaml, wrist_camera.urdf.xacro')


if __name__ == '__main__':
    main()
