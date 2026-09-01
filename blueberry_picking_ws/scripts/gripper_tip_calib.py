#!/usr/bin/env python3
"""Fit agx_gripper tip_offset_link6 from fingertip contact at known base_link GT.

Usage:
  # 夹爪指尖轻触标定点后按 Enter（可多次采样取平均）
  python3 scripts/gripper_tip_calib.py --gt 0.097 0.337 0.254
  python3 scripts/gripper_tip_calib.py --gt 0.097 0.337 0.254 --apply
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from piper_position_ik import fk_link6, tip_xyz  # noqa: E402

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
PROFILE_PATH = ROOT / 'config' / 'end_effector_profiles.yaml'
PROFILE_NAME = 'agx_gripper_v1'


def _tip_offset_from_gt(
    joints: Sequence[float], gt: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, p = fk_link6(list(joints))
    gt_v = np.asarray(gt, dtype=np.float64).reshape(3)
    off = R.T @ (gt_v - p)
    tip = p + R @ off
    return off, tip, p


def _read_joints(timeout_s: float) -> Optional[List[float]]:
    """Prefer /feedback/joint_states; fall back to /joint_states."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node('gripper_tip_calib_read')
    got: dict = {'feedback': None, 'joint_states': None}

    def _extract(msg: JointState) -> Optional[List[float]]:
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(j in name_to_pos for j in ARM_JOINTS):
            return [float(name_to_pos[j]) for j in ARM_JOINTS]
        return None

    def on_feedback(msg: JointState) -> None:
        q = _extract(msg)
        if q is not None:
            got['feedback'] = q

    def on_js(msg: JointState) -> None:
        q = _extract(msg)
        if q is not None:
            got['joint_states'] = q

    node.create_subscription(JointState, '/feedback/joint_states', on_feedback, 10)
    node.create_subscription(JointState, '/joint_states', on_js, 10)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.2)
        if got['feedback'] is not None:
            break
    out = got['feedback'] if got['feedback'] is not None else got['joint_states']
    node.destroy_node()
    rclpy.shutdown()
    return out


def _apply_profile(off: Sequence[float]) -> None:
    text = PROFILE_PATH.read_text(encoding='utf-8')
    block = re.search(
        rf'({PROFILE_NAME}:\n(?:  .+\n)+)',
        text,
    )
    if not block:
        raise RuntimeError(f'profile {PROFILE_NAME} not found in {PROFILE_PATH}')
    old = block.group(1)
    new_off = (
        f'  tip_offset_link6: [{off[0]:.6f}, {off[1]:.6f}, {off[2]:.6f}]\n'
    )
    updated = re.sub(
        r'  tip_offset_link6: \[[^\]]+\]\n',
        new_off,
        old,
        count=1,
    )
    PROFILE_PATH.write_text(text.replace(old, updated), encoding='utf-8')


def run(args: argparse.Namespace) -> int:
    gt = [float(v) for v in args.gt]
    samples: List[dict] = []
    n = max(1, int(args.samples))
    once = bool(getattr(args, 'once', False))
    if once:
        n = 1

    print('夹爪 TCP 标定 — 指尖轻触已知 base_link 标定点')
    print(f'GT base_link (m): {tuple(round(x, 4) for x in gt)}')
    if once:
        print('模式: --once（立即采 1 次当前关节）')
    else:
        print(f'采样次数: {n}（每次对准后按 Enter，q 结束）')
    print()

    for i in range(n):
        if not once:
            prompt = f'[{i + 1}/{n}] 对准后 Enter（q 结束）: '
            try:
                line = input(prompt).strip().lower()
            except EOFError:
                break
            if line in ('q', 'quit', 'exit'):
                break
        else:
            print(f'[{i + 1}/{n}] 采样中 ...')
        joints = _read_joints(float(args.joint_timeout_s))
        if joints is None:
            print('ERROR: 未收到 joint_states（feedback 或 /joint_states）',
                  file=sys.stderr)
            return 2
        off, tip, p_l6 = _tip_offset_from_gt(joints, gt)
        err = float(np.linalg.norm(tip - np.asarray(gt)))
        row = {
            'joints_rad': joints,
            'tip_offset_link6': off.tolist(),
            'tip_base_fk': tip.tolist(),
            'link6_base': p_l6.tolist(),
            'err_m': err,
        }
        samples.append(row)
        print(
            f'  off_link6=({off[0]:.5f}, {off[1]:.5f}, {off[2]:.5f})  '
            f'err={err * 1000:.2f} mm'
        )

    if not samples:
        print('无采样', file=sys.stderr)
        return 3

    offs = np.array([s['tip_offset_link6'] for s in samples], dtype=np.float64)
    mean_off = offs.mean(axis=0)
    std_off = offs.std(axis=0) if len(samples) > 1 else np.zeros(3)
    verify = [
        float(np.linalg.norm(
            tip_xyz(s['joints_rad'], tip_offset_link6=mean_off) - np.asarray(gt)))
        for s in samples
    ]

    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = ROOT / 'log' / 'real_robot' / 'gripper_tip_calib' / session
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'session': session,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'gt_contact_base': gt,
        'profile': PROFILE_NAME,
        'n_samples': len(samples),
        'tip_offset_link6_mean': mean_off.tolist(),
        'tip_offset_link6_std': std_off.tolist(),
        'verify_err_m': verify,
        'samples': samples,
    }
    out_json = out_dir / 'gripper_tip_calib.json'
    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print()
    print(f'均值 tip_offset_link6: ({mean_off[0]:.6f}, {mean_off[1]:.6f}, {mean_off[2]:.6f})')
    if len(samples) > 1:
        print(
            f'标准差 (mm): ({std_off[0]*1000:.2f}, {std_off[1]*1000:.2f}, {std_off[2]*1000:.2f})'
        )
    print(f'回代误差 (mm): {[round(e * 1000, 2) for e in verify]}')
    print(f'json={out_json}')
    print()
    print('建议写入 real_robot.env:')
    print(f'  ARM_TCP_OFFSET=[0,0,0,0,0,0]  # 偏移已在 {PROFILE_NAME}.tip_offset_link6')
    print(f'  # end_effector_profiles.yaml → tip_offset_link6: {mean_off.round(6).tolist()}')

    if args.apply:
        _apply_profile(mean_off)
        print(f'已写入 {PROFILE_PATH} → {PROFILE_NAME}.tip_offset_link6')

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', nargs=3, type=float, required=True, metavar=('X', 'Y', 'Z'))
    p.add_argument('--samples', type=int, default=3, help='Max contact samples')
    p.add_argument('--once', action='store_true',
                   help='Non-interactive: sample current pose once immediately')
    p.add_argument('--joint-timeout-s', type=float, default=4.0)
    p.add_argument('--apply', action='store_true',
                   help=f'Write mean offset to {PROFILE_NAME} in end_effector_profiles.yaml')
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
