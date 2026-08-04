#!/usr/bin/env python3
"""Write one ALIGNING decision JSON for reach_fsm_node file-judge mode.

Agent estimates HOW MUCH to move (position control), then after hold judges coarse facing.

Examples:

  # Command phase: absolute joint targets (degrees)
  python3 scripts/write_align_decision.py \\
    --session-dir log/real_robot/qa/20260804_200431 \\
    --step-idx 0 --phase command --action set_joints \\
    --joints-deg 'joint1=26,joint2=-12,joint3=15,joint5=-43' \\
    --reason 'fixed: plant ~25deg right of tip; aim j1 to plant_yaw, pitch down'

  # Judge phase: coarse facing OK → wrist fine control
  python3 scripts/write_align_decision.py \\
    --session-dir log/real_robot/qa/20260804_200431 \\
    --step-idx 0 --phase judge --action coarse_ok \\
    --reason 'held pose: arm tip faces plant in fixed mono'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from align_judge import ALIGN_ACTIONS  # noqa: E402


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=path.parent, encoding='utf-8') as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=True)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def parse_deg_map(text: str) -> dict:
    """Parse 'joint1=25,joint5=-30' into a dict of floats."""
    out = {}
    text = (text or '').strip()
    if not text:
        return out
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f'expected name=value, got {part!r}')
        k, v = part.split('=', 1)
        out[k.strip()] = float(v.strip())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', required=True)
    parser.add_argument('--step-idx', type=int, required=True)
    parser.add_argument('--phase', choices=('command', 'judge'), default='command')
    parser.add_argument('--action', required=True, choices=list(ALIGN_ACTIONS))
    parser.add_argument('--reason', required=True)
    parser.add_argument('--confidence', type=float, default=0.8)
    parser.add_argument('--provider', default='agent')
    parser.add_argument(
        '--joints-deg', default='',
        help='Absolute targets: joint1=26,joint2=-12,joint3=15,joint5=-43')
    parser.add_argument(
        '--delta-deg', default='',
        help='Relative deltas: joint1=20,joint5=-25')
    args = parser.parse_args()

    session = Path(args.session_dir)
    if not session.is_dir():
        print(f'session dir missing: {session}', file=sys.stderr)
        return 2

    payload = {
        'action': args.action,
        'phase': args.phase,
        'reason': args.reason,
        'confidence': float(args.confidence),
        'provider': args.provider,
        'step_idx': int(args.step_idx),
        'session': session.name,
    }
    joints = parse_deg_map(args.joints_deg)
    deltas = parse_deg_map(args.delta_deg)
    if joints:
        payload['joints_deg'] = joints
    if deltas:
        payload['delta_deg'] = deltas
    if args.action == 'set_joints' and not joints and not deltas:
        print('set_joints requires --joints-deg or --delta-deg', file=sys.stderr)
        return 2

    out = session / 'align_decision.json'
    atomic_write_json(out, payload)
    print(
        f'wrote {out} action={args.action} phase={args.phase} step={args.step_idx}',
        flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
