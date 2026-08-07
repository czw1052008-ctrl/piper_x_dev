#!/usr/bin/env python3
"""Write align_decision.json from FSM align request/judge observation.

Uses align_judge.decide_action() — targets derived from plant_yaw, current joints,
and fine visibility in the request payload. No hardcoded joint angles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from align_judge import decide_action  # noqa: E402
from write_align_decision import atomic_write_json  # noqa: E402


def _request_candidates(session: Path, step_idx: int, phase: str) -> List[Path]:
    if phase == 'judge':
        tags = (
            f'align_{step_idx:02d}_judge.json',
            f'align_{step_idx:02d}_after_set_joints.json',
            f'align_{step_idx:02d}_request.json',
        )
    else:
        tags = (f'align_{step_idx:02d}_request.json',)
    return [session / t for t in tags]


def load_align_request(
    session: Path, step_idx: int, phase: str,
) -> Dict[str, object]:
    for path in _request_candidates(session, step_idx, phase):
        if not path.is_file():
            continue
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if not isinstance(payload.get('observation'), dict):
            raise ValueError(f'{path}: missing observation dict')
        return payload
    tried = ', '.join(str(p.name) for p in _request_candidates(session, step_idx, phase))
    raise FileNotFoundError(
        f'no align request for step={step_idx} phase={phase} in {session} (tried {tried})')


def infer_step_and_phase(session: Path) -> Optional[tuple[int, str]]:
    """Latest align_* file by mtime when step/phase not specified."""
    files = sorted(
        session.glob('align_*_request.json')
        + list(session.glob('align_*_judge.json'))
        + list(session.glob('align_*_after_set_joints.json')),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return None
    path = files[-1]
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    step_idx = int(payload.get('step_idx', 0))
    if path.name.endswith('_judge.json') or path.name.endswith('_after_set_joints.json'):
        phase = 'judge'
    else:
        phase = str(payload.get('phase', 'command')).strip().lower()
        if phase not in ('command', 'judge'):
            phase = 'command'
    return step_idx, phase


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', required=True)
    parser.add_argument('--step-idx', type=int, default=None)
    parser.add_argument('--phase', choices=('command', 'judge'), default=None)
    parser.add_argument('--provider', default='heuristic')
    parser.add_argument('--yaw-deadband-deg', type=float, default=8.0)
    parser.add_argument('--pitch-target-rad', type=float, default=-0.75)
    parser.add_argument('--fine-visible-conf', type=float, default=0.25)
    args = parser.parse_args()

    session = Path(args.session_dir)
    if not session.is_dir():
        print(f'session dir missing: {session}', file=sys.stderr)
        return 2

    if args.step_idx is None or args.phase is None:
        inferred = infer_step_and_phase(session)
        if inferred is None:
            print(f'no align request files in {session}', file=sys.stderr)
            return 2
        step_idx, phase = inferred
        if args.step_idx is not None:
            step_idx = args.step_idx
        if args.phase is not None:
            phase = args.phase
    else:
        step_idx, phase = args.step_idx, args.phase

    payload = load_align_request(session, step_idx, phase)
    obs = payload['observation']
    failed = payload.get('failed_actions') or []

    decision = decide_action(
        obs,
        failed_actions=failed,
        yaw_deadband_deg=args.yaw_deadband_deg,
        pitch_target_rad=args.pitch_target_rad,
        fine_visible_conf=args.fine_visible_conf,
        phase=phase,
    )

    out: Dict[str, object] = {
        'action': decision['action'],
        'phase': phase,
        'reason': str(decision.get('reason', 'decide_action from observation')),
        'confidence': 0.75,
        'provider': args.provider,
        'step_idx': step_idx,
        'session': session.name,
    }
    if 'joints_deg' in decision:
        out['joints_deg'] = decision['joints_deg']
    if 'delta_deg' in decision:
        out['delta_deg'] = decision['delta_deg']

    out_path = session / 'align_decision.json'
    atomic_write_json(out_path, out)
    joints = out.get('joints_deg', {})
    print(
        f'wrote {out_path} action={out["action"]} phase={phase} step={step_idx} '
        f'joints_deg={joints}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
