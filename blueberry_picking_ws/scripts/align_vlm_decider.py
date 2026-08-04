#!/usr/bin/env python3
"""Write ALIGNING decisions for reach_fsm_node file-judge mode.

Providers:
  heuristic — estimate absolute joint targets (set_joints) / coarse_ok after hold
  agent     — Cursor agent / human writes align_decision.json (write_align_decision.py)
  qwen      — reserved until a local/remote Qwen2.5-VL backend is wired
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from align_judge import decide_action  # noqa: E402


def latest_session(qa_dir: Path) -> Optional[Path]:
    """Prefer FSM session dirs (YYYYMMDD_HHMMSS), ignore qa_before/qa_after snaps."""
    sessions = [
        p for p in qa_dir.iterdir()
        if p.is_dir() and p.name[:8].isdigit() and '_' in p.name
    ]
    if not sessions:
        sessions = [p for p in qa_dir.iterdir() if p.is_dir()]
    if not sessions:
        return None
    return max(sessions, key=lambda p: p.name)


def latest_request(
    session_dir: Path, processed: Set[Tuple[int, str]],
) -> Optional[Path]:
    files = sorted(session_dir.glob('align_*_request.json')) + sorted(
        session_dir.glob('align_*_judge.json'))
    for path in reversed(files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            step_idx = int(payload.get('step_idx', -1))
            phase = str(payload.get('phase', 'command')).strip().lower()
            if path.name.endswith('_judge.json'):
                phase = 'judge'
            elif path.name.endswith('_request.json'):
                phase = phase if phase in ('command', 'judge') else 'command'
        except Exception:
            continue
        if (step_idx, phase) in processed:
            continue
        return path
    return None


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=path.parent, encoding='utf-8') as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=True)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def pending_paths(session_dir: Path, step_idx: int, phase: str) -> Dict[str, str]:
    tags = (
        [f'align_{step_idx:02d}_judge', f'align_{step_idx:02d}_after_set_joints',
         f'align_{step_idx:02d}_request']
        if phase == 'judge'
        else [f'align_{step_idx:02d}_request', f'align_{step_idx:02d}_before_set_joints']
    )
    out: Dict[str, str] = {}
    for tag in tags:
        for key in ('fixed', 'wrist', 'global_viz', 'fine_viz'):
            if key in out:
                continue
            p = session_dir / f'{tag}_{key}.png'
            if p.exists():
                out[key] = str(p)
        if len(out) >= 2:
            break
    return out


AGENT_HINT_COMMAND = (
    'Estimate HOW MUCH to move from fixed mono (plant vs arm tip). '
    'Write action=set_joints with --joints-deg absolute targets '
    '(joint1/joint2/joint3/joint5 degrees) or --delta-deg. '
    'If already roughly facing plant: action=coarse_ok. '
    'Do NOT use fixed-step whole_arm_* oscillation.'
)
AGENT_HINT_JUDGE = (
    'Pose is HELD after the position move. Read fixed + wrist. '
    'If arm roughly faces the plant: action=coarse_ok (enter wrist fine control). '
    'If not: action=set_joints with a new estimated joints_deg/delta_deg.'
)


def run(args: argparse.Namespace) -> int:
    qa_dir = Path(args.qa_dir)
    qa_dir.mkdir(parents=True, exist_ok=True)
    last_signature = None
    processed: Set[Tuple[int, str]] = set()
    print(
        f'[align_vlm_decider] watching qa_dir={qa_dir} provider={args.provider}',
        flush=True,
    )
    while True:
        session_dir = Path(args.session_dir) if args.session_dir else latest_session(qa_dir)
        if session_dir is None or not session_dir.exists():
            time.sleep(args.poll_s)
            continue
        req = latest_request(session_dir, processed)
        if req is None:
            time.sleep(args.poll_s)
            continue
        sig = (str(req), req.stat().st_mtime_ns)
        if sig == last_signature:
            time.sleep(args.poll_s)
            continue
        try:
            with open(req, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as exc:
            print(f'[align_vlm_decider] failed to read {req}: {exc}', flush=True)
            time.sleep(args.poll_s)
            continue
        obs = payload.get('observation')
        if not isinstance(obs, dict):
            print(f'[align_vlm_decider] request missing observation: {req}', flush=True)
            time.sleep(args.poll_s)
            continue
        step_idx = int(payload.get('step_idx', -1))
        phase = str(payload.get('phase', 'command')).strip().lower()
        if req.name.endswith('_judge.json'):
            phase = 'judge'
        failed_actions = payload.get('failed_actions') or []
        imgs = pending_paths(session_dir, step_idx, phase)

        if args.provider == 'agent':
            pending = {
                'status': 'waiting_for_agent',
                'step_idx': step_idx,
                'phase': phase,
                'session': session_dir.name,
                'request_file': req.name,
                'images': imgs,
                'failed_actions': failed_actions,
                'observation': obs,
                'hint': AGENT_HINT_JUDGE if phase == 'judge' else AGENT_HINT_COMMAND,
            }
            pending_path = session_dir / 'align_pending.json'
            atomic_write_json(pending_path, pending)
            print(
                f'[align_vlm_decider] AGENT pending step={step_idx} phase={phase} '
                f'fixed={imgs.get("fixed", "?")} — waiting for align_decision.json',
                flush=True,
            )
            decision_path = session_dir / 'align_decision.json'
            deadline = time.time() + args.agent_wait_s
            while time.time() < deadline:
                if decision_path.exists():
                    try:
                        with open(decision_path, 'r', encoding='utf-8') as f:
                            decision = json.load(f)
                        if int(decision.get('step_idx', -1)) != step_idx:
                            time.sleep(args.poll_s)
                            continue
                        dec_phase = str(decision.get('phase', phase)).strip().lower()
                        if dec_phase not in ('', phase) and dec_phase != phase:
                            time.sleep(args.poll_s)
                            continue
                        processed.add((step_idx, phase))
                        print(
                            f'[align_vlm_decider] agent decision step={step_idx} '
                            f'phase={phase} action={decision.get("action")}',
                            flush=True,
                        )
                        last_signature = None
                        break
                    except Exception:
                        pass
                time.sleep(args.poll_s)
            else:
                print(
                    f'[align_vlm_decider] still waiting for agent step={step_idx} phase={phase}',
                    flush=True,
                )
            if args.once:
                return 0
            time.sleep(args.poll_s)
            continue

        if args.provider == 'qwen':
            raise ValueError(
                'provider=qwen is reserved for Qwen2.5-VL; use provider=agent '
                '(Cursor agent reads images) until a local/remote Qwen backend is configured'
            )

        if args.provider != 'heuristic':
            raise ValueError(f'unsupported provider: {args.provider}')

        decision = decide_action(
            obs,
            failed_actions=failed_actions,
            yaw_deadband_deg=args.yaw_deadband_deg,
            pitch_target_rad=args.pitch_target_rad,
            fine_visible_conf=args.fine_visible_conf,
            ee_angle_trigger_deg=args.ee_angle_trigger_deg,
            phase=phase,
        )
        decision.update({
            'provider': args.provider,
            'step_idx': step_idx,
            'phase': phase,
            'request_file': req.name,
            'session': session_dir.name,
            'images': imgs,
        })
        out = session_dir / 'align_decision.json'
        atomic_write_json(out, decision)
        processed.add((step_idx, phase))
        print(
            f'[align_vlm_decider] wrote {out.name} step={decision["step_idx"]} '
            f'phase={phase} action={decision["action"]} reason={decision["reason"]}',
            flush=True,
        )
        last_signature = sig
        if args.once:
            return 0
        time.sleep(args.poll_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'log', 'real_robot', 'qa'))
    parser.add_argument('--session-dir', default='',
                        help='Optional explicit QA session directory; default watches latest session')
    parser.add_argument(
        '--provider', choices=('heuristic', 'agent', 'qwen'), default='heuristic')
    parser.add_argument('--poll-s', type=float, default=0.3)
    parser.add_argument('--agent-wait-s', type=float, default=120.0,
                        help='How long agent provider waits for align_decision.json per step')
    parser.add_argument('--once', action='store_true', help='Process one request then exit')
    parser.add_argument('--yaw-deadband-deg', type=float, default=8.0)
    parser.add_argument('--pitch-target-rad', type=float, default=-0.75)
    parser.add_argument('--fine-visible-conf', type=float, default=0.20)
    parser.add_argument('--ee-angle-trigger-deg', type=float, default=20.0)
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
