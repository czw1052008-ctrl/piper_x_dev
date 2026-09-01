#!/usr/bin/env python3
"""Pick a cluster from lock_region_request.json and write lock_region_decision.json.

Usage:
  # List clusters detected on fixed camera, pick best confidence:
  python3 scripts/wait_and_lock_cluster.py --qa-dir log/real_robot/qa --auto

  # Pick by index (from lock_region_request candidates):
  python3 scripts/wait_and_lock_cluster.py --session-dir log/real_robot/qa/20260831_120000 --index 1

  # Wait for newest session after /reach/cmd start:
  python3 scripts/wait_and_lock_cluster.py --qa-dir log/real_robot/qa --wait-start --auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent


def _find_latest_session(qa_dir: Path, *, max_age_s: float = 180.0) -> Optional[Path]:
    now = time.time()
    for d in sorted(qa_dir.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        req = d / 'lock_region_request.json'
        if req.is_file() and (now - req.stat().st_mtime) <= max_age_s:
            return d
    return None


def _wait_for_session(qa_dir: Path, timeout_s: float) -> Path:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        sess = _find_latest_session(qa_dir, max_age_s=timeout_s)
        if sess is not None:
            return sess
        time.sleep(0.5)
    raise TimeoutError(f'no lock_region_request.json within {timeout_s}s under {qa_dir}')


def _load_candidates(session: Path) -> List[dict]:
    req = session / 'lock_region_request.json'
    with open(req, encoding='utf-8') as f:
        data = json.load(f)
    return list(data.get('candidates') or [])


def _pick_index(candidates: List[dict], index: Optional[int], auto: bool) -> int:
    if not candidates:
        raise SystemExit('no cluster candidates in lock_region_request.json')
    if index is not None:
        if index < 0 or index >= len(candidates):
            raise SystemExit(f'index {index} out of range 0..{len(candidates)-1}')
        return index
    if auto:
        best = max(range(len(candidates)), key=lambda i: float(candidates[i].get('confidence', 0)))
        return best
    print('Candidates (fixed-mono cluster detections):')
    for c in candidates:
        xyz = c.get('xyz', [0, 0, 0])
        print(f"  [{c.get('index', '?')}] conf={c.get('confidence', 0):.2f} "
              f"xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})")
    raise SystemExit('Specify --index N or --auto')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qa-dir', default=str(ROOT / 'log' / 'real_robot' / 'qa'))
    parser.add_argument('--session-dir', default='')
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--auto', action='store_true', help='Pick highest-confidence cluster')
    parser.add_argument('--wait-start', action='store_true',
                        help='Wait for lock_region_request.json after reach start')
    parser.add_argument('--timeout-s', type=float, default=90.0)
    parser.add_argument('--reason', default='VLM cluster align: operator selected cluster')
    args = parser.parse_args()

    if args.session_dir:
        session = Path(args.session_dir)
    elif args.wait_start:
        session = _wait_for_session(Path(args.qa_dir), args.timeout_s)
    else:
        session = _find_latest_session(Path(args.qa_dir), max_age_s=args.timeout_s)
        if session is None:
            raise SystemExit(f'no recent session under {args.qa_dir} (use --wait-start)')

    if not (session / 'lock_region_request.json').is_file():
        raise SystemExit(f'missing {session}/lock_region_request.json')

    candidates = _load_candidates(session)
    idx = _pick_index(candidates, args.index, args.auto)
    c = candidates[idx]
    xyz = c.get('xyz', [0, 0, 0])

    out = {
        'action': 'lock_region',
        'index': idx,
        'reason': args.reason,
        'confidence': float(c.get('confidence', 0.8)),
        'provider': 'wait_and_lock_cluster',
        'session': session.name,
        'xyz': xyz,
    }
    path = session / 'lock_region_decision.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'[lock] session={session.name} index={idx} conf={out["confidence"]:.2f} '
          f'xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})')
    print(f'[lock] wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
