#!/usr/bin/env python3
"""Write LOCKING (pre-ALIGN) region decision for reach_fsm_node file mode.

Agent picks a COLLECTION REGION (plant/cluster) from fixed-mono candidates
in lock_region_request.json — not a single berry for contact.

Example:

  python3 scripts/write_lock_region_decision.py \\
    --session-dir log/real_robot/qa/20260805_120000 \\
    --index 0 \\
    --reason 'fixed mono: leftmost plant cluster in FOV'
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=path.parent, encoding='utf-8') as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=True)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', required=True)
    parser.add_argument('--index', type=int, required=True,
                        help='Candidate index from lock_region_request.json')
    parser.add_argument('--reason', required=True)
    parser.add_argument('--confidence', type=float, default=0.8)
    parser.add_argument('--provider', default='agent')
    args = parser.parse_args()

    session = Path(args.session_dir)
    if not session.is_dir():
        print(f'session dir missing: {session}', flush=True)
        return 2

    payload = {
        'action': 'lock_region',
        'index': int(args.index),
        'reason': args.reason,
        'confidence': float(args.confidence),
        'provider': args.provider,
        'session': session.name,
    }
    out = session / 'lock_region_decision.json'
    atomic_write_json(out, payload)
    print(f'wrote {out} action=lock_region index={args.index}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
