#!/usr/bin/env python3
"""Write manual wrist target bbox for calib / fine_detector IoU pin.

Example (pixels in entry_label_wrist.png, origin top-left):

  python3 scripts/write_target_bbox_decision.py \\
    --session-dir log/real_robot/qa/20260812_150532 \\
    --bbox 310 228 358 276 \\
    --reason 'user: contact berry lower cluster'
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-dir', required=True, help='QA session with entry_label_wrist.png')
    p.add_argument('--bbox', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'),
                   required=True, help='BBox in wrist RGB pixels (inclusive x0,y0; exclusive x1,y1 ok)')
    p.add_argument('--reason', default='user_manual')
    p.add_argument('--also-global', action='store_true',
                   help='Also write log/real_robot/target_bbox_lock.json for fine_detector')
    args = p.parse_args()

    session = Path(args.session_dir)
    if not session.is_dir():
        print(f'missing session: {session}')
        return 2
    x0, y0, x1, y1 = [int(v) for v in args.bbox]
    if x1 <= x0 or y1 <= y0:
        print('invalid bbox: need x1>x0 and y1>y0')
        return 2
    cu = 0.5 * (x0 + x1)
    cv = 0.5 * (y0 + y1)
    payload = {
        'action': 'lock_bbox',
        'bbox_xyxy': [x0, y0, x1, y1],
        'center_uv': [cu, cv],
        'reason': args.reason,
        'session': session.name,
        'provider': 'user',
    }
    out = session / 'target_bbox_decision.json'
    atomic_write_json(out, payload)
    print(f'wrote {out} bbox=[{x0},{y0},{x1},{y1}] center=({cu:.1f},{cv:.1f})')

    if args.also_global:
        root = session.parent.parent
        global_path = root / 'target_bbox_lock.json'
        atomic_write_json(global_path, payload)
        print(f'wrote {global_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
