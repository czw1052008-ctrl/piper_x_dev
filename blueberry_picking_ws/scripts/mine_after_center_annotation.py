#!/usr/bin/env python3
"""Mine wrist RGB for after-center / cup-aim pose labeling (batch5).

Prioritizes frames where YOLO often misses after mid-range center:
  servo_*_after_wrist, near_handoff*_wrist, probe_tri*_wrist, fresh_lock*_wrist.
Default out: datasets/blueberry/images_batch5/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from collections import defaultdict
from glob import glob

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

SKIP_SUBSTR = (
    'sidebyside', 'depth', 'fine_viz', 'global_viz', 'fixed',
    'lock_viz', 'preview', 'plots', 'annotated', 'cup_overlay', 'cup_rel',
)

# Higher score = more useful for after-center miss labeling.
PRIORITY = (
    ('optical_fresh_lock_wrist', 100),
    ('near_handoff_pending_wrist', 95),
    ('probe_tri_contact_wrist', 90),
    ('servo_', 50),  # boosted further if after_wrist
    ('refine_fruit_lock_wrist', 40),
    ('lock_wait', 20),
)


def _is_raw_wrist(path: str) -> bool:
    name = os.path.basename(path).lower()
    if not name.endswith(('.png', '.jpg', '.jpeg')):
        return False
    if any(s in path.lower() for s in SKIP_SUBSTR):
        return False
    return 'wrist' in name and name.endswith('wrist.png')


def _priority(path: str) -> int:
    name = os.path.basename(path).lower()
    score = 0
    for key, w in PRIORITY:
        if key in name:
            score = max(score, w)
    if 'after_wrist' in name:
        score = max(score, 85)
    if 'motion_' in name:
        score = min(score, 30)  # mid-traj less useful
    return score


def _phash(bgr: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            bits |= 1 << i
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _session_of(path: str, qa_root: str) -> str:
    rel = os.path.relpath(path, qa_root)
    return rel.split(os.sep)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--qa-root', default=os.path.join(ROOT, 'log', 'real_robot', 'qa'))
    parser.add_argument(
        '--out',
        default=os.path.join(ROOT, 'datasets', 'blueberry', 'images_batch5'))
    parser.add_argument('--target', type=int, default=100)
    parser.add_argument('--max-per-session', type=int, default=12)
    parser.add_argument('--hamming-min', type=int, default=6)
    parser.add_argument(
        '--session-prefix', default='20260807',
        help='Only sessions starting with this (empty = all)')
    parser.add_argument(
        '--min-priority', type=int, default=40,
        help='Skip low-priority wrist frames (e.g. lock_wait)')
    args = parser.parse_args()

    qa_root = os.path.abspath(args.qa_root)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    for old in glob(os.path.join(out_dir, '*')):
        if os.path.isfile(old):
            os.remove(old)

    paths = [
        p for p in glob(os.path.join(qa_root, '**', '*'), recursive=True)
        if _is_raw_wrist(p)
    ]
    if args.session_prefix:
        paths = [
            p for p in paths
            if _session_of(p, qa_root).startswith(args.session_prefix)
        ]
    paths = [p for p in paths if _priority(p) >= args.min_priority]
    paths.sort(
        key=lambda p: (_priority(p), _session_of(p, qa_root), p),
        reverse=True,
    )

    kept: list[tuple[str, int]] = []
    per_sess: dict[str, int] = defaultdict(int)
    manifest = []

    for path in paths:
        if len(kept) >= args.target:
            break
        sess = _session_of(path, qa_root)
        if per_sess[sess] >= args.max_per_session:
            continue
        img = cv2.imread(path)
        if img is None or img.size == 0:
            continue
        h = _phash(img)
        if any(_hamming(h, kh) < args.hamming_min for _, kh in kept):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        digest = hashlib.sha1(path.encode()).hexdigest()[:8]
        out_name = f'b5_{sess}_{stem}_{digest}.png'
        out_path = os.path.join(out_dir, out_name)
        shutil.copy2(path, out_path)
        kept.append((out_path, h))
        per_sess[sess] += 1
        manifest.append({
            'out': out_name,
            'src': os.path.relpath(path, ROOT),
            'session': sess,
            'priority': _priority(path),
        })

    man_path = os.path.join(os.path.dirname(out_dir), 'batch5_manifest.txt')
    with open(man_path, 'w', encoding='utf-8') as f:
        f.write('# out_name\tsession\tpriority\tsrc\n')
        for m in manifest:
            f.write(
                f"{m['out']}\t{m['session']}\t{m['priority']}\t{m['src']}\n")

    print(f'[mine] wrote {len(kept)} images → {out_dir}')
    print(f'[mine] sessions: {dict(per_sess)}')
    print(f'[mine] manifest: {os.path.abspath(man_path)}')
    print('[mine] Next:')
    print(
        '  bash scripts/run_prelabel_blueberry.sh '
        '--images datasets/blueberry/images_batch5 '
        '--labels datasets/blueberry/labels_batch5')
    print(
        '  bash scripts/run_annotate_blueberry.sh '
        '--images datasets/blueberry/images_batch5 '
        '--labels datasets/blueberry/labels_batch5')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
