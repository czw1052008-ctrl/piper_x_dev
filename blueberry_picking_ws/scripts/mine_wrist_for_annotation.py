#!/usr/bin/env python3
"""Mine diverse raw wrist RGB frames from QA logs for YOLO annotation.

Prefer clean wrist RGB (no side-by-side / depth / fine_viz overlays).
Subsample across sessions + perceptual hash to avoid near-duplicates.
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
    'lock_viz', 'preview', 'plots',
)


def _is_candidate(path: str) -> bool:
    name = os.path.basename(path).lower()
    if not name.endswith(('.png', '.jpg', '.jpeg')):
        return False
    if any(s in path.lower() for s in SKIP_SUBSTR):
        return False
    # Prefer raw wrist RGB naming.
    ok_names = (
        'wrist.png', 'wrist_rgb', '_wrist.png', 'wrist_rgb_lock',
        'wrist_rgb_now', 'wrist_rgb_retry',
    )
    return any(k in name for k in ok_names) or name.endswith('wrist.png')


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


def _step_key(path: str) -> int:
    m = re.search(r'servo_(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-root', default=os.path.join(ROOT, 'log', 'real_robot', 'qa'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'datasets', 'blueberry', 'images_batch4'))
    parser.add_argument('--target', type=int, default=80)
    parser.add_argument('--max-per-session', type=int, default=8)
    parser.add_argument('--hamming-min', type=int, default=8,
                        help='Min perceptual-hash distance vs already kept')
    parser.add_argument('--prefer-recent', action='store_true', default=True)
    args = parser.parse_args()

    qa_root = os.path.abspath(args.qa_root)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    # Clear previous batch4 images (labels handled separately).
    for old in glob(os.path.join(out_dir, '*')):
        if os.path.isfile(old):
            os.remove(old)

    paths = [p for p in glob(os.path.join(qa_root, '**', '*'), recursive=True) if _is_candidate(p)]
    # Prefer recent sessions by name sort (YYYYMMDD_...).
    paths.sort(key=lambda p: (_session_of(p, qa_root), _step_key(p), p), reverse=True)

    by_sess: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_sess[_session_of(p, qa_root)].append(p)

    # Round-robin sessions so far/near/entry all appear.
    sessions = sorted(by_sess.keys(), reverse=True)
    kept: list[tuple[str, int]] = []  # (path, phash)
    per_sess: dict[str, int] = defaultdict(int)

    # First pass: stride within each session.
    candidates: list[str] = []
    for sess in sessions:
        items = by_sess[sess]
        if len(items) <= args.max_per_session:
            candidates.extend(items)
        else:
            # Even spacing + always keep first/last.
            idxs = sorted(set(
                [0, len(items) - 1]
                + [int(round(i * (len(items) - 1) / (args.max_per_session - 1)))
                   for i in range(args.max_per_session)]
            ))
            candidates.extend(items[i] for i in idxs if 0 <= i < len(items))

    for path in candidates:
        if len(kept) >= args.target:
            break
        sess = _session_of(path, qa_root)
        if per_sess[sess] >= args.max_per_session:
            continue
        bgr = cv2.imread(path)
        if bgr is None or bgr.size == 0:
            continue
        if min(bgr.shape[:2]) < 200:
            continue
        h = _phash(bgr)
        if any(_hamming(h, kh) < args.hamming_min for _, kh in kept):
            continue
        kept.append((path, h))
        per_sess[sess] += 1

    # Second pass: fill if under target (relax hash a bit).
    if len(kept) < args.target:
        for path in paths:
            if len(kept) >= args.target:
                break
            sess = _session_of(path, qa_root)
            if per_sess[sess] >= args.max_per_session + 2:
                continue
            if any(path == kp for kp, _ in kept):
                continue
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            h = _phash(bgr)
            if any(_hamming(h, kh) < max(4, args.hamming_min - 3) for _, kh in kept):
                continue
            kept.append((path, h))
            per_sess[sess] += 1

    manifest = []
    for i, (path, _) in enumerate(kept, 1):
        sess = _session_of(path, qa_root)
        stem = os.path.splitext(os.path.basename(path))[0]
        # Stable unique name.
        digest = hashlib.md5(path.encode()).hexdigest()[:6]
        name = f'b4_{sess}_{stem}_{digest}.png'
        dst = os.path.join(out_dir, name)
        shutil.copy2(path, dst)
        manifest.append(f'{name}\t{path}')

    man_path = os.path.join(out_dir, 'MANIFEST.tsv')
    with open(man_path, 'w', encoding='utf-8') as f:
        f.write('name\tsource\n')
        f.write('\n'.join(manifest) + '\n')

    print(f'[mine] candidates={len(paths)} sessions={len(sessions)} kept={len(kept)}')
    print(f'[mine] out={out_dir}')
    print(f'[mine] per-session: {dict(sorted(per_sess.items(), key=lambda x: -x[1])[:12])}...')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
