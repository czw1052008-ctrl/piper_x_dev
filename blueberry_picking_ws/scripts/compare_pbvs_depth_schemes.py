#!/usr/bin/env python3
"""Compare depth schemes on a PBVS QA session (offline).

Reads ``pbvs_stream.jsonl`` depth fields when present; always reports entry
``refine_fruit_lock.json`` depth vs mono vs legacy fusion.

Usage:
  python3 scripts/compare_pbvs_depth_schemes.py log/real_robot/qa/20260811_155612
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pbvs_qa_recorder import load_stream, render_pbvs_replay_html  # noqa: E402


def _fused_z(z_depth: float, z_mono: float) -> float:
    return 0.65 * z_depth + 0.35 * z_mono


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    vals = sorted(vals)
    n = len(vals)
    return {
        'n': n,
        'min': vals[0],
        'max': vals[-1],
        'mean': sum(vals) / n,
        'spread_mm': (vals[-1] - vals[0]) * 1000.0,
    }


def analyze_session(session_dir: str) -> int:
    session_dir = os.path.abspath(session_dir)
    rows = load_stream(os.path.join(session_dir, 'pbvs_stream.jsonl'))
    lock_path = os.path.join(session_dir, 'refine_fruit_lock.json')
    lock: Dict[str, Any] = {}
    if os.path.isfile(lock_path):
        with open(lock_path, encoding='utf-8') as f:
            lock = json.load(f)

    print(f'=== PBVS depth scheme analysis: {os.path.basename(session_dir)} ===')
    print(f'frames in jsonl: {len(rows)}')

    if lock:
        zd = lock.get('z_depth_m')
        zm = lock.get('z_mono_m')
        if zd and zm:
            print('\n--- entry lock (refine_fruit_lock.json) ---')
            print(f'  z_depth   = {float(zd)*1000:.1f} mm')
            print(f'  z_mono    = {float(zm)*1000:.1f} mm  (delta vs depth {(float(zm)-float(zd))*1000:+.1f} mm)')
            print(f'  fused65/35= {_fused_z(float(zd), float(zm))*1000:.1f} mm')
            print(f'  depth_raw = {float(zd)*1000:.1f} mm  (new fine_detector default)')
            print(f'  berry_base Z = {float(lock["berry_base"][2])*1000:.1f} mm')

    has_depth = any(r.get('z_depth_m') is not None for r in rows)
    if not has_depth:
        print('\n[!] jsonl has no per-frame z_depth_m — re-run PBVS after recorder update.')
    else:
        print('\n--- per-frame SERVO/APPROACH (mm) ---')
        for key in ('z_depth_m', 'z_mono_m', 'z_used_m', 'berry_z_m', 'cup_dist_m'):
            vals = [
                float(r[key]) for r in rows
                if r.get(key) is not None and r.get('pbvs_state') in ('SERVO', 'APPROACH')
            ]
            st = _stats(vals)
            if st:
                print(
                    f'  {key:12s} n={int(st["n"]):4d}  '
                    f'min={st["min"]*1000:6.1f}  max={st["max"]*1000:6.1f}  '
                    f'spread={st["spread_mm"]:5.1f} mm')

        # Handoff frame
        handoff = next((r for r in rows if r.get('pbvs_state') == 'APPROACH'), None)
        if handoff:
            print('\n--- first APPROACH frame ---')
            print(f'  frame={handoff.get("frame")} cup={handoff.get("cup_dist_m")}')
            for k in ('z_depth_m', 'z_mono_m', 'z_used_m', 'berry_z_m', 'depth_mode'):
                v = handoff.get(k)
                if v is not None:
                    unit = ' mm' if k.endswith('_m') else ''
                    val = float(v) * 1000 if k.endswith('_m') else v
                    print(f'  {k}={val}{unit}')

        coast_n = sum(1 for r in rows if 'coast' in str(r.get('depth_mode', '')))
        print(f'\n--- coast frames: {coast_n} / {len(rows)} ---')

    html = os.path.join(session_dir, 'pbvs_replay.html')
    render_pbvs_replay_html(session_dir, html)
    print(f'\nregenerated: {html}')
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description='Compare PBVS depth schemes on QA session')
    ap.add_argument('session_dir', help='QA session directory')
    args = ap.parse_args()
    raise SystemExit(analyze_session(args.session_dir))


if __name__ == '__main__':
    main()
