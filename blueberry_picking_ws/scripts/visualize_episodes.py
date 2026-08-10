#!/usr/bin/env python3
"""Episode visualiser for BC alignment training data.

Typical workflow after each collection session
-----------------------------------------------
# After running reach_fsm_node with --collect-data, inspect the latest episode:
    python scripts/visualize_episodes.py --data-dir data/align_episodes --latest
    # → opens data/align_episodes/viz/episode_NNNNN.html  (open in any browser)

# List all collected episodes with stats:
    python scripts/visualize_episodes.py --data-dir data/align_episodes

# Render HTML for every episode + summary page:
    python scripts/visualize_episodes.py --data-dir data/align_episodes --all

# Render a specific episode with an optional video:
    python scripts/visualize_episodes.py --data-dir data/align_episodes --episode 3 --video

Output layout
-------------
  data/align_episodes/viz/summary.html          dataset-level stats table
  data/align_episodes/viz/episode_NNNNN.html    per-episode step-by-step report
  data/align_episodes/viz/episode_NNNNN.mp4     (--video only, requires opencv)

Per-step HTML report shows
--------------------------
  • Global camera image with:
      - green circle = locked target (fixed_plant_u/v)
      - red cross    = EE projection (fixed_ee_u/v)
      - LOCKED / NO PROJ badge
  • Wrist camera image with:
      - green circle = detected berry (fine_u/v) when fine_visible=1
      - confidence label
      - NOT VISIBLE overlay when not detected
  • Numerical obs table (angles, pixel errors, distances)
  • Action block (type, joint targets, reason text)
  • Per-step status badges: LOCKED / DETECTED / phase
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
from typing import Dict, List, Optional

# Obs keys shown in the numeric table (excludes raw pixel coords used only for drawing).
_OBS_TABLE_KEYS = (
    'joint1_deg', 'joint2_deg', 'joint3_deg', 'joint5_deg',
    'yaw_error_deg', 'pitch_error_rad',
    'ee_target_angle_deg', 'horiz_dist',
    'fine_visible', 'fine_confidence',
    'fine_du', 'fine_dv',
    'fixed_dx_px', 'fixed_dy_px',
    'fixed_has_view',
)

# ---------------------------------------------------------------------------
# Image annotation (opencv)
# ---------------------------------------------------------------------------

def _annotate_wrist(img, obs: Dict) -> 'np.ndarray':
    """Draw fine-detector overlay onto wrist image (in-place copy)."""
    import cv2
    import numpy as np
    img = img.copy()
    h, w = img.shape[:2]

    visible = float(obs.get('fine_visible', 0)) >= 1.0
    fu = obs.get('fine_u')
    fv = obs.get('fine_v')
    # Fallback: reconstruct from centre + delta.
    if fu is None:
        cx = obs.get('fine_cx', w / 2.0)
        fu = cx + obs.get('fine_du', 0.0)
    if fv is None:
        cy = obs.get('fine_cy', h / 2.0)
        fv = cy + obs.get('fine_dv', 0.0)

    # Image-centre crosshair.
    cv2.drawMarker(img, (w // 2, h // 2), (80, 80, 200),
                   cv2.MARKER_CROSS, 18, 1)

    if visible:
        cx_det, cy_det = int(fu), int(fv)
        conf = float(obs.get('fine_confidence', 0.0))
        cv2.circle(img, (cx_det, cy_det), 22, (0, 220, 0), 2)
        cv2.drawMarker(img, (cx_det, cy_det), (0, 220, 0),
                       cv2.MARKER_CROSS, 12, 1)
        cv2.putText(img, f'{conf:.2f}', (cx_det + 6, cy_det - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1)
        _badge(img, 'DETECTED', (4, 4), (0, 180, 0))
    else:
        _badge(img, 'NOT VISIBLE', (4, 4), (0, 60, 200))

    return img


def _annotate_global(img, obs: Dict) -> 'np.ndarray':
    """Draw locked-target + EE projection onto global image."""
    import cv2
    import numpy as np
    img = img.copy()

    has_proj = float(obs.get('fixed_has_view', 0)) >= 0.5
    pu = obs.get('fixed_plant_u')
    pv = obs.get('fixed_plant_v')
    eu = obs.get('fixed_ee_u')
    ev = obs.get('fixed_ee_v')

    if has_proj and pu is not None and not _isnan(pu):
        # Target circle (green = locked; position comes from lock)
        cv2.circle(img, (int(pu), int(pv)), 20, (0, 220, 0), 2)
        cv2.putText(img, 'TARGET', (int(pu) + 6, int(pv) - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)

    if has_proj and eu is not None and not _isnan(eu):
        # EE cross (red)
        cv2.drawMarker(img, (int(eu), int(ev)), (60, 60, 220),
                       cv2.MARKER_CROSS, 22, 2)
        cv2.putText(img, 'EE', (int(eu) + 8, int(ev) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 220), 1)

    label = 'LOCKED' if has_proj else 'NO PROJ'
    color = (0, 160, 0) if has_proj else (0, 80, 160)
    _badge(img, label, (4, 4), color)
    return img


def _badge(img, text: str, xy, color) -> None:
    import cv2
    x, y = xy
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x, y), (x + tw + 6, y + th + 6), color, -1)
    cv2.putText(img, text, (x + 3, y + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _img_to_b64(path: str, obs: Dict = None, mode: str = 'global') -> str:
    """Load image, draw overlays, return base64 data URI."""
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return ''
        if obs is not None:
            if mode == 'wrist':
                img = _annotate_wrist(img, obs)
            else:
                img = _annotate_global(img, obs)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()
        return f'data:image/jpeg;base64,{b64}'
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:monospace;background:#111;color:#ccc;margin:16px;line-height:1.4}
h1,h2,h3{color:#61afef}
a{color:#56b6c2}
table{border-collapse:collapse;width:100%;margin-bottom:16px}
th{background:#1e1e1e;color:#61afef;padding:5px 9px;text-align:left;border-bottom:2px solid #333}
td{padding:3px 7px;border-bottom:1px solid #222;font-size:11px;vertical-align:top}
tr:hover td{background:#1a1a1a}
.ok{color:#98c379}.fail{color:#e06c75}.warn{color:#e5c07b}
.step{display:flex;gap:10px;margin-bottom:18px;padding:10px;
       border:1px solid #2a2a2a;border-radius:4px;background:#151515}
.imgs{display:flex;gap:6px;flex-shrink:0}
.imgs img{width:260px;height:auto;border:1px solid #333;border-radius:2px}
.img-lbl{font-size:10px;color:#666;text-align:center;margin-top:2px}
.right{flex:1;min-width:0}
.step-hdr{font-size:13px;color:#d19a66;font-weight:bold;margin-bottom:6px}
.badges{display:flex;gap:4px;margin-bottom:6px;flex-wrap:wrap}
.badge{font-size:10px;padding:2px 7px;border-radius:10px;font-weight:bold}
.b-locked{background:#1a4a1a;color:#98c379;border:1px solid #98c379}
.b-detected{background:#1a3a1a;color:#56d364;border:1px solid #56d364}
.b-nodet{background:#3a1a1a;color:#e06c75;border:1px solid #e06c75}
.b-phase{background:#1a2a3a;color:#61afef;border:1px solid #61afef}
.obs-tbl{font-size:10.5px;width:100%;margin-bottom:6px}
.obs-tbl td:first-child{color:#abb2bf;width:55%}
.obs-tbl td:last-child{color:#e5c07b;text-align:right}
.action{background:#1a1a2a;border:1px solid #2a2a3a;border-radius:3px;
         padding:6px 8px;font-size:11px;color:#98c379;white-space:pre-wrap;margin-top:4px}
"""

_HEAD = f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">' \
        f'<style>{_CSS}</style></head><body>\n'
_FOOT = '</body></html>\n'


def _obs_table(obs: Dict) -> str:
    rows = []
    for k in _OBS_TABLE_KEYS:
        v = obs.get(k)
        if v is None:
            continue
        rows.append(f'<tr><td>{k}</td><td>{v:.3f}</td></tr>')
    return f'<table class="obs-tbl">{"".join(rows)}</table>'


def _action_block(action: Dict) -> str:
    lines = []
    for k in ('action', 'source', 'phase'):
        if k in action:
            lines.append(f'{k}: {action[k]}')
    jd = action.get('joints_deg') or action.get('delta_deg')
    if isinstance(jd, dict):
        lines.append('joints: ' + '  '.join(
            f'{k}={v:.1f}°' for k, v in sorted(jd.items())))
    reason = action.get('reason', '')
    if reason:
        lines.append(f'reason: {reason}')
    return f'<div class="action">{"chr(10)".join(lines)}</div>'.replace(
        'chr(10)', '\n')


def _step_badges(obs: Dict, action: Dict) -> str:
    parts = []
    parts.append(f'<span class="badge b-locked">LOCKED</span>')
    visible = float(obs.get('fine_visible', 0)) >= 1.0
    if visible:
        parts.append('<span class="badge b-detected">WRIST DETECTED</span>')
    else:
        parts.append('<span class="badge b-nodet">WRIST NOT VISIBLE</span>')
    has_proj = float(obs.get('fixed_has_view', 0)) >= 0.5
    if has_proj:
        parts.append('<span class="badge b-detected">GLOBAL PROJ</span>')
    phase = action.get('phase', action.get('action', ''))
    if phase:
        parts.append(f'<span class="badge b-phase">{phase.upper()}</span>')
    return f'<div class="badges">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Per-episode HTML
# ---------------------------------------------------------------------------

def render_episode_html(ep: Dict, viz_dir: str) -> str:
    meta   = ep['meta']
    steps  = ep['steps']
    ep_dir = ep['ep_dir']
    ep_id  = meta.get('episode_id', '?')

    ok_cls   = 'ok' if meta.get('success') else 'fail'
    ok_str   = 'SUCCESS ✓' if meta.get('success') else 'FAIL ✗'
    plant    = meta.get('plant_xyz')
    plant_s  = (f'[{plant[0]:.3f}, {plant[1]:.3f}, {plant[2]:.3f}] m'
                if plant else 'unknown')

    html = [_HEAD, f'<h1>Episode {ep_id} &nbsp;'
            f'<span class="{ok_cls}">{ok_str}</span></h1>']

    # Metadata summary
    html.append('<table>')
    for k, v in [('teacher', meta.get('teacher', '?')),
                 ('n_steps', meta.get('n_steps', '?')),
                 ('duration', f'{meta.get("duration_s", 0):.1f} s'),
                 ('plant_xyz (locked)', plant_s)]:
        html.append(f'<tr><td>{k}</td><td>{v}</td></tr>')
    html.append('</table>')

    html.append(f'<h2>Steps ({len(steps)})</h2>')

    for step in steps:
        idx    = step['step_idx']
        obs    = step.get('obs', {})
        action = step.get('action', {})

        g_path = os.path.join(ep_dir, f'step_{idx:03d}_global.jpg')
        w_path = os.path.join(ep_dir, f'step_{idx:03d}_wrist.jpg')
        g_b64  = _img_to_b64(g_path, obs, 'global')
        w_b64  = _img_to_b64(w_path, obs, 'wrist')
        g_tag  = f'<img src="{g_b64}">' if g_b64 else '<i style="color:#666">no img</i>'
        w_tag  = f'<img src="{w_b64}">' if w_b64 else '<i style="color:#666">no img</i>'

        act_str = action.get('action', '?')

        html.append('<div class="step">')
        html.append(f'  <div class="imgs">'
                    f'    <div><div class="img-lbl">global</div>{g_tag}</div>'
                    f'    <div><div class="img-lbl">wrist</div>{w_tag}</div>'
                    f'  </div>')
        html.append('  <div class="right">')
        html.append(f'    <div class="step-hdr">Step {idx} — {act_str}</div>')
        html.append(_step_badges(obs, action))
        html.append(_obs_table(obs))
        html.append(_action_block(action))
        html.append('  </div>')
        html.append('</div>')

    html.append(_FOOT)

    os.makedirs(viz_dir, exist_ok=True)
    out_path = os.path.join(viz_dir, f'episode_{ep_id}.html')
    with open(out_path, 'w') as f:
        f.write('\n'.join(html))
    return out_path


# ---------------------------------------------------------------------------
# Per-episode video (optional)
# ---------------------------------------------------------------------------

def render_episode_video(ep: Dict, viz_dir: str, fps: float = 1.5) -> str:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print('[warn] opencv not available — skipping video')
        return ''

    meta   = ep['meta']
    steps  = ep['steps']
    ep_dir = ep['ep_dir']
    ep_id  = meta.get('episode_id', '?')

    W, H = 700, 280
    os.makedirs(viz_dir, exist_ok=True)
    out_path = os.path.join(viz_dir, f'episode_{ep_id}.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    for step in steps:
        idx    = step['step_idx']
        obs    = step.get('obs', {})
        action = step.get('action', {})

        frame = np.zeros((H, W, 3), dtype=np.uint8)

        for col, (fname, mode) in enumerate([
            (f'step_{idx:03d}_global.jpg', 'global'),
            (f'step_{idx:03d}_wrist.jpg',  'wrist'),
        ]):
            img = cv2.imread(os.path.join(ep_dir, fname))
            if img is not None:
                if mode == 'wrist':
                    img = _annotate_wrist(img, obs)
                else:
                    img = _annotate_global(img, obs)
                img = cv2.resize(img, (240, 240))
                x0 = col * 248 + 4
                frame[20:260, x0:x0 + 240] = img

        # Right info panel
        px = 510
        act_str = action.get('action', '?')
        src_str = action.get('source', '')
        cv2.putText(frame, f'Step {idx}', (px, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 180, 100), 1)
        cv2.putText(frame, f'{act_str} ({src_str})', (px, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 220, 100), 1)
        y = 80
        for k in ('yaw_error_deg', 'pitch_error_rad', 'fine_visible',
                  'fine_du', 'fine_dv', 'fixed_dx_px', 'horiz_dist'):
            v = obs.get(k)
            if v is not None:
                cv2.putText(frame, f'{k[:16]}: {v:.2f}', (px, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 170), 1)
                y += 17

        # Column labels
        for col, lbl in enumerate(['global', 'wrist']):
            cv2.putText(frame, lbl, (col * 248 + 90, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        writer.write(frame)

    writer.release()
    return out_path


# ---------------------------------------------------------------------------
# Summary HTML
# ---------------------------------------------------------------------------

def render_summary_html(episodes: List[Dict], viz_dir: str) -> str:
    n_total   = len(episodes)
    n_success = sum(1 for e in episodes if e['meta'].get('success'))
    teachers: Dict[str, int] = {}
    for e in episodes:
        t = e['meta'].get('teacher', 'unknown')
        teachers[t] = teachers.get(t, 0) + 1
    step_counts = [e['meta'].get('n_steps', 0) for e in episodes if e['meta'].get('success')]
    avg_steps   = sum(step_counts) / max(1, len(step_counts))

    html = [_HEAD, '<h1>Alignment Dataset — Summary</h1>', '<table>']
    for label, val in [
        ('Total episodes', n_total),
        ('Successful', f'<span class="ok">{n_success}</span>'),
        ('Failed', f'<span class="fail">{n_total - n_success}</span>'),
        ('Success rate', f'{n_success / max(1, n_total) * 100:.1f}%'),
        ('Avg steps (success)', f'{avg_steps:.1f}'),
        *[(f'teacher={t}', c) for t, c in sorted(teachers.items())],
    ]:
        html.append(f'<tr><td>{label}</td><td>{val}</td></tr>')
    html.append('</table>')

    # Obs statistics from all successful steps
    from collections import defaultdict
    vals: Dict[str, list] = defaultdict(list)
    for ep in episodes:
        if not ep['meta'].get('success'):
            continue
        for step in ep['steps']:
            for k in _OBS_TABLE_KEYS:
                v = step.get('obs', {}).get(k)
                if v is not None:
                    try:
                        vals[k].append(float(v))
                    except (ValueError, TypeError):
                        pass

    if vals:
        html.append('<h2>Obs statistics (successful steps)</h2><table>')
        html.append('<tr><th>Key</th><th>min</th><th>mean</th>'
                    '<th>max</th><th>std</th><th>n</th></tr>')
        for k in _OBS_TABLE_KEYS:
            v = vals.get(k)
            if not v:
                continue
            mn   = min(v)
            mx   = max(v)
            mean = sum(v) / len(v)
            std  = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
            html.append(f'<tr><td>{k}</td><td>{mn:.2f}</td><td>{mean:.2f}</td>'
                        f'<td>{mx:.2f}</td><td>{std:.2f}</td><td>{len(v)}</td></tr>')
        html.append('</table>')

    # Episode index
    html.append('<h2>Episode index</h2><table>')
    html.append('<tr><th>ID</th><th>Result</th><th>Steps</th>'
                '<th>Duration</th><th>Teacher</th><th>Report</th></tr>')
    for ep in reversed(episodes):   # newest first
        m   = ep['meta']
        eid = m.get('episode_id', '?')
        ok  = m.get('success', False)
        html.append(
            f'<tr>'
            f'<td>{eid}</td>'
            f'<td class="{"ok" if ok else "fail"}">{"SUCCESS" if ok else "FAIL"}</td>'
            f'<td>{m.get("n_steps","?")}</td>'
            f'<td>{m.get("duration_s",0):.1f}s</td>'
            f'<td>{m.get("teacher","?")}</td>'
            f'<td><a href="episode_{eid}.html">view</a></td>'
            f'</tr>'
        )
    html.append('</table>' + _FOOT)

    os.makedirs(viz_dir, exist_ok=True)
    out_path = os.path.join(viz_dir, 'summary.html')
    with open(out_path, 'w') as f:
        f.write('\n'.join(html))
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_episodes(data_dir: str) -> List[Dict]:
    eps = []
    for name in sorted(os.listdir(data_dir)):
        if not name.startswith('episode_'):
            continue
        ep_dir = os.path.join(data_dir, name)
        meta_path  = os.path.join(ep_dir, 'metadata.json')
        steps_path = os.path.join(ep_dir, 'steps.json')
        if not (os.path.exists(meta_path) and os.path.exists(steps_path)):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with open(steps_path) as f:
            steps = json.load(f)
        eps.append({'meta': meta, 'steps': steps, 'ep_dir': ep_dir})
    return eps


def _latest_episode_id(data_dir: str) -> Optional[int]:
    ids = []
    for name in os.listdir(data_dir):
        if name.startswith('episode_'):
            try:
                ids.append(int(name.split('_')[1]))
            except (IndexError, ValueError):
                pass
    return max(ids) if ids else None


def _print_table(episodes: List[Dict]) -> None:
    if not episodes:
        return
    n_ok = sum(1 for e in episodes if e['meta'].get('success'))
    print(f'\n{"ID":>6}  {"Result":8}  {"Steps":>5}  {"Dur(s)":>7}  Teacher')
    print('─' * 50)
    for ep in reversed(episodes):
        m   = ep['meta']
        eid = m.get('episode_id', '?')
        res = 'SUCCESS' if m.get('success') else 'FAIL'
        print(f'{str(eid):>6}  {res:8}  {str(m.get("n_steps","?")):>5}  '
              f'{m.get("duration_s", 0.0):>7.1f}  {m.get("teacher","?")}')
    print('─' * 50)
    print(f'Total {len(episodes)}  ·  Success {n_ok}'
          f'  ·  Rate {n_ok/max(1,len(episodes))*100:.1f}%\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Visualise BC alignment episodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--data-dir', required=True,
                    help='Directory containing episode_NNNNN/ subdirs')
    ap.add_argument('--episode', type=int, default=None,
                    help='Render a single episode by numeric ID')
    ap.add_argument('--latest', action='store_true',
                    help='Render the most recently collected episode')
    ap.add_argument('--all', action='store_true',
                    help='Render HTML reports for every episode')
    ap.add_argument('--video', action='store_true',
                    help='Also render MP4 video per episode (requires opencv)')
    ap.add_argument('--viz-dir', default=None,
                    help='Output directory (default: <data-dir>/viz)')
    args = ap.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f'ERROR: data-dir not found: {data_dir}')
        sys.exit(1)

    viz_dir = args.viz_dir or os.path.join(data_dir, 'viz')

    # ── Single episode (explicit or --latest) ──────────────────────────
    ep_id = args.episode
    if args.latest:
        ep_id = _latest_episode_id(data_dir)
        if ep_id is None:
            print(f'No episodes found in {data_dir}')
            sys.exit(0)
        print(f'Latest episode: {ep_id}')

    if ep_id is not None:
        ep_name = f'episode_{ep_id:05d}'
        ep_dir  = os.path.join(data_dir, ep_name)
        if not os.path.isdir(ep_dir):
            ep_dir = os.path.join(data_dir, f'episode_{ep_id}')
        meta_path  = os.path.join(ep_dir, 'metadata.json')
        steps_path = os.path.join(ep_dir, 'steps.json')
        if not (os.path.exists(meta_path) and os.path.exists(steps_path)):
            print(f'ERROR: episode {ep_id} not found or incomplete in {data_dir}')
            sys.exit(1)
        with open(meta_path) as f:
            meta = json.load(f)
        with open(steps_path) as f:
            steps = json.load(f)
        ep = {'meta': meta, 'steps': steps, 'ep_dir': ep_dir}

        html_path = render_episode_html(ep, viz_dir)
        print(f'HTML report  : {html_path}')
        if args.video:
            vid = render_episode_video(ep, viz_dir)
            if vid:
                print(f'Video        : {vid}')
        # Always regenerate summary when viewing a single episode
        eps = load_all_episodes(data_dir)
        summary = render_summary_html(eps, viz_dir)
        print(f'Summary      : {summary}')
        return

    # ── List / --all ───────────────────────────────────────────────────
    eps = load_all_episodes(data_dir)
    if not eps:
        print(f'No episodes found in {data_dir}')
        sys.exit(0)

    _print_table(eps)
    summary = render_summary_html(eps, viz_dir)
    print(f'Summary HTML : {summary}')

    if args.all:
        print('Rendering all episodes …')
        for ep in eps:
            html_path = render_episode_html(ep, viz_dir)
            vid_path  = render_episode_video(ep, viz_dir) if args.video else ''
            status    = '✓' if ep['meta'].get('success') else '✗'
            print(f'  [{status}] {html_path}'
                  + (f'\n       {vid_path}' if vid_path else ''))
        print(f'\nDone. Open {viz_dir}/summary.html in a browser.')


if __name__ == '__main__':
    main()
