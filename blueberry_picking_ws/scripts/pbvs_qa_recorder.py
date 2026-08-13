"""PBVS REFINING QA recorder — every control tick + 4 camera views + HTML replay.

Layout under ``<qa_dir>/<session>/``::

    pbvs_stream.jsonl          one JSON object per 25 Hz tick (always)
    pbvs_frames/frame_NNNNN_{fixed,wrist,global_viz,fine_viz}.jpg
    pbvs_session.json          summary + outcome
    pbvs_replay.html           browser timeline (open after run)

Also appends key events to ``pbvs_events.jsonl`` (handoff, contact, done, error).
"""

from __future__ import annotations

import json
import os
import time
from html import escape
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


class PbvsQaRecorder:
    """Record PBVS ticks into QA session dir; render HTML on finalize."""

    def __init__(
        self,
        qa_dir: str,
        session_id: str,
        *,
        jpeg_quality: int = 82,
    ) -> None:
        self._session_id = session_id
        self._session_dir = os.path.join(qa_dir, session_id)
        self._frames_dir = os.path.join(self._session_dir, 'pbvs_frames')
        self._jsonl_path = os.path.join(self._session_dir, 'pbvs_stream.jsonl')
        self._events_path = os.path.join(self._session_dir, 'pbvs_events.jsonl')
        self._jpeg_quality = int(jpeg_quality)
        self._frame_idx = 0
        self._t0 = time.time()
        self._outcome = 'unknown'
        self._events: List[Dict[str, Any]] = []
        os.makedirs(self._frames_dir, exist_ok=True)
        # Append if jsonl already exists (should not happen mid-session).
        if not os.path.isfile(self._jsonl_path):
            open(self._jsonl_path, 'w', encoding='utf-8').close()
        if not os.path.isfile(self._events_path):
            open(self._events_path, 'w', encoding='utf-8').close()
        # Resume frame index from existing stream.
        for row in load_stream(self._jsonl_path):
            self._frame_idx = max(self._frame_idx, int(row.get('frame', -1)) + 1)

    @property
    def session_dir(self) -> str:
        return self._session_dir

    @property
    def frame_count(self) -> int:
        return self._frame_idx

    def log_event(self, kind: str, detail: Optional[Dict[str, Any]] = None) -> None:
        ev = {
            't': time.time(),
            't_rel_s': round(time.time() - self._t0, 3),
            'frame': self._frame_idx,
            'kind': kind,
            'detail': _json_safe(detail or {}),
        }
        self._events.append(ev)
        with open(self._events_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(ev, ensure_ascii=True) + '\n')

    def record_tick(
        self,
        meta: Dict[str, Any],
        images: Dict[str, Optional[np.ndarray]],
    ) -> None:
        """Save one PBVS control tick (images may be None if not yet received)."""
        idx = self._frame_idx
        rel_paths: Dict[str, str] = {}
        for key, rgb in images.items():
            if rgb is None:
                continue
            fname = f'frame_{idx:05d}_{key}.jpg'
            abspath = os.path.join(self._frames_dir, fname)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
            cv2.imwrite(
                abspath, bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            rel_paths[key] = f'pbvs_frames/{fname}'

        record = {
            'frame': idx,
            't': time.time(),
            't_rel_s': round(time.time() - self._t0, 4),
            'images': rel_paths,
            **_json_safe(meta),
        }
        with open(self._jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')
        self._frame_idx += 1

    def finalize(self, outcome: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Write session summary + HTML; return path to replay.html."""
        self._outcome = outcome
        summary = {
            'session': self._session_id,
            'outcome': outcome,
            'n_frames': self._frame_idx,
            'duration_s': round(time.time() - self._t0, 2),
            'stream_jsonl': 'pbvs_stream.jsonl',
            'events_jsonl': 'pbvs_events.jsonl',
            'frames_dir': 'pbvs_frames',
            'extra': _json_safe(extra or {}),
        }
        with open(os.path.join(self._session_dir, 'pbvs_session.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)

        html_path = os.path.join(self._session_dir, 'pbvs_replay.html')
        render_pbvs_replay_html(self._session_dir, html_path)
        return html_path


def load_stream(jsonl_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(jsonl_path):
        return rows
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def render_pbvs_replay_html(session_dir: str, out_path: str) -> None:
    """Build interactive PBVS replay player (scrub/play + depth charts)."""
    rows = load_stream(os.path.join(session_dir, 'pbvs_stream.jsonl'))
    events = load_stream(os.path.join(session_dir, 'pbvs_events.jsonl'))

    lock_meta: Dict[str, Any] = {}
    lock_path = os.path.join(session_dir, 'refine_fruit_lock.json')
    if os.path.isfile(lock_path):
        try:
            with open(lock_path, encoding='utf-8') as f:
                lock_meta = json.load(f)
        except Exception:
            pass

    session_name = os.path.basename(session_dir)
    payload = {
        'session': session_name,
        'lock': lock_meta,
        'frames': rows,
        'events': events,
    }
    data_json = json.dumps(_json_safe(payload), ensure_ascii=True)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PBVS replay {escape(session_name)}</title>
<style>
:root {{
  --bg:#12141c; --panel:#1a1f2e; --accent:#5b9cff; --warn:#ff6b6b; --text:#e8ecf4; --muted:#8b93a7;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--text); }}
header {{ padding:12px 16px; border-bottom:1px solid #2a3145; }}
h1 {{ margin:0; font-size:1.05rem; }}
.sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.layout {{ display:grid; grid-template-columns:1fr 320px; gap:12px; padding:12px; min-height:calc(100vh - 60px); }}
@media (max-width:1100px) {{ .layout {{ grid-template-columns:1fr; }} }}
.viewer {{ background:var(--panel); border-radius:10px; padding:10px; }}
.grid4 {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.grid4 figure {{ margin:0; background:#000; border-radius:6px; overflow:hidden; position:relative; }}
.grid4 figure.fine-viz-full {{ grid-column:1 / -1; }}
.grid4 img {{ width:100%; display:block; aspect-ratio:4/3; object-fit:contain; background:#000; }}
.grid4 figure.fine-viz-full img {{ aspect-ratio:16/10; max-height:520px; }}
.grid4 figcaption {{ position:absolute; left:6px; top:6px; font-size:12px; font-weight:600; background:rgba(0,0,0,.72); padding:3px 8px; border-radius:4px; z-index:1; }}
.grid4 figure.fine-viz-full figcaption {{ background:rgba(20,40,20,.85); }}
.side {{ display:flex; flex-direction:column; gap:10px; }}
.panel {{ background:var(--panel); border-radius:10px; padding:10px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
button, select, input[type=number] {{
  background:#252b3d; color:var(--text); border:1px solid #3a4460; border-radius:6px; padding:6px 10px;
}}
button:hover {{ border-color:var(--accent); cursor:pointer; }}
input[type=range] {{ flex:1; min-width:120px; }}
.meta {{ font-family:ui-monospace,monospace; font-size:11px; line-height:1.5; white-space:pre-wrap; }}
canvas {{ width:100%; height:120px; background:#0d1018; border-radius:6px; }}
.event-btns {{ display:flex; flex-wrap:wrap; gap:6px; }}
.event-btns button {{ font-size:11px; }}
.legend {{ font-size:11px; color:var(--muted); }}
.coast {{ color:var(--warn); }}
</style></head><body>
<header>
  <h1>PBVS replay — {escape(session_name)}</h1>
  <div class="sub" id="hdr">loading…</div>
</header>
<div class="layout">
  <div class="viewer">
    <div class="controls panel" style="margin-bottom:10px">
      <button id="btnPlay">▶ 播放</button>
      <label>速度 <select id="fpsSel"><option value="5">5 fps</option><option value="10" selected>10 fps</option><option value="25">25 fps</option></select></label>
      <label>帧 <input id="frameIn" type="number" min="0" value="0" style="width:72px"/></label>
      <span id="timeLbl">t=0.0s</span>
      <input id="scrub" type="range" min="0" max="0" value="0" step="1"/>
    </div>
    <div class="grid4" id="imgs"></div>
  </div>
  <div class="side">
    <div class="panel">
      <div class="legend">曲线：cup_dist(mm) · z_depth · z_mono · z_used · berry_z</div>
      <canvas id="chart" height="140"></canvas>
    </div>
    <div class="panel meta" id="meta"></div>
    <div class="panel">
      <div style="font-size:12px;margin-bottom:6px">事件跳转</div>
      <div class="event-btns" id="evBtns"></div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
const FRAMES = DATA.frames || [];
const EVENTS = DATA.events || [];
let idx = 0, playing = false, timer = null;
const imgsEl = document.getElementById('imgs');
const metaEl = document.getElementById('meta');
const scrub = document.getElementById('scrub');
const frameIn = document.getElementById('frameIn');
const timeLbl = document.getElementById('timeLbl');
const hdr = document.getElementById('hdr');
const chart = document.getElementById('chart');
const ctx = chart.getContext('2d');

function fmt(v, d=3) {{ return (v==null||Number.isNaN(v)) ? '—' : Number(v).toFixed(d); }}
function mm(v) {{ return v==null ? '—' : (v*1000).toFixed(1)+' mm'; }}
function xyzm(v) {{
  if (!v || !v.length) return '—';
  return '(' + v.map(x => Number(x).toFixed(4)).join(', ') + ') m';
}}
function xyzmm(v) {{
  if (!v || !v.length) return '—';
  return '(' + v.map(x => (Number(x)*1000).toFixed(1)).join(', ') + ') mm';
}}
function dist3(a, b) {{
  if (!a || !b || a.length < 3 || b.length < 3) return null;
  const dx = Number(a[0]) - Number(b[0]);
  const dy = Number(a[1]) - Number(b[1]);
  const dz = Number(a[2]) - Number(b[2]);
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}}
function dxyz3(a, b) {{
  if (!a || !b || a.length < 3 || b.length < 3) return null;
  return [Number(a[0]) - Number(b[0]), Number(a[1]) - Number(b[1]), Number(a[2]) - Number(b[2])];
}}
function dxyzmm(v) {{
  if (!v || !v.length) return '—';
  return '(' + v.map(x => (Number(x)*1000).toFixed(1)).join(', ') + ') mm';
}}

function buildImages() {{
  imgsEl.innerHTML = '';
  const labels = {{
    fixed: 'fixed',
    wrist: 'wrist',
    global_viz: 'global_viz',
    fine_viz: 'fine_viz — locked target only',
  }};
  ['fixed','wrist','global_viz','fine_viz'].forEach(k => {{
    const fig = document.createElement('figure');
    if (k === 'fine_viz') fig.className = 'fine-viz-full';
    const cap = document.createElement('figcaption'); cap.textContent = labels[k] || k;
    const img = document.createElement('img'); img.id = 'img_'+k; img.alt = k;
    fig.appendChild(cap); fig.appendChild(img); imgsEl.appendChild(fig);
  }});
}}

function showFrame(i) {{
  if (!FRAMES.length) return;
  idx = Math.max(0, Math.min(FRAMES.length-1, i));
  const r = FRAMES[idx];
  scrub.value = idx; frameIn.value = r.frame ?? idx; timeLbl.textContent = 't=' + fmt(r.t_rel_s,2) + 's  frame=' + (r.frame??idx);
  const imgs = r.images || {{}};
  ['fixed','wrist','global_viz','fine_viz'].forEach(k => {{
    const el = document.getElementById('img_'+k);
    if (el) {{ el.src = imgs[k] || ''; el.style.visibility = imgs[k] ? 'visible' : 'hidden'; }}
  }});
  const coast = (r.depth_mode||'').includes('coast');
  const berryBase = r.berry_base_locked || r.frozen_target || r.berry_xyz;
  const cupBase = r.cup_open_xyz || r.tip_xyz;  // tip === cup opening
  const lockId = (DATA.lock && DATA.lock.track_id != null) ? DATA.lock.track_id : null;
  const tid = (r.track_id != null) ? r.track_id : lockId;
  const cupBerryDist = (r.cup_berry_dist_m != null) ? r.cup_berry_dist_m : (
    (berryBase && cupBase) ? dist3(berryBase, cupBase) : null);
  const cupBerryDxyz = r.cup_berry_dxyz_m || (
    (berryBase && cupBase) ? dxyz3(berryBase, cupBase) : null);
  const liveBase = r.live_berry_base || null;
  const cupLiveDist = r.cup_live_berry_dist_m;
  metaEl.innerHTML = [
    '—— 圈选目标 (base_link) ——',
    'track_id=' + (tid != null ? tid : '—'),
    'berry_base(frozen)=' + xyzm(berryBase),
    'berry_base_mm=' + xyzmm(berryBase),
    (liveBase ? 'berry_base(live)=' + xyzm(liveBase) : ''),
    'cup_open=' + xyzm(cupBase),
    'cup_open_mm=' + xyzmm(cupBase),
    '',
    'cup↔frozen=' + mm(cupBerryDist) + '  Δxyz=' + dxyzmm(cupBerryDxyz) +
      (r.contact_exec_tol_m != null ? '  (tol=' + mm(r.contact_exec_tol_m) + ')' : ''),
    (cupLiveDist != null ? 'cup↔live(diag)=' + mm(cupLiveDist) : ''),
    'state=' + (r.pbvs_state||'') + (coast ? ' <span class="coast">[COAST]</span>' : ''),
    'source=' + (r.source||''),
    'd_cam_surface=' + mm(r.d_cam_surface_m),
    'cup_dist=' + mm(r.cup_dist_m),
    '|e|=' + mm(r.err_m),
    'cam_dist=' + mm(r.cam_dist_m),
    'z_depth=' + mm(r.z_depth_m),
    'z_mono=' + mm(r.z_mono_m),
    'z_used=' + mm(r.z_used_m),
    'depth_mode=' + (r.depth_mode||''),
    'tcp_base=' + xyzm(r.tcp_base),
    'des=' + xyzm(r.p_des),
    'kf_unc=' + fmt(r.kf_uncertainty,6),
  ].join('\\n');
  drawChart(idx);
}}

function drawChart(cur) {{
  const dpr = window.devicePixelRatio || 1;
  const w = chart.clientWidth, h = chart.clientHeight;
  chart.width = w*dpr; chart.height = h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  if (!FRAMES.length) return;
  const keys = [
    {{k:'cup_dist_m', c:'#5b9cff', scale:1000}},
    {{k:'z_depth_m', c:'#7dffb3', scale:1000}},
    {{k:'z_mono_m', c:'#ffd166', scale:1000}},
    {{k:'z_used_m', c:'#ff9f43', scale:1000}},
    {{k:'berry_z_m', c:'#c792ea', scale:1000}},
  ];
  const series = keys.map(({{k,scale}}) => FRAMES.map(r => (r[k]==null?null:r[k]*scale)));
  const vals = series.flat().filter(v => v!=null);
  if (!vals.length) {{ ctx.fillStyle='#8b93a7'; ctx.fillText('无 depth 曲线数据（需新录制 jsonl）', 10, 20); return; }}
  let ymin = Math.min(...vals), ymax = Math.max(...vals);
  if (ymax-ymin < 5) {{ ymin -= 5; ymax += 5; }}
  const pad = 8;
  function x(i) {{ return pad + (w-2*pad) * (FRAMES.length<=1 ? 0.5 : i/(FRAMES.length-1)); }}
  function y(v) {{ return h-pad - (h-2*pad)*((v-ymin)/(ymax-ymin)); }}
  keys.forEach(({{k,c,scale}}, si) => {{
    ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.beginPath(); let started=false;
    series[si].forEach((v,i) => {{
      if (v==null) {{ started=false; return; }}
      const px=x(i), py=y(v);
      if (!started) {{ ctx.moveTo(px,py); started=true; }} else ctx.lineTo(px,py);
    }});
    ctx.stroke();
  }});
  const cx = x(cur);
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, pad); ctx.lineTo(cx, h-pad); ctx.stroke();
  ctx.fillStyle='#8b93a7'; ctx.font='10px system-ui';
  keys.forEach(({{k,c}}, i) => {{ ctx.fillStyle=c; ctx.fillText(k, pad + i*88, 12); }});
}}

function jumpEvent(ev) {{
  const f = ev.frame ?? 0;
  const i = FRAMES.findIndex(r => (r.frame??0) >= f);
  showFrame(i >= 0 ? i : 0);
}}

function setupEvents() {{
  const box = document.getElementById('evBtns');
  EVENTS.forEach(ev => {{
    const b = document.createElement('button');
    b.textContent = (ev.kind||'?') + ' @' + fmt(ev.t_rel_s,1) + 's';
    b.onclick = () => jumpEvent(ev);
    box.appendChild(b);
  }});
  const lock = DATA.lock || {{}};
  if (lock.berry_base) {{
    hdr.textContent = 'lock tid=' + lock.track_id + ' berry_base=' + JSON.stringify(lock.berry_base) +
      ' z_depth=' + fmt(lock.z_depth_m) + ' z_mono=' + fmt(lock.z_mono_m) + '  frames=' + FRAMES.length;
  }} else {{
    hdr.textContent = 'frames=' + FRAMES.length + ' events=' + EVENTS.length;
  }}
}}

document.getElementById('btnPlay').onclick = () => {{
  playing = !playing;
  document.getElementById('btnPlay').textContent = playing ? '⏸ 暂停' : '▶ 播放';
  if (playing) {{
    const fps = Number(document.getElementById('fpsSel').value)||10;
    timer = setInterval(() => {{
      if (idx >= FRAMES.length-1) {{ playing=false; clearInterval(timer); document.getElementById('btnPlay').textContent='▶ 播放'; return; }}
      showFrame(idx+1);
    }}, 1000/fps);
  }} else if (timer) clearInterval(timer);
}};
scrub.max = Math.max(0, FRAMES.length-1);
scrub.oninput = () => {{ if (playing) document.getElementById('btnPlay').click(); showFrame(Number(scrub.value)); }};
frameIn.onchange = () => {{
  const target = Number(frameIn.value);
  const i = FRAMES.findIndex(r => (r.frame??0) === target);
  showFrame(i >= 0 ? i : Number(scrub.value));
}};
document.getElementById('fpsSel').onchange = () => {{ if (playing) {{ document.getElementById('btnPlay').click(); document.getElementById('btnPlay').click(); }} }};
window.addEventListener('keydown', e => {{
  if (e.code==='Space') {{ e.preventDefault(); document.getElementById('btnPlay').click(); }}
  if (e.code==='ArrowRight') showFrame(idx+1);
  if (e.code==='ArrowLeft') showFrame(idx-1);
}});
window.addEventListener('resize', () => drawChart(idx));
buildImages(); setupEvents(); showFrame(0);
</script></body></html>'''
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python pbvs_qa_recorder.py <qa_session_dir>')
        sys.exit(1)
    d = sys.argv[1]
    out = os.path.join(d, 'pbvs_replay.html')
    render_pbvs_replay_html(d, out)
    print(f'wrote {out}')
