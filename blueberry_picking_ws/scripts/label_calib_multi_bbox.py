#!/usr/bin/env python3
"""Multi-view bbox labeler for extrinsic XY calibration sessions.

  python3 scripts/label_calib_multi_bbox.py \\
    --session-dir log/real_robot/calib_extrinsic_xy/<session>
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def _encode_image(path: Path) -> str:
    ext = path.suffix.lower().lstrip('.')
    mime = 'image/png' if ext == 'png' else f'image/{ext}'
    return f'data:{mime};base64,{base64.b64encode(path.read_bytes()).decode("ascii")}'


def _html(session: str, views: List[Dict[str, Any]]) -> str:
    payload = json.dumps(views, ensure_ascii=True)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><title>外参标定标框 — {session}</title>
<style>
body {{ font-family: system-ui,sans-serif; margin:16px; background:#1a1a1a; color:#eee; }}
h1 {{ font-size:1.1rem; }}
.nav {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; }}
.nav button {{ padding:6px 12px; border-radius:6px; border:none; cursor:pointer; }}
.nav button.active {{ background:#2d6a4f; color:#fff; }}
.nav button.done {{ background:#1b4332; color:#a7f3d0; }}
#wrap {{ position:relative; display:inline-block; cursor:crosshair; }}
canvas {{ border:1px solid #444; display:block; }}
.bar {{ margin:12px 0; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
#save {{ background:#2d6a4f; color:#fff; padding:8px 16px; border:none; border-radius:6px; }}
#save:disabled {{ background:#444; color:#888; }}
#coords, #msg {{ font-family:monospace; }}
#msg.err {{ color:#ff8a8a; }}
</style></head><body>
<h1>多视角标框 — 同一颗接触果（GT 已固定）</h1>
<p>每个角度分别框住<strong>同一颗果</strong>，切换视角后点「保存当前视角」，全部完成后点「提交全部」。</p>
<div class="nav" id="nav"></div>
<div id="wrap"><canvas id="c"></canvas></div>
<div class="bar">
  <span id="coords">尚未画框</span>
  <button type="button" id="reset">重画</button>
  <button type="button" id="save" disabled>保存当前视角</button>
  <button type="button" id="skip">跳过此视角</button>
  <button type="button" id="submit">提交全部</button>
</div>
<div id="msg"></div>
<script>
const views = {payload};
let idx = 0, box = null, drawing = false, start = null;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const nav = document.getElementById('nav');
const imgs = views.map(() => new Image());

function normBox(x0,y0,x1,y1) {{
  const a0=Math.round(Math.min(x0,x1)), a1=Math.round(Math.min(y0,y1));
  const b0=Math.round(Math.max(x0,x1)), b1=Math.round(Math.max(y0,y1));
  if (b0-a0<4||b1-a1<4) return null;
  return [a0,a1,b0,b1];
}}

function draw() {{
  const v = views[idx];
  const img = imgs[idx];
  if (!img.complete) return;
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  if (!box) return;
  const [x0,y0,x1,y1]=box;
  ctx.strokeStyle='#00ff88'; ctx.lineWidth=2;
  ctx.strokeRect(x0,y0,x1-x0,y1-y0);
  ctx.fillStyle='rgba(0,255,136,0.15)';
  ctx.fillRect(x0,y0,x1-x0,y1-y0);
}}

function updateNav() {{
  nav.innerHTML = '';
  views.forEach((v,i) => {{
    const b = document.createElement('button');
    b.textContent = v.id + (v.skipped ? ' 跳过' : (v.bbox_xyxy ? ' ✓' : ''));
    if (i===idx) b.className='active' + (v.bbox_xyxy?' done':'');
    else if (v.skipped) b.className='';
    else if (v.bbox_xyxy) b.className='done';
    b.onclick = () => {{ idx=i; box=v.bbox_xyxy? [...v.bbox_xyxy]:null; updateUi(); draw(); }};
    nav.appendChild(b);
  }});
}}

function updateUi() {{
  const el=document.getElementById('coords');
  const save=document.getElementById('save');
  if (!box) {{ el.textContent='尚未画框'; save.disabled=true; return; }}
  el.textContent=`x0=${{box[0]}} y0=${{box[1]}} x1=${{box[2]}} y1=${{box[3]}}`;
  save.disabled=false;
}}

views.forEach((v,i) => {{
  imgs[i].src = v.img_b64;
  if (v.bbox_xyxy) v.bbox_xyxy = [...v.bbox_xyxy];
  imgs[i].onload = () => {{ if (i===idx) draw(); }};
}});

canvas.onmousedown = e => {{
  const r=canvas.getBoundingClientRect(), sx=canvas.width/r.width, sy=canvas.height/r.height;
  start=[(e.clientX-r.left)*sx,(e.clientY-r.top)*sy]; drawing=true; box=null; updateUi();
}};
canvas.onmousemove = e => {{
  if (!drawing||!start) return;
  const r=canvas.getBoundingClientRect(), sx=canvas.width/r.width, sy=canvas.height/r.height;
  const x=(e.clientX-r.left)*sx, y=(e.clientY-r.top)*sy;
  const nb=normBox(start[0],start[1],x,y); if(nb) box=nb; draw(); updateUi();
}};
window.onmouseup = () => drawing=false;

document.getElementById('reset').onclick = () => {{ box=null; draw(); updateUi(); }};
document.getElementById('save').onclick = () => {{
  if (!box) return;
  views[idx].bbox_xyxy = [...box];
  updateNav(); document.getElementById('msg').textContent = views[idx].id + ' 已暂存';
}};
document.getElementById('skip').onclick = () => {{
  views[idx].skipped = true;
  views[idx].bbox_xyxy = null;
  box = null;
  updateNav(); updateUi();
  document.getElementById('msg').textContent = views[idx].id + ' 已标记跳过';
}};
document.getElementById('submit').onclick = async () => {{
  const missing = views.filter(v => !v.skipped && !v.bbox_xyxy);
  if (missing.length) {{
    alert('还有未标视角: ' + missing.map(v=>v.id).join(', ') + '（或点「跳过此视角」）');
    return;
  }}
  const msg=document.getElementById('msg');
  try {{
    const res = await fetch('/save', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{ views: views.map(v => ({{
        id:v.id, bbox_xyxy:v.bbox_xyxy, skipped: !!v.skipped
      }})) }}),
    }});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error||res.statusText);
    msg.textContent = '已保存 ' + data.path;
  }} catch(e) {{ msg.className='err'; msg.textContent='失败: '+e.message; }}
}};

idx=0; box=views[0].bbox_xyxy?[...views[0].bbox_xyxy]:null;
updateNav(); updateUi();
</script></body></html>"""


def run_server(session_dir: Path, port: int, open_browser: bool) -> int:
    manifest = json.loads((session_dir / 'manifest.json').read_text(encoding='utf-8'))
    views_js: List[Dict[str, Any]] = []
    for v in manifest['views']:
        img_path = session_dir / v['image']
        views_js.append({
            'id': v['id'],
            'img_b64': _encode_image(img_path),
            'bbox_xyxy': v.get('bbox_xyxy'),
            'skipped': bool(v.get('skipped')),
        })
    page = _html(manifest['session'], views_js)
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            pass

        def do_GET(self) -> None:
            if urlparse(self.path).path in ('/', '/index.html'):
                body = page.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != '/save':
                self.send_error(404)
                return
            n = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(n).decode('utf-8'))
            by_id = {v['id']: v for v in data['views']}
            for v in manifest['views']:
                if v['id'] not in by_id:
                    continue
                row = by_id[v['id']]
                if row.get('skipped'):
                    v['skipped'] = True
                    v['bbox_xyxy'] = None
                    v['labeled'] = False
                    v['skip_reason'] = 'user_no_berry'
                elif row.get('bbox_xyxy'):
                    v['bbox_xyxy'] = row['bbox_xyxy']
                    v['labeled'] = True
                    v['skipped'] = False
            manifest['all_labeled'] = all(
                v.get('labeled') or v.get('skipped') for v in manifest['views'])
            tmp = session_dir / 'manifest.json.tmp'
            tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            os.replace(tmp, session_dir / 'manifest.json')
            body = json.dumps({'ok': True, 'path': str(session_dir / 'manifest.json')})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body.encode('utf-8'))
            print(f'saved bboxes to {session_dir / "manifest.json"}')
            done.set()

    httpd = HTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{port}/'
    print(f'多视角标框: {url}  ({len(manifest["views"])} views)')
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.timeout = 0.5
        while not done.is_set():
            httpd.handle_request()
    except KeyboardInterrupt:
        return 130
    httpd.server_close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-dir', required=True)
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--no-browser', action='store_true')
    args = p.parse_args()
    session_dir = Path(args.session_dir)
    if not session_dir.is_dir():
        session_dir = ROOT / args.session_dir
    if not (session_dir / 'manifest.json').is_file():
        print('missing manifest.json', file=sys.stderr)
        return 2
    return run_server(session_dir, args.port, not args.no_browser)


if __name__ == '__main__':
    raise SystemExit(main())
