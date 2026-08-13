#!/usr/bin/env python3
"""Interactive wrist target bbox labeler.

Draw a rectangle on the entry image in the browser; coordinates are written
automatically to target_bbox_decision.json (and optionally target_bbox_lock.json).

  python3 scripts/label_target_bbox.py \\
    --session-dir log/real_robot/qa/20260812_152447 --also-global

Open the printed URL, drag on the image, click「保存标框」.
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
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from write_target_bbox_decision import atomic_write_json  # noqa: E402


def _encode_image_b64(path: Path) -> str:
    data = path.read_bytes()
    ext = path.suffix.lower().lstrip('.')
    mime = 'image/png' if ext == 'png' else f'image/{ext}'
    return f'data:{mime};base64,{base64.b64encode(data).decode("ascii")}'


def _html_page(img_b64: str, session_name: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>标接触果 — {session_name}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 16px; background: #1a1a1a; color: #eee; }}
  h1 {{ font-size: 1.1rem; font-weight: 600; }}
  p {{ color: #aaa; max-width: 720px; line-height: 1.5; }}
  #wrap {{ position: relative; display: inline-block; cursor: crosshair; user-select: none; }}
  canvas {{ display: block; border: 1px solid #444; }}
  .bar {{ margin: 12px 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  button {{ padding: 8px 16px; font-size: 14px; cursor: pointer; border-radius: 6px; border: none; }}
  #save {{ background: #2d6a4f; color: #fff; }}
  #save:disabled {{ background: #444; color: #888; cursor: not-allowed; }}
  #reset {{ background: #444; color: #eee; }}
  #coords {{ font-family: monospace; font-size: 15px; min-width: 220px; }}
  #msg {{ color: #95d5b2; min-height: 1.2em; }}
  #msg.err {{ color: #ff8a8a; }}
</style>
</head>
<body>
<h1>标出接触果（拖拽画框）</h1>
<p>在图上按住鼠标左键拖拽，框住<strong>上次 single 接触的那颗果</strong>。
松开后可调整；满意后点「保存标框」。坐标会自动写入 session，无需手抄数字。</p>
<div id="wrap">
  <canvas id="c"></canvas>
</div>
<div class="bar">
  <span id="coords">尚未画框</span>
  <button id="reset" type="button">重画</button>
  <button id="save" type="button" disabled>保存标框</button>
</div>
<div id="msg"></div>
<script>
const img = new Image();
img.src = {json.dumps(img_b64)};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let drawing = false, start = null, box = null;

function draw() {{
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  if (!box) return;
  const [x0, y0, x1, y1] = box;
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.fillStyle = 'rgba(0,255,136,0.15)';
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
}}

img.onload = () => draw();

function normBox(x0, y0, x1, y1) {{
  const a0 = Math.round(Math.min(x0, x1));
  const a1 = Math.round(Math.min(y0, y1));
  const b0 = Math.round(Math.max(x0, x1));
  const b1 = Math.round(Math.max(y0, y1));
  if (b0 - a0 < 4 || b1 - a1 < 4) return null;
  return [a0, a1, b0, b1];
}}

function updateUi() {{
  const el = document.getElementById('coords');
  const save = document.getElementById('save');
  if (!box) {{
    el.textContent = '尚未画框';
    save.disabled = true;
    return;
  }}
  el.textContent = `x0=${{box[0]}}  y0=${{box[1]}}  x1=${{box[2]}}  y1=${{box[3]}}`;
  save.disabled = false;
}}

canvas.addEventListener('mousedown', (e) => {{
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width, sy = canvas.height / r.height;
  start = [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
  drawing = true;
  box = null;
  updateUi();
}});

canvas.addEventListener('mousemove', (e) => {{
  if (!drawing || !start) return;
  const r = canvas.getBoundingClientRect();
  const sx = canvas.width / r.width, sy = canvas.height / r.height;
  const x = (e.clientX - r.left) * sx, y = (e.clientY - r.top) * sy;
  const nb = normBox(start[0], start[1], x, y);
  if (nb) box = nb;
  draw();
  updateUi();
}});

window.addEventListener('mouseup', () => {{ drawing = false; }});

document.getElementById('reset').onclick = () => {{
  box = null; draw(); updateUi();
  document.getElementById('msg').textContent = '';
  document.getElementById('msg').className = '';
}};

document.getElementById('save').onclick = async () => {{
  if (!box) return;
  const msg = document.getElementById('msg');
  msg.className = '';
  msg.textContent = '保存中…';
  try {{
    const res = await fetch('/save', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ bbox_xyxy: box, reason: 'user_browser_label' }}),
    }});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    msg.textContent = '已保存: ' + data.path + (data.global ? '  + ' + data.global : '');
  }} catch (err) {{
    msg.className = 'err';
    msg.textContent = '保存失败: ' + err.message;
  }}
}};
</script>
</body>
</html>
"""


def run_server(
    session_dir: Path,
    image_path: Path,
    also_global: bool,
    port: int,
    open_browser: bool,
) -> int:
    img_b64 = _encode_image_b64(image_path)
    page = _html_page(img_b64, session_dir.name)
    done = threading.Event()
    result: dict = {'code': 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            pass

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
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
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode('utf-8'))
                bb = payload.get('bbox_xyxy')
                if not bb or len(bb) != 4:
                    raise ValueError('bbox_xyxy required')
                x0, y0, x1, y1 = [int(v) for v in bb]
                if x1 <= x0 or y1 <= y0:
                    raise ValueError('invalid bbox')
                reason = str(payload.get('reason') or 'user_browser_label')
                cu = 0.5 * (x0 + x1)
                cv = 0.5 * (y0 + y1)
                out_payload = {
                    'action': 'lock_bbox',
                    'bbox_xyxy': [x0, y0, x1, y1],
                    'center_uv': [cu, cv],
                    'reason': reason,
                    'session': session_dir.name,
                    'provider': 'user',
                }
                out_path = session_dir / 'target_bbox_decision.json'
                atomic_write_json(out_path, out_payload)
                global_path: Optional[Path] = None
                if also_global:
                    global_path = ROOT / 'log' / 'real_robot' / 'target_bbox_lock.json'
                    atomic_write_json(global_path, out_payload)
                resp = {
                    'ok': True,
                    'path': str(out_path),
                    'bbox_xyxy': [x0, y0, x1, y1],
                    'global': str(global_path) if global_path else None,
                }
                body = json.dumps(resp, ensure_ascii=True).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                print(f'saved {out_path} bbox=[{x0},{y0},{x1},{y1}]')
                if global_path:
                    print(f'saved {global_path}')
                result['bbox'] = (x0, y0, x1, y1)
                done.set()
            except Exception as exc:
                body = json.dumps({'error': str(exc)}).encode('utf-8')
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(body)

    httpd = HTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{port}/'
    print(f'session={session_dir.name}')
    print(f'image={image_path}')
    print(f'标框页面: {url}')
    print('拖拽画框 → 点「保存标框」→ 终端会打印坐标；Ctrl+C 退出。')
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.timeout = 0.5
        while not done.is_set():
            httpd.handle_request()
    except KeyboardInterrupt:
        print('\n已退出（未保存则请重新运行）')
        return 130 if 'bbox' not in result else 0
    httpd.server_close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-dir', required=True)
    p.add_argument('--image', default='', help='Override image path')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--also-global', action='store_true')
    p.add_argument('--no-browser', action='store_true')
    args = p.parse_args()

    session = Path(args.session_dir)
    if not session.is_dir():
        session = ROOT / args.session_dir
    if not session.is_dir():
        print(f'missing session: {args.session_dir}', file=sys.stderr)
        return 2

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = session / 'entry_label_wrist.png'
    if not image_path.is_file():
        print(f'missing image: {image_path}', file=sys.stderr)
        return 2

    return run_server(
        session,
        image_path,
        args.also_global,
        args.port,
        open_browser=not args.no_browser,
    )


if __name__ == '__main__':
    raise SystemExit(main())
