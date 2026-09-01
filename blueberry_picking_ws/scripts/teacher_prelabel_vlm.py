#!/usr/bin/env python3
"""VLM teacher pseudo-labels for blueberry CLUSTERS on fixed mono images.

Supports:
  --provider ollama     Qwen2.5-VL local via Ollama (default)
  --provider openai     OpenAI-compatible API (DashScope, vLLM, etc.)

Output: YOLO txt labels class_id=1 (berry_cluster) by default for unified model.

Usage:
  python3 scripts/teacher_prelabel_vlm.py --images datasets/fixed_mono/images
  python3 scripts/teacher_prelabel_vlm.py --provider openai \\
      --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
      --model qwen-vl-max --api-key-env DASHSCOPE_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CALIBRATION = os.path.join(ROOT, 'datasets', 'fixed_mono', 'vlm_calibration.json')


def _load_calibration(path: str) -> dict:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f'calibration required: {path} — run scripts/calibrate_vlm_cluster_prompt.py')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _cal_half_sizes(cal: dict, img_w: int, img_h: int) -> Tuple[int, int]:
    ref_w, ref_h = cal.get('image_size', [640, 480])
    sx = img_w / max(ref_w, 1)
    sy = img_h / max(ref_h, 1)
    bp = cal['box_px']
    hw = max(18, int(bp['pw']['median'] * sx * 0.5))
    hh = max(18, int(bp['ph']['median'] * sy * 0.5))
    return hw, hh


def _build_cluster_prompt(img_w: int, img_h: int, cal: dict) -> str:
    """Single prompt: center_xy + human-label calibration for box expansion."""
    cpi = cal['clusters_per_image']
    bp = cal['box_px']
    ref_w, ref_h = cal.get('image_size', [640, 480])
    sx = img_w / max(ref_w, 1)
    sy = img_h / max(ref_h, 1)
    pw_med = int(bp['pw']['median'] * sx)
    ph_med = int(bp['ph']['median'] * sy)
    area_med = cal['box_norm']['area_frac']['median']
    return f"""\
You mark blueberry **cluster centers** for a fixed overhead camera (robot picking).

IMAGE: {img_w} x {img_h} px, origin top-left.

## Scene
- Ripe berries: small **dark blue/purple round dots** on a potted decorative plant.
- One **cluster** = ≥3 berries on the **same short branch fork**.
- Ignore: pot, leaves, stems alone, robot arm, people, background.

## Task — center points ONLY

Human labels: **{int(cpi['min'])}–{int(cpi['max'])} clusters/image** (median {cpi['median']:.0f}).
Each separate branch fork → one center_xy. Never pad to a count. Never one center for whole plant.
We expand each center to ~{pw_med}x{ph_med}px (area ~{area_med*100:.1f}% of image).

## Output — JSON only

{{
  "clusters": [
    {{"center_xy": [x, y], "confidence": 0.85}}
  ]
}}

- center_xy: integer pixel (x, y). Empty: {{"clusters": []}}
"""


def _list_images(image_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def _encode_jpeg_b64(rgb: np.ndarray, max_width: int = 1280, quality: int = 85) -> str:
    h, w = rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        rgb = cv2.resize(rgb, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('jpeg encode failed')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _parse_json_blob(text: str) -> Optional[dict]:
    text = text.strip()
    if '```' in text:
        m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        if m:
            text = m.group(1)
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_bbox(raw, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    vals = [float(v) for v in raw[:4]]

    def _clamp_box(x0: int, y0: int, x1: int, y1: int) -> Optional[Tuple[int, int, int, int]]:
        x0, x1 = sorted((max(0, x0), min(w - 1, x1)))
        y0, y1 = sorted((max(0, y0), min(h - 1, y1)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return x0, y0, x1, y1

    # 0–1 normalized
    if all(0.0 <= v <= 1.0 for v in vals):
        x0 = int(vals[0] * w)
        y0 = int(vals[1] * h)
        x1 = int(vals[2] * w)
        y1 = int(vals[3] * h)
        return _clamp_box(x0, y0, x1, y1)

    # Pixel coords — prefer when they fit image bounds (common VLM output).
    px = [int(round(v)) for v in vals]
    if (px[2] <= w and px[3] <= h and px[0] >= 0 and px[1] >= 0
            and px[2] > px[0] and px[3] > px[1]):
        return _clamp_box(px[0], px[1], px[2], px[3])

    # Qwen 0–1000 normalized (only when pixel interpretation invalid).
    if all(0 <= v <= 1000 for v in vals):
        x0 = int(vals[0] / 1000.0 * w)
        y0 = int(vals[1] / 1000.0 * h)
        x1 = int(vals[2] / 1000.0 * w)
        y1 = int(vals[3] / 1000.0 * h)
        return _clamp_box(x0, y0, x1, y1)

    return _clamp_box(px[0], px[1], px[2], px[3])


def _item_to_bbox(
    item: dict, w: int, h: int, *, default_half: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Parse cluster dict → pixel bbox (supports center_xy or bbox_xyxy)."""
    if 'center_xy' in item:
        c = item['center_xy']
        hs = item.get('half_size_px') or item.get('half_size')
        if hs is None and default_half is not None:
            hs = list(default_half)
        if hs is None:
            hs = [50, 50]
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            return None
        cx, cy = int(round(float(c[0]))), int(round(float(c[1])))
        hs_list = hs if isinstance(hs, (list, tuple)) else [hs, hs]
        hw = int(round(float(hs_list[0])))
        hh = int(round(float(hs_list[1] if len(hs_list) > 1 else hs_list[0])))
        hw = max(18, min(hw, int(w * 0.15)))
        hh = max(18, min(hh, int(h * 0.15)))
        return _normalize_bbox([cx - hw, cy - hh, cx + hw, cy + hh], w, h)

    box = item.get('bbox_xyxy') or item.get('bbox_2d') or item.get('bbox')
    if box is None:
        return None
    return _normalize_bbox(box, w, h)


def _bbox_area_frac(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> float:
    return float((x1 - x0) * (y1 - y0)) / float(max(w * h, 1))


def _nms_boxes(
    boxes: List[Tuple[int, int, int, int, float]],
    iou_thresh: float = 0.45,
) -> List[Tuple[int, int, int, int, float]]:
    """Greedy NMS by confidence."""
    if not boxes:
        return []
    arr = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: List[Tuple[int, int, int, int, float]] = []

    def iou(a, b) -> float:
        ax0, ay0, ax1, ay1 = a[:4]
        bx0, by0, bx1, by1 = b[:4]
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
        area_b = max(1, (bx1 - bx0) * (by1 - by0))
        return inter / float(area_a + area_b - inter)

    for cand in arr:
        if all(iou(cand, k) < iou_thresh for k in kept):
            kept.append(cand)
    return kept


def _filter_cluster_boxes(
    boxes: List[Tuple[int, int, int, int, float]],
    w: int,
    h: int,
    *,
    min_area_frac: float,
    max_area_frac: float,
    min_side_px: int,
    max_side_frac: float,
) -> List[Tuple[int, int, int, int, float]]:
    out = []
    for x0, y0, x1, y1, conf in boxes:
        bw, bh = x1 - x0, y1 - y0
        if bw < min_side_px or bh < min_side_px:
            continue
        if bw > w * max_side_frac or bh > h * max_side_frac:
            continue
        frac = _bbox_area_frac(x0, y0, x1, y1, w, h)
        if frac < min_area_frac or frac > max_area_frac:
            continue
        out.append((x0, y0, x1, y1, conf))
    return out


def _bbox_to_yolo_line(x0: int, y0: int, x1: int, y1: int, w: int, h: int, class_id: int) -> str:
    cx = (x0 + x1) * 0.5 / w
    cy = (y0 + y1) * 0.5 / h
    bw = max(x1 - x0, 1) / w
    bh = max(y1 - y0, 1) / h
    return f'{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'


class VlmClusterTeacher:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._provider = args.provider
        self._calibration = _load_calibration(getattr(args, 'calibration', '') or DEFAULT_CALIBRATION)
        print(f'[teacher] calibration: {args.calibration or DEFAULT_CALIBRATION}')

    def detect_clusters(self, rgb: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        h, w = rgb.shape[:2]
        raw = self._call_vlm(rgb, w, h)
        parsed = _parse_json_blob(raw)
        if parsed is None:
            raise RuntimeError(f'VLM returned unparseable JSON: {raw[:200]}')
        half = _cal_half_sizes(self._calibration, w, h)
        clusters = parsed.get('clusters') or []
        out: List[Tuple[int, int, int, int, float]] = []
        for item in clusters:
            if not isinstance(item, dict):
                continue
            bb = _item_to_bbox(item, w, h, default_half=half)
            if bb is None:
                continue
            conf = float(item.get('confidence', 0.8))
            out.append((*bb, conf))
        out = _filter_cluster_boxes(
            out, w, h,
            min_area_frac=float(self._args.min_box_area_frac),
            max_area_frac=float(self._args.max_box_area_frac),
            min_side_px=int(self._args.min_box_side_px),
            max_side_frac=float(self._args.max_box_side_frac),
        )
        return _nms_boxes(out, iou_thresh=float(self._args.nms_iou))

    def _call_vlm(self, rgb: np.ndarray, img_w: int, img_h: int) -> str:
        if self._provider == 'ollama':
            return self._call_ollama(rgb, img_w, img_h)
        return self._call_openai(rgb, img_w, img_h)

    def _call_ollama(self, rgb: np.ndarray, img_w: int, img_h: int) -> str:
        import ollama

        client = ollama.Client(host=self._args.ollama_host)
        b64 = _encode_jpeg_b64(rgb, self._args.max_img_width, self._args.jpeg_quality)
        prompt = _build_cluster_prompt(img_w, img_h, self._calibration)
        resp = client.chat(
            model=self._args.model,
            messages=[{'role': 'user', 'content': prompt, 'images': [b64]}],
            options={'temperature': 0.02, 'num_predict': 1024},
        )
        return str(resp['message']['content'])

    def _call_openai(self, rgb: np.ndarray, img_w: int, img_h: int) -> str:
        from openai import OpenAI

        api_key = os.environ.get(self._args.api_key_env, '')
        if not api_key:
            raise RuntimeError(f'set env {self._args.api_key_env} for API key')
        client = OpenAI(api_key=api_key, base_url=self._args.base_url or None)
        b64 = _encode_jpeg_b64(rgb, self._args.max_img_width, self._args.jpeg_quality)
        url = f'data:image/jpeg;base64,{b64}'
        prompt = _build_cluster_prompt(img_w, img_h, self._calibration)
        resp = client.chat.completions.create(
            model=self._args.model,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': url}},
                ],
            }],
            temperature=0.02,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ''


def _draw_preview(rgb: np.ndarray, boxes: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
    out = rgb.copy()
    for x0, y0, x1, y1, conf in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(out, f'cluster {conf:.2f}', (x0, max(y0 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', default=os.path.join(ROOT, 'datasets', 'fixed_mono', 'images'))
    parser.add_argument('--labels', default=os.path.join(ROOT, 'datasets', 'fixed_mono', 'labels'))
    parser.add_argument('--preview-dir', default=os.path.join(ROOT, 'datasets', 'fixed_mono', 'previews'))
    parser.add_argument('--provider', choices=('ollama', 'openai'), default='ollama')
    parser.add_argument('--model', default='qwen2.5vl:7b')
    parser.add_argument('--ollama-host', default='http://localhost:11434')
    parser.add_argument('--base-url', default='')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--calibration', default=DEFAULT_CALIBRATION,
                        help='vlm_calibration.json (required)')
    parser.add_argument('--class-id', type=int, default=1,
                        help='YOLO class for cluster (1 in unified model; human annotate uses 0)')
    parser.add_argument('--min-confidence', type=float, default=0.55)
    parser.add_argument('--min-box-area-frac', type=float, default=0.0045)
    parser.add_argument('--max-box-area-frac', type=float, default=0.035)
    parser.add_argument('--min-box-side-px', type=int, default=25)
    parser.add_argument('--max-box-side-frac', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.45)
    parser.add_argument('--max-img-width', type=int, default=1280)
    parser.add_argument('--jpeg-quality', type=int, default=85)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.labels, exist_ok=True)
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)

    paths = _list_images(args.images)
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        print(f'ERROR: no images in {args.images}', file=sys.stderr)
        return 1

    teacher = VlmClusterTeacher(args)
    ok = 0
    for i, path in enumerate(paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        lab_path = os.path.join(args.labels, f'{stem}.txt')
        if os.path.isfile(lab_path) and not args.overwrite:
            print(f'[skip] {stem}')
            continue

        bgr = cv2.imread(path)
        if bgr is None:
            print(f'[warn] unreadable {path}', file=sys.stderr)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        t0 = time.time()
        try:
            boxes = teacher.detect_clusters(rgb)
        except Exception as exc:
            print(f'[error] {stem}: {exc}', file=sys.stderr)
            continue
        elapsed = time.time() - t0

        lines = []
        kept = []
        for x0, y0, x1, y1, conf in boxes:
            if conf < args.min_confidence:
                continue
            lines.append(_bbox_to_yolo_line(x0, y0, x1, y1, w, h, args.class_id))
            kept.append((x0, y0, x1, y1, conf))

        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))

        if args.preview_dir:
            prev = _draw_preview(rgb, kept)
            cv2.imwrite(os.path.join(args.preview_dir, f'{stem}_vlm.jpg'),
                        cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))

        print(f'[{i + 1}/{len(paths)}] {stem}: {len(kept)} clusters ({elapsed:.1f}s)')
        ok += 1

    print(f'\n[teacher] labeled {ok}/{len(paths)} → {args.labels}')
    return 0 if ok > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
