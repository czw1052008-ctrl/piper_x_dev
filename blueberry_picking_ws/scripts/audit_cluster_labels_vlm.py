#!/usr/bin/env python3
"""VLM semantic audit: validate human cluster bboxes (yes/no per box).

Uses crop + full-context overlay. Does NOT re-detect; only judges existing boxes.

Usage:
  python3 scripts/audit_cluster_labels_vlm.py
  python3 scripts/audit_cluster_labels_vlm.py --labels datasets/fixed_mono/labels_gt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

AUDIT_PROMPT = """\
You audit ONE human-drawn bounding box on a fixed overhead camera image (potted blueberry plant).

The **green rectangle** marks the annotation under review.

## Valid "berry cluster" (should be YES)
- ≥3 **ripe dark blue/purple** round berries on the **same short branch fork**
- Box should tightly wrap that fork's berries (small margin OK)

## Invalid (should be NO)
- Whole plant / large branch / pot / leaves only
- Single berry or only 1–2 berries
- Box mostly empty or wrong object (stem, decoration, arm, person)
- Box far too loose (covers multiple separate forks)

## Output — JSON only
{
  "is_valid_cluster": true,
  "confidence": 0.9,
  "issue": "none",
  "note": "brief reason"
}

issue one of: none, not_cluster, too_few_berries, too_loose, too_tight, wrong_object, empty
"""


def _encode_jpeg_bgr(bgr: np.ndarray, max_width: int = 1280, quality: int = 88) -> str:
    h, w = bgr.shape[:2]
    if w > max_width:
        s = max_width / w
        bgr = cv2.resize(bgr, (max_width, int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('jpeg encode failed')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    if '```' in text:
        m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        if m:
            text = m.group(1)
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _yolo_to_xyxy(line: str, w: int, h: int) -> Tuple[int, int, int, int]:
    parts = line.strip().split()
    cx, cy, bw, bh = map(float, parts[1:5])
    x0 = int((cx - bw / 2) * w)
    y0 = int((cy - bh / 2) * h)
    x1 = int((cx + bw / 2) * w)
    y1 = int((cy + bh / 2) * h)
    return max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)


def _draw_audit_frame(bgr: np.ndarray, box: Tuple[int, int, int, int], idx: int) -> np.ndarray:
    vis = bgr.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.putText(vis, f'audit #{idx}', (x0, max(y0 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def _call_ollama(b64: str, prompt: str, model: str, host: str) -> str:
    import ollama
    client = ollama.Client(host=host)
    resp = client.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt, 'images': [b64]}],
        options={'temperature': 0.0, 'num_predict': 256},
    )
    return str(resp['message']['content'])


class VlmLabelAuditor:
    def __init__(self, model: str, host: str) -> None:
        self._model = model
        self._host = host

    def audit_box(self, bgr: np.ndarray, box: Tuple[int, int, int, int], idx: int) -> dict:
        vis = _draw_audit_frame(bgr, box, idx)
        b64 = _encode_jpeg_bgr(vis)
        raw = _call_ollama(b64, AUDIT_PROMPT, self._model, self._host)
        parsed = _parse_json(raw) or {}
        parsed['_raw'] = raw[:400]
        return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', default=os.path.join(ROOT, 'datasets', 'fixed_mono', 'images'))
    parser.add_argument('--labels', default=os.path.join(ROOT, 'datasets', 'fixed_mono', 'labels_gt'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'datasets', 'fixed_mono', 'vlm_audit_report.json'))
    parser.add_argument('--preview-dir', default=os.path.join(ROOT, 'datasets', 'fixed_mono', 'audit_previews'))
    parser.add_argument('--model', default='qwen2.5vl:7b')
    parser.add_argument('--ollama-host', default='http://localhost:11434')
    args = parser.parse_args()

    os.makedirs(args.preview_dir, exist_ok=True)
    auditor = VlmLabelAuditor(args.model, args.ollama_host)

    box_results: List[dict] = []
    image_results: List[dict] = []
    n_valid = n_invalid = n_unclear = 0

    paths = sorted(glob(os.path.join(args.images, '*')))
    paths = [p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_path in paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lab_path = os.path.join(args.labels, f'{stem}.txt')
        if not os.path.isfile(lab_path):
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        lines = [ln for ln in open(lab_path, encoding='utf-8') if ln.strip()]
        boxes = [_yolo_to_xyxy(ln, w, h) for ln in lines]

        img_entry = {'stem': stem, 'n_boxes': len(boxes), 'boxes': []}
        for bi, box in enumerate(boxes, 1):
            t0 = time.time()
            try:
                verdict = auditor.audit_box(bgr, box, bi)
            except Exception as exc:
                verdict = {'is_valid_cluster': None, 'issue': 'error', 'note': str(exc)}
            elapsed = time.time() - t0

            valid = verdict.get('is_valid_cluster')
            if valid is True:
                n_valid += 1
                flag = 'PASS'
            elif valid is False:
                n_invalid += 1
                flag = 'FAIL'
            else:
                n_unclear += 1
                flag = 'UNCLEAR'

            entry = {
                'stem': stem, 'box_idx': bi, 'bbox_xyxy': list(box),
                'flag': flag,
                'is_valid_cluster': valid,
                'confidence': verdict.get('confidence'),
                'issue': verdict.get('issue', ''),
                'note': verdict.get('note', ''),
                'elapsed_s': round(elapsed, 2),
            }
            box_results.append(entry)
            img_entry['boxes'].append(entry)

            prev = _draw_audit_frame(bgr, box, bi)
            color = (0, 255, 0) if flag == 'PASS' else (0, 0, 255) if flag == 'FAIL' else (0, 165, 255)
            cv2.putText(prev, f'{flag} {verdict.get("issue", "")}', (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imwrite(os.path.join(args.preview_dir, f'{stem}_box{bi:02d}_{flag}.jpg'), prev)
            print(f'  [{stem} #{bi}] {flag} conf={verdict.get("confidence")} '
                  f'issue={verdict.get("issue")} — {verdict.get("note", "")[:60]}')

        image_results.append(img_entry)

    total = n_valid + n_invalid + n_unclear
    summary = {
        'n_images': len(image_results),
        'n_boxes': total,
        'pass': n_valid,
        'fail': n_invalid,
        'unclear': n_unclear,
        'pass_rate': n_valid / max(total, 1),
        'per_box': box_results,
        'per_image': image_results,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\n[audit] boxes={total} PASS={n_valid} FAIL={n_invalid} UNCLEAR={n_unclear}')
    print(f'[audit] pass_rate={summary["pass_rate"]:.1%}')
    print(f'[audit] report → {args.out}')
    print(f'[audit] previews → {args.preview_dir}')
    if n_invalid:
        print('\nFailed boxes:')
        for e in box_results:
            if e['flag'] == 'FAIL':
                print(f"  {e['stem']} #{e['box_idx']}: {e['issue']} — {e['note']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
