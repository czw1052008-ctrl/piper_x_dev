#!/usr/bin/env python3
"""Offline REFINING step replay: QA snaps + step JSON / FSM log → HTML + plots.

Example:
  python3 scripts/refine_replay.py \\
    --qa-dir log/real_robot/qa/20260805_183607 \\
    --fsm-log log/real_robot/reach_fsm.log \\
    --out log/real_robot/refine_tune_viz/replay_20260805_183607
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# FK for wrist-cam origin in base_link (same model as FSM).
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
try:
    from piper_position_ik import fk_link6_T  # type: ignore
except Exception:  # pragma: no cover
    fk_link6_T = None  # type: ignore

# Static mount used on real robot (matches real_robot.env / tf publisher).
_T_LINK6_CAM = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, -0.08),
    (0.0, 0.0, 1.0, -0.04),
    (0.0, 0.0, 0.0, 1.0),
)


SERVO_RE = re.compile(
    r'REFINING servo\[(\d+)\]:\s*'
    r'(?:(?:cartesian_ik|position_diff_ik)(?P<est0>\s+est)?\s+)?'
    r'z_cam=(?P<z_cam>-?[\d.]+)\s+'
    r'dist_cup=(?P<dist_cup>-?[\d.]+)\s+'
    r'(?:step_m=(?P<step_m>-?[\d.]+)\s+)?'
    r'(?P<est>\s+est)?\s*'
    r'pix_err=(?P<pix_err>-?[\d.]+)\s+'
    r'err_yaw_deg=(?P<err_yaw>-?[\d.]+)\s+'
    r'err_pitch_deg=(?P<err_pitch>-?[\d.]+)\s+'
    r'reach_scale=(?P<reach_scale>-?[\d.]+)'
    r'(?:\s+tgt=\((?P<tx>[+\-]?\d+\.?\d*),(?P<ty>[+\-]?\d+\.?\d*),(?P<tz>[+\-]?\d+\.?\d*)\))?'
    r'(?:\s+Δj1=(?P<dj1>[+\-][\d.]+)deg\s+'
    r'Δj2=(?P<dj2>[+\-][\d.]+)deg\s+'
    r'Δj3=(?P<dj3>[+\-][\d.]+)deg\s+'
    r'Δj5=(?P<dj5>[+\-][\d.]+)deg\s+'
    r'→ j=\((?P<j1>[+\-]?\d+\.?\d*),(?P<j2>[+\-]?\d+\.?\d*),'
    r'(?P<j3>[+\-]?\d+\.?\d*),(?P<j5>[+\-]?\d+\.?\d*)\))?'
)

FB_RE = re.compile(
    r'REFINING servo step OK fb_j=\('
    r'(?P<j1>[+\-]?\d+\.?\d*),(?P<j2>[+\-]?\d+\.?\d*),'
    r'(?P<j3>[+\-]?\d+\.?\d*),(?P<j5>[+\-]?\d+\.?\d*)\)'
)

LOCK_RE = re.compile(
    r'REFINING fruit lock \(nearest cup\):\s*'
    r'track_id=(?P<tid>-?\d+)\s+'
    r'conf=(?P<conf>[\d.]+)\s+'
    r'base=\((?P<bx>[+\-]?\d+\.?\d*),(?P<by>[+\-]?\d+\.?\d*),(?P<bz>[+\-]?\d+\.?\d*)\)\s+'
    r'cup_dist=(?P<cup>-?[\d.]+)\s+'
    r'z_cam=(?P<z_cam>-?[\d.]+)'
)

QA_SNAP_RE = re.compile(r'QA snap (?P<sess>[^/]+)/servo_(?P<step>\d+)_after')


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _xyz3(v: Any) -> Optional[List[float]]:
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return None
    try:
        return [float(v[0]), float(v[1]), float(v[2])]
    except (TypeError, ValueError):
        return None


def _mat4_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(float(a[i][k]) * float(b[k][j]) for k in range(4))
    return out


def _cam_origin_base(q_rad: Sequence[float]) -> Optional[List[float]]:
    if fk_link6_T is None:
        return None
    try:
        T_l6 = fk_link6_T(q_rad)
        # numpy ndarray or list-like 4x4
        rows = T_l6.tolist() if hasattr(T_l6, 'tolist') else T_l6
        T = _mat4_mul(rows, _T_LINK6_CAM)
        return [float(T[0][3]), float(T[1][3]), float(T[2][3])]
    except Exception:
        return None


def _joint_rad_for_frame(
    fr: Dict[str, Any],
    step: Dict[str, Any],
    *,
    frame_i: int,
    n_frames: int,
) -> Optional[List[float]]:
    jd = fr.get('joints_deg') or {}
    if not isinstance(jd, dict):
        return None
    tj = step.get('target_j_deg') if isinstance(step.get('target_j_deg'), dict) else {}
    dj = step.get('delta_j_deg') if isinstance(step.get('delta_j_deg'), dict) else {}
    frac = float(frame_i) / float(max(n_frames - 1, 1))

    def jdeg(name: str) -> Optional[float]:
        if name in jd and jd[name] is not None:
            return float(jd[name])
        # Motion snaps often omit j4 — interpolate from plan.
        if name == 'j4' and tj.get('j4') is not None:
            end = float(tj['j4'])
            start = end - float(dj.get('j4') or 0.0)
            return start + frac * (end - start)
        return None

    vals = [jdeg(n) for n in ('j1', 'j2', 'j3', 'j4', 'j5')]
    if any(v is None for v in vals[:3]) or vals[4] is None:
        return None
    if vals[3] is None:
        vals[3] = 0.0
    return [math.radians(float(v)) for v in vals] + [0.0]


def _linspace_xyz(a: List[float], b: List[float], n: int = 24) -> Dict[str, List[float]]:
    n = max(2, int(n))
    xs, ys, zs = [], [], []
    for i in range(n):
        t = i / (n - 1)
        xs.append(a[0] + t * (b[0] - a[0]))
        ys.append(a[1] + t * (b[1] - a[1]))
        zs.append(a[2] + t * (b[2] - a[2]))
    return {'x': xs, 'y': ys, 'z': zs}


def _build_step_scene3d(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Abstract base_link scene: cup tip, wrist cam, planned path, berry."""
    berry = _xyz3(step.get('berry_base'))
    tcp0 = _xyz3(step.get('tcp_base'))
    if tcp0 is None:
        mf0 = (step.get('motion_frames') or [{}])
        if mf0:
            tcp0 = _xyz3(mf0[0].get('tcp_base'))
    tcp_goal = _xyz3(step.get('tcp_goal'))
    if tcp_goal is None and tcp0 is not None:
        d = _xyz3(step.get('cmd_d_base_m') or step.get('cmd_dxyz'))
        if d is not None:
            tcp_goal = [tcp0[0] + d[0], tcp0[1] + d[1], tcp0[2] + d[2]]

    frames = step.get('motion_frames') or []
    cup_x, cup_y, cup_z = [], [], []
    cam_x, cam_y, cam_z = [], [], []
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            continue
        tcp = _xyz3(fr.get('tcp_base'))
        if tcp is not None:
            cup_x.append(tcp[0]); cup_y.append(tcp[1]); cup_z.append(tcp[2])
        q = _joint_rad_for_frame(fr, step, frame_i=i, n_frames=max(len(frames), 1))
        if q is not None:
            cam = _cam_origin_base(q)
            if cam is not None:
                cam_x.append(cam[0]); cam_y.append(cam[1]); cam_z.append(cam[2])

    if berry is None and tcp0 is None and not cup_x:
        return None

    planned = None
    if tcp0 is not None and tcp_goal is not None:
        planned = _linspace_xyz(tcp0, tcp_goal, n=32)

    return {
        'frame': 'base_link',
        'step': int(step.get('step', -1)),
        'control': step.get('control'),
        'ik_method': step.get('ik_method'),
        'method': step.get('method'),
        'berry': berry,
        'tcp_start': tcp0,
        'tcp_goal': tcp_goal,
        'planned_cup_path': planned,
        'executed_cup_path': {'x': cup_x, 'y': cup_y, 'z': cup_z} if cup_x else None,
        'executed_cam_path': {'x': cam_x, 'y': cam_y, 'z': cam_z} if cam_x else None,
        'dist_cup': _f(step.get('dist_cup')),
        'dist_cup_after': _f(step.get('dist_cup_after')),
        'travel_m': _f(step.get('travel_m')),
        'n_motion': len(frames),
    }


def _build_scene3d(timeline: Dict[str, Any]) -> Dict[str, Any]:
    steps_out: List[Dict[str, Any]] = []
    for s in timeline.get('steps') or []:
        sc = _build_step_scene3d(s)
        if sc is not None:
            steps_out.append(sc)
    return {
        'session': timeline.get('session'),
        'frame': 'base_link',
        'units': 'm',
        'notes': (
            'cup=TCP tip; cam=wrist color optical origin via FK(link6)×T_link6_cam; '
            'planned=tcp_start→tcp_goal; executed=motion_frames tcp / cam'
        ),
        'T_link6_cam_xyz': [0.0, -0.08, -0.04],
        'steps': steps_out,
    }


def _scan_images(qa_dir: Path) -> Dict[int, Dict[str, str]]:
    """Map step -> relative image paths (wrist, fine_viz, fixed, global_viz)."""
    by_step: Dict[int, Dict[str, str]] = {}
    for p in sorted(qa_dir.glob('servo_*_after_*.png')):
        m = re.match(r'servo_(\d+)_after_(.+)\.png$', p.name)
        if not m:
            continue
        step = int(m.group(1))
        kind = m.group(2)
        by_step.setdefault(step, {})[kind] = p.name
    return by_step


def _scan_motion_from_disk(qa_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Fallback: discover servo_XX_motion_YY_{wrist,fine_viz}.png if JSON missing list."""
    by_step: Dict[int, Dict[int, Dict[str, str]]] = {}
    for p in sorted(qa_dir.glob('servo_*_motion_*_*.png')):
        m = re.match(r'servo_(\d+)_motion_(\d+)_(.+)\.png$', p.name)
        if not m:
            continue
        step, mi, kind = int(m.group(1)), int(m.group(2)), m.group(3)
        by_step.setdefault(step, {}).setdefault(mi, {})[kind] = p.name
    out: Dict[int, List[Dict[str, Any]]] = {}
    for step, frames in by_step.items():
        lst = []
        for mi in sorted(frames):
            fr = frames[mi]
            lst.append({
                'i': mi,
                'wrist': fr.get('wrist'),
                'fine_viz': fr.get('fine_viz'),
                'fixed': fr.get('fixed'),
                'global_viz': fr.get('global_viz'),
                'tag': f'servo_{step:02d}_motion_{mi:02d}',
            })
        out[step] = lst
    return out


def _load_step_jsons(qa_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for p in sorted(qa_dir.glob('servo_*.json')):
        m = re.match(r'servo_(\d+)\.json$', p.name)
        if not m:
            continue
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        out[int(m.group(1))] = data
    return out


def _load_lock_json(qa_dir: Path) -> Optional[Dict[str, Any]]:
    p = qa_dir / 'refine_fruit_lock.json'
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _load_depth_qa_json(qa_dir: Path) -> Optional[Dict[str, Any]]:
    p = qa_dir / 'center_depth_qa.json'
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _scan_depth_qa_images(qa_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kind in ('wrist', 'fine_viz', 'fixed', 'global_viz',
                 'cup_overlay', 'cup_rel', 'sidebyside'):
        p = qa_dir / f'center_depth_qa_{kind}.png'
        if p.is_file():
            out[kind] = p.name
    return out


def _load_event_meta(qa_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tag in ('optical_fresh_lock', 'probe_tri_contact', 'near_handoff_pending'):
        p = qa_dir / f'{tag}_meta.json'
        if p.is_file():
            try:
                out[tag] = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                pass
    return out


def _infer_event_meta(
    qa_dir: Path, tag: str, timeline: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Best-effort meta for sessions recorded before event meta JSON existed."""
    steps = timeline.get('steps') or []
    center_step = next(
        (s for s in steps if s.get('refine_phase') == 'center_oneshot' or s.get('control') == 'center_oneshot_cart'),
        None,
    )
    nh_path = qa_dir / 'near_handoff_request.json'
    nh: Dict[str, Any] = {}
    if nh_path.is_file():
        try:
            nh = json.loads(nh_path.read_text(encoding='utf-8'))
        except Exception:
            nh = {}

    w, h = 640, 480
    if center_step:
        imgs = center_step.get('images') or {}
        wrist_name = imgs.get('wrist') or ''
        wrist_path = qa_dir / wrist_name if wrist_name else qa_dir / f'{tag}_wrist.png'
        if wrist_path.is_file():
            try:
                import cv2  # type: ignore
                im = cv2.imread(str(wrist_path))
                if im is not None:
                    h, w = im.shape[:2]
            except Exception:
                pass

    optical_uv = [w * 0.5, h * 0.5]
    berry_uv_reproject = None
    if center_step and center_step.get('berry_uv_after'):
        berry_uv_reproject = center_step['berry_uv_after']
    src_after = center_step.get('berry_uv_after_src') if center_step else None

    if tag == 'optical_fresh_lock':
        lock = timeline.get('lock') or {}
        return {
            'source': 'yolo_fresh_lock (inferred — no snap meta)',
            'track_id': lock.get('track_id'),
            'berry_uv_yolo': lock.get('berry_uv'),
            'optical_uv': optical_uv,
            'inferred': True,
        }

    if tag in ('probe_tri_contact', 'near_handoff_pending'):
        source = 'probe_tri_reproject (inferred)'
        if src_after == 'yolo_miss_at_center':
            source = 'probe_tri_reproject (YOLO miss at center)'
        if tag == 'near_handoff_pending' and nh:
            source = 'near_handoff_probe_tri (inferred)'
        return {
            'source': source,
            'track_id': None,
            'berry_uv_reproject': berry_uv_reproject,
            'optical_uv': optical_uv,
            'z_cam_m': nh.get('z_cam_m'),
            'fruit_anchor': nh.get('fruit_anchor'),
            'dist_cup_m': nh.get('dist_cup_m'),
            'inferred': True,
        }
    return None


def _annotate_event_wrist_offline(
    qa_dir: Path, tag: str, meta: Dict[str, Any], out_path: Path,
) -> bool:
    """Draw source markers on saved wrist PNG for replay (offline)."""
    wrist = qa_dir / f'{tag}_wrist.png'
    if not wrist.is_file():
        return False
    try:
        import cv2  # type: ignore
        import math as _math
    except ImportError:
        return False
    img_bgr = cv2.imread(str(wrist))
    if img_bgr is None:
        return False
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    optical = meta.get('optical_uv') or [w * 0.5, h * 0.5]
    ox, oy = int(optical[0]), int(optical[1])
    cv2.drawMarker(
        img, (ox, oy), (160, 160, 160),
        markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)

    source = str(meta.get('source') or tag)
    berry_uv_yolo = meta.get('berry_uv_yolo')
    if berry_uv_yolo and len(berry_uv_yolo) >= 2:
        u, v = int(berry_uv_yolo[0]), int(berry_uv_yolo[1])
        cv2.drawMarker(
            img, (u, v), (80, 255, 120),
            markerType=cv2.MARKER_DIAMOND, markerSize=24, thickness=2)
        cv2.rectangle(img, (u - 28, v - 28), (u + 28, v + 28), (80, 255, 120), 2)
        tid = meta.get('track_id')
        lbl = f'FSM YOLO LOCK T{tid}' if tid is not None else 'FSM YOLO LOCK'
        cv2.putText(
            img, lbl, (u + 14, v + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 120), 2, cv2.LINE_AA)

    berry_uv_rep = meta.get('berry_uv_reproject')
    pix_off = None
    if berry_uv_rep and len(berry_uv_rep) >= 2:
        bu, bv = int(berry_uv_rep[0]), int(berry_uv_rep[1])
        cv2.drawMarker(
            img, (bu, bv), (255, 80, 200),
            markerType=cv2.MARKER_STAR, markerSize=26, thickness=2)
        cv2.putText(
            img, 'PROBE TRI 3D', (bu + 12, bv + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 200), 2, cv2.LINE_AA)
        pix_off = _math.hypot(bu - ox, bv - oy)
        cv2.arrowedLine(img, (ox, oy), (bu, bv), (255, 80, 200), 2, tipLength=0.12)

    for det in meta.get('live_yolo') or []:
        u, v = det.get('u'), det.get('v')
        if u is None or v is None:
            continue
        cv2.circle(img, (int(u), int(v)), 16, (255, 200, 0), 2)
        cv2.putText(
            img, f'YOLO T{det.get("track_id", "?")}', (int(u) + 14, int(v) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

    z_cam = meta.get('z_cam_m')
    cv2.putText(
        img, f'SOURCE: {source}', (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f'z_cam={(z_cam if z_cam is not None else -1):.3f}m  '
        f'pix_off={(pix_off if pix_off is not None else -1):.1f}px',
        (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1, cv2.LINE_AA)
    cv2.putText(
        img, 'grey=optical  green=YOLO lock  magenta=probe tri  yellow=live YOLO',
        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return True


def _scan_event_images(qa_dir: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for tag in ('optical_fresh_lock', 'probe_tri_contact', 'near_handoff_pending'):
        imgs: Dict[str, str] = {}
        for kind in ('annotated', 'sidebyside', 'fine_viz', 'wrist', 'fixed', 'global_viz'):
            p = qa_dir / f'{tag}_{kind}.png'
            if p.is_file():
                imgs[kind] = p.name
        if imgs:
            out[tag] = imgs
    return out


def _parse_fsm_log(
    log_path: Path, session: str,
) -> tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Parse lock + servo steps; prefer lines after QA snap for this session."""
    if not log_path.is_file():
        return None, {}

    text = log_path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    # Find first QA snap for this session to bound the run window.
    start_idx = 0
    for i, line in enumerate(lines):
        m = QA_SNAP_RE.search(line)
        if m and m.group('sess') == session:
            # Walk back a bit for fruit lock / anchors before first snap.
            start_idx = max(0, i - 40)
            break
        if f'QA snap {session}/' in line:
            start_idx = max(0, i - 40)
            break

    # End at next different session snap or end of file.
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        m = QA_SNAP_RE.search(lines[i])
        if m and m.group('sess') != session:
            end_idx = i
            break
        if 'QA snap ' in lines[i] and f'QA snap {session}/' not in lines[i]:
            # other session
            if re.search(r'QA snap \d{8}_\d{6}/', lines[i]):
                end_idx = i
                break

    window = lines[start_idx:end_idx]
    lock: Optional[Dict[str, Any]] = None
    steps: Dict[int, Dict[str, Any]] = {}
    pending_fb_step: Optional[int] = None

    for line in window:
        lm = LOCK_RE.search(line)
        if lm:
            lock = {
                'track_id': int(lm.group('tid')),
                'confidence': float(lm.group('conf')),
                'berry_base': [
                    float(lm.group('bx')), float(lm.group('by')), float(lm.group('bz')),
                ],
                'cup_dist': float(lm.group('cup')),
                'z_cam': float(lm.group('z_cam')),
                'source': 'fsm_log',
            }
            continue

        sm = SERVO_RE.search(line)
        if sm:
            step = int(sm.group(1))
            prev = steps.get(step, {'step': step})
            using_est = bool(sm.group('est') or sm.group('est0'))
            rec: Dict[str, Any] = {
                'step': step,
                'z_cam': float(sm.group('z_cam')),
                'dist_cup': float(sm.group('dist_cup')),
                'using_estimate': using_est or bool(prev.get('using_estimate')),
                'pix_err': float(sm.group('pix_err')),
                'err_yaw_deg': float(sm.group('err_yaw')),
                'err_pitch_deg': float(sm.group('err_pitch')),
                'reach_scale': float(sm.group('reach_scale')),
                'source': 'fsm_log',
            }
            if sm.group('step_m') is not None:
                rec['step_m'] = float(sm.group('step_m'))
            if sm.group('tx') is not None:
                rec['target_xyz'] = [
                    float(sm.group('tx')), float(sm.group('ty')), float(sm.group('tz')),
                ]
            if sm.group('dj1') is not None:
                rec['delta_j_deg'] = {
                    'j1': float(sm.group('dj1')),
                    'j2': float(sm.group('dj2')),
                    'j3': float(sm.group('dj3')),
                    'j5': float(sm.group('dj5')),
                }
                rec['target_j_deg'] = {
                    'j1': float(sm.group('j1')),
                    'j2': float(sm.group('j2')),
                    'j3': float(sm.group('j3')),
                    'j5': float(sm.group('j5')),
                }
            else:
                if 'delta_j_deg' in prev:
                    rec['delta_j_deg'] = prev['delta_j_deg']
                if 'target_j_deg' in prev:
                    rec['target_j_deg'] = prev['target_j_deg']
            if lock is not None:
                rec.setdefault('track_id', lock.get('track_id'))
                rec.setdefault('berry_base', lock.get('berry_base'))
            steps[step] = {**prev, **rec}
            pending_fb_step = step
            continue

        fm = FB_RE.search(line)
        if fm and pending_fb_step is not None:
            steps.setdefault(pending_fb_step, {'step': pending_fb_step})
            steps[pending_fb_step]['fb_j_deg'] = {
                'j1': float(fm.group('j1')),
                'j2': float(fm.group('j2')),
                'j3': float(fm.group('j3')),
                'j5': float(fm.group('j5')),
            }
            pending_fb_step = None

    return lock, steps


def _merge_timeline(
    qa_dir: Path,
    images: Dict[int, Dict[str, str]],
    json_steps: Dict[int, Dict[str, Any]],
    log_steps: Dict[int, Dict[str, Any]],
    lock: Optional[Dict[str, Any]],
    depth_qa: Optional[Dict[str, Any]] = None,
    depth_qa_images: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    disk_motion = _scan_motion_from_disk(qa_dir)
    step_ids = sorted(set(images) | set(json_steps) | set(log_steps) | set(disk_motion))
    steps_out: List[Dict[str, Any]] = []
    for s in step_ids:
        rec: Dict[str, Any] = {'step': s}
        # Prefer structured JSON, then log.
        if s in log_steps:
            rec.update(log_steps[s])
        if s in json_steps:
            rec.update(json_steps[s])
            rec['source'] = json_steps[s].get('source', 'step_json')
        if lock and 'track_id' not in rec:
            rec['track_id'] = lock.get('track_id')
        if lock and 'berry_base' not in rec:
            rec['berry_base'] = lock.get('berry_base')
        imgs = images.get(s, {})
        rec['images'] = imgs
        motion = rec.get('motion_frames') or disk_motion.get(s) or []
        rec['motion_frames'] = motion
        rec['n_motion_frames'] = len(motion)
        steps_out.append(rec)

    lock_images = {}
    for kind in ('wrist', 'fine_viz', 'fixed', 'global_viz', 'sidebyside',
                 'cup_overlay', 'cup_rel'):
        p = qa_dir / f'refine_fruit_lock_{kind}.png'
        if p.is_file():
            lock_images[kind] = p.name

    return {
        'qa_dir': str(qa_dir),
        'session': qa_dir.name,
        'lock': lock,
        'lock_images': lock_images,
        'event_images': _scan_event_images(qa_dir),
        'event_meta': _load_event_meta(qa_dir),
        'depth_qa': depth_qa,
        'depth_qa_images': depth_qa_images or {},
        'steps': steps_out,
        'n_steps': len(steps_out),
    }


def _write_plots(timeline: Dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'plots skipped (matplotlib): {exc}')
        return

    steps = timeline['steps']
    if not steps:
        return
    xs = [s['step'] for s in steps]

    def series(key: str, nested: Optional[str] = None) -> List[Optional[float]]:
        vals: List[Optional[float]] = []
        for s in steps:
            if nested:
                d = s.get(key) or {}
                vals.append(_f(d.get(nested)) if isinstance(d, dict) else None)
            else:
                vals.append(_f(s.get(key)))
        return vals

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    ax = axes[0]
    ax.plot(xs, series('pix_err'), 'o-', label='pix_err')
    ax.plot(xs, series('err_yaw_deg'), 's-', label='err_yaw_deg')
    ax.plot(xs, series('err_pitch_deg'), '^-', label='err_pitch_deg')
    ax.plot(xs, series('reach_scale'), 'd-', label='reach_scale')
    ax.set_ylabel('image / scale')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"REFINING replay {timeline['session']} (n={timeline['n_steps']})")

    ax = axes[1]
    ax.plot(xs, series('z_cam'), 'o-', label='z_cam')
    ax.plot(xs, series('dist_cup'), 's-', label='dist_cup')
    ax.plot(xs, series('dist_cup_after'), 's--', alpha=0.7, label='dist_cup_after')
    ax.set_ylabel('range (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # berry_rel components (list)
    def rel_comp(i: int, key: str = 'berry_rel_cup') -> List[Optional[float]]:
        vals: List[Optional[float]] = []
        for s in steps:
            r = s.get(key)
            if isinstance(r, (list, tuple)) and len(r) > i:
                vals.append(_f(r[i]))
            else:
                vals.append(None)
        return vals

    ax.plot(xs, rel_comp(0), 'o-', label='rel_x (cup→berry)')
    ax.plot(xs, rel_comp(1), 's-', label='rel_y')
    ax.plot(xs, rel_comp(2), '^-', label='rel_z')
    ax.axhline(0.0, color='#666', lw=0.8)
    ax.set_ylabel('berry−cup (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(xs, series('target_j_deg', 'j1'), 'o-', label='target j1')
    ax.plot(xs, series('target_j_deg', 'j5'), 's-', label='target j5')
    ax.plot(xs, series('fb_j_deg', 'j1'), 'o--', alpha=0.6, label='fb j1')
    ax.plot(xs, series('fb_j_deg', 'j5'), 's--', alpha=0.6, label='fb j5')
    ax.set_ylabel('joints (deg)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(xs, series('delta_j_deg', 'j1'), 'o-', label='Δj1')
    ax.plot(xs, series('delta_j_deg', 'j2'), 's-', label='Δj2')
    ax.plot(xs, series('delta_j_deg', 'j3'), '^-', label='Δj3')
    ax.plot(xs, series('delta_j_deg', 'j5'), 'd-', label='Δj5')
    ax.set_ylabel('command Δj (deg)')
    ax.set_xlabel('servo step')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _rel_img(name: Optional[str]) -> str:
    if not name:
        return ''
    base = _img_basename(name)
    return f'./images/{base}' if base else ''


def _img_basename(name: Optional[str]) -> str:
    if not name:
        return ''
    n = str(name).replace('\\', '/')
    if n.startswith('./images/'):
        n = n[len('./images/'):]
    elif n.startswith('images/'):
        n = n[len('images/'):]
    return n


def _write_html(timeline: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    # Copy images into out_dir/images and also inline them into HTML so file:// works.
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    qa = Path(timeline['qa_dir'])
    event_meta: Dict[str, Dict[str, Any]] = dict(timeline.get('event_meta') or {})
    event_images: Dict[str, Dict[str, str]] = dict(timeline.get('event_images') or {})
    for tag in ('optical_fresh_lock', 'probe_tri_contact', 'near_handoff_pending'):
        meta = event_meta.get(tag)
        if not meta:
            meta = _infer_event_meta(qa, tag, timeline)
            if meta:
                event_meta[tag] = meta
        ann_name = f'{tag}_annotated.png'
        ann_qa = qa / ann_name
        ann_out = img_dir / ann_name
        if ann_qa.is_file():
            shutil.copy2(ann_qa, ann_out)
        elif meta and (qa / f'{tag}_wrist.png').is_file():
            _annotate_event_wrist_offline(qa, tag, meta, ann_out)
        if ann_out.is_file():
            event_images.setdefault(tag, {})['annotated'] = ann_name
    timeline = dict(timeline)
    timeline['event_meta'] = event_meta
    timeline['event_images'] = event_images
    scene3d = _build_scene3d(timeline)
    timeline['scene3d'] = {
        'session': scene3d.get('session'),
        'frame': scene3d.get('frame'),
        'n_steps': len(scene3d.get('steps') or []),
    }
    (out_dir / 'scene3d.json').write_text(
        json.dumps(scene3d, indent=2, ensure_ascii=True), encoding='utf-8')

    names = set()
    for kind, name in (timeline.get('lock_images') or {}).items():
        if name:
            names.add(_img_basename(name))
    for _tag, imgs in (timeline.get('event_images') or {}).items():
        for name in imgs.values():
            if name:
                names.add(_img_basename(name))
    for kind, name in (timeline.get('depth_qa_images') or {}).items():
        if name:
            names.add(_img_basename(name))
    for s in timeline['steps']:
        for name in (s.get('images') or {}).values():
            if name:
                names.add(_img_basename(name))
        for fr in (s.get('motion_frames') or []):
            for k in ('wrist', 'fine_viz', 'fixed', 'global_viz'):
                if fr.get(k):
                    names.add(_img_basename(fr[k]))
    for name in names:
        src = qa / name
        if src.is_file():
            shutil.copy2(src, img_dir / name)

    embedded: Dict[str, str] = {}
    for name in names:
        src = qa / name
        if not src.is_file():
            continue
        suf = src.suffix.lower()
        mime = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }.get(suf)
        if not mime:
            continue
        try:
            raw = src.read_bytes()
            embedded[name] = f'data:{mime};base64,' + base64.b64encode(raw).decode('ascii')
        except Exception:
            continue

    def _asset_src(name: Optional[str]) -> str:
        base = _img_basename(name)
        if not base:
            return ''
        return embedded.get(base) or _rel_img(base)

    # Embed timeline JSON (images paths rewritten).
    tl = json.loads(json.dumps(timeline))
    for kind, name in list((tl.get('lock_images') or {}).items()):
        tl['lock_images'][kind] = _asset_src(name)
    for tag, imgs in list((tl.get('event_images') or {}).items()):
        tl['event_images'][tag] = {k: _asset_src(v) for k, v in imgs.items()}
    tl['event_meta'] = timeline.get('event_meta') or {}
    for kind, name in list((tl.get('depth_qa_images') or {}).items()):
        tl['depth_qa_images'][kind] = _asset_src(name)
    for s in tl['steps']:
        s['images'] = {k: _asset_src(v) for k, v in (s.get('images') or {}).items()}
        for fr in (s.get('motion_frames') or []):
            if fr.get('wrist'):
                fr['wrist'] = _asset_src(fr['wrist'])
            if fr.get('fine_viz'):
                fr['fine_viz'] = _asset_src(fr['fine_viz'])
            if fr.get('fixed'):
                fr['fixed'] = _asset_src(fr['fixed'])
            if fr.get('global_viz'):
                fr['global_viz'] = _asset_src(fr['global_viz'])

    data_js = json.dumps(tl, ensure_ascii=True, indent=2)
    scene3d_js = json.dumps(scene3d, ensure_ascii=True, indent=2)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<base href="./"/>
<title>REFINING replay {timeline['session']}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg: #12141a; --panel: #1c2030; --text: #e8eaef; --muted: #9aa3b5;
    --accent: #5ec8a0; --line: #2a3144;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text);
  }}
  header {{
    padding: 12px 18px; border-bottom: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  }}
  header h1 {{ margin: 0; font-size: 1.05rem; font-weight: 600; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; }}
  .openHint {{
    margin: 0 18px 10px; padding: 8px 12px; background: #2a2030;
    border: 1px solid var(--line); border-radius: 6px; font-size: 0.82rem;
    color: var(--muted);
  }}
  .openHint code {{ color: var(--accent); }}
  #slider, #motionSlider {{ width: min(420px, 52vw); }}
  .motionBar {{
    width: 100%; padding: 8px 18px; border-bottom: 1px solid var(--line);
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    background: #161922;
  }}
  .scene3dWrap {{
    margin: 10px 18px; padding: 10px; background: var(--panel);
    border-radius: 8px; border: 1px solid var(--line);
  }}
  .scene3dWrap h2 {{
    margin: 0 0 6px; font-size: 0.95rem; color: var(--accent);
  }}
  .scene3dMeta {{ color: var(--muted); font-size: 0.78rem; margin-bottom: 6px; }}
  #scene3d {{ width: 100%; height: min(520px, 58vh); }}
  main {{
    display: grid; grid-template-columns: minmax(0, 1fr) 320px;
    gap: 12px; padding: 12px 18px 18px;
  }}
  @media (max-width: 1200px) {{ main {{ grid-template-columns: 1fr; }} }}
  .imgs {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
  }}
  @media (max-width: 960px) {{ .imgs {{ grid-template-columns: 1fr; }} }}
  figure {{
    margin: 0; background: var(--panel); border-radius: 8px; padding: 8px;
  }}
  img {{
    width: 100%; height: auto; display: block; background: #000;
    border-radius: 4px; min-height: 420px; object-fit: contain;
  }}
  @media (max-width: 960px) {{ img {{ min-height: 300px; }} }}
  figcaption {{ font-size: 0.78rem; color: var(--muted); margin-top: 6px; }}
  .panel {{
    background: var(--panel); border-radius: 8px; padding: 12px; overflow: auto;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{
    text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line);
    vertical-align: top;
  }}
  th {{ color: var(--muted); font-weight: 500; width: 42%; }}
  .hint {{ color: var(--muted); font-size: 0.8rem; padding: 0 18px 16px; }}
  .events {{
    padding: 0 18px 12px; display: grid; grid-template-columns: 1fr;
    gap: 10px;
  }}
  .events figure {{ margin: 0; background: var(--panel); border-radius: 8px; padding: 8px; }}
  .events h2 {{
    grid-column: 1 / -1; margin: 0; font-size: 0.95rem; color: var(--accent);
  }}
  .eventTabs {{
    display: flex; flex-wrap: wrap; gap: 8px;
  }}
  .eventTabs button {{
    background: #20283a; color: var(--text); border: 1px solid var(--line);
    border-radius: 6px; padding: 6px 10px; cursor: pointer;
  }}
  .eventTabs button.active {{
    border-color: var(--accent); color: var(--accent);
  }}
</style>
</head>
<body>
<header>
  <h1>REFINING replay</h1>
  <span class="meta" id="meta"></span>
  <label>step <input type="range" id="slider" min="0" max="0" value="0"/></label>
  <span id="stepLabel"></span>
  <button type="button" id="framePrev" title="上一帧">◀</button>
  <button type="button" id="frameNext" title="下一帧">▶</button>
  <span id="frameLabel" class="meta"></span>
</header>
<p class="openHint">
  若图片裂图：在 replay 目录运行 <code>bash serve.sh</code>，浏览器打开
  <code>http://127.0.0.1:8765/replay.html</code>。3D 规划场景在下方（base_link，单位 m）。
</p>
<div class="motionBar">
  <span class="meta">逐帧回放</span>
  <label>frame <input type="range" id="motionSlider" min="0" max="0" value="0"/></label>
  <span id="motionLabel" class="meta"></span>
  <button type="button" id="motionPlay">▶ play</button>
  <button type="button" id="motionAfter">show after</button>
</div>
<section class="scene3dWrap">
  <h2>base_link 3D 规划 / 执行</h2>
  <div class="scene3dMeta" id="scene3dMeta">magenta=果 · cyan=杯口轨迹(执行) · orange=杯口规划 · yellow=腕相机光心 · grey=轴</div>
  <div id="scene3d"></div>
</section>
<section class="events" id="eventSection" hidden>
  <h2>中心后关键事件</h2>
  <div class="eventTabs" id="eventTabs"></div>
  <figure id="eventViewer" hidden>
    <img id="imgEventMain" alt="event_main"/>
    <figcaption id="capEventMain"></figcaption>
  </figure>
</section>
<main>
  <section class="imgs">
    <figure>
      <img id="imgWrist" alt="wrist"/>
      <figcaption id="capWrist">wrist detection viz</figcaption>
    </figure>
    <figure>
      <img id="imgFixed" alt="fixed"/>
      <figcaption id="capFixed">fixed mono RGB</figcaption>
    </figure>
  </section>
  <aside class="panel">
    <table id="metrics"></table>
  </aside>
</main>
<p class="hint">腕部图优先显示 BoT-SORT 检测叠字（T{{id}}/深度）；关键事件图用 sidebyside 多视角。用 ← → 或顶部按钮切帧。3D：杯口=TCP，相机=link6×T_link6_cam 光心。</p>
<script>
const DATA = {data_js};
const SCENE3D = {scene3d_js};

function sceneForStep(stepNum) {{
  const arr = (SCENE3D && SCENE3D.steps) || [];
  return arr.find(s => Number(s.step) === Number(stepNum)) || null;
}}
function pathTrace(path, name, color, width) {{
  if (!path || !path.x || !path.x.length) return null;
  return {{
    type: 'scatter3d', mode: 'lines+markers', name: name,
    x: path.x, y: path.y, z: path.z,
    line: {{ color: color, width: width || 5 }},
    marker: {{ size: 2.5, color: color }},
    hovertemplate: name + '<br>x=%{{x:.3f}} y=%{{y:.3f}} z=%{{z:.3f}}<extra></extra>',
  }};
}}
function renderScene3d(stepRec, motionIdx) {{
  const el = document.getElementById('scene3d');
  const meta = document.getElementById('scene3dMeta');
  if (!el || typeof Plotly === 'undefined') {{
    if (meta) meta.textContent = 'Plotly 未加载（需联网 CDN）— 仍可读 scene3d.json';
    return;
  }}
  const sc = sceneForStep(stepRec ? stepRec.step : -1);
  if (!sc) {{
    if (meta) meta.textContent = '本 step 无 3D 轨迹数据';
    Plotly.react(el, [], {{ paper_bgcolor: '#1c2030', plot_bgcolor: '#1c2030' }});
    return;
  }}
  const traces = [];
  const axes = [
    {{x:[0,0.15], y:[0,0], z:[0,0], name:'X', color:'#aa4444'}},
    {{x:[0,0], y:[0,0.15], z:[0,0], name:'Y', color:'#44aa66'}},
    {{x:[0,0], y:[0,0], z:[0,0.15], name:'Z↑', color:'#4488cc'}},
  ];
  axes.forEach(a => traces.push({{
    type: 'scatter3d', mode: 'lines', name: a.name,
    x: a.x, y: a.y, z: a.z,
    line: {{ color: a.color, width: 6 }},
    showlegend: true,
  }}));
  if (sc.berry) {{
    traces.push({{
      type: 'scatter3d', mode: 'markers', name: 'berry',
      x: [sc.berry[0]], y: [sc.berry[1]], z: [sc.berry[2]],
      marker: {{ size: 8, color: '#e040a0', symbol: 'diamond' }},
      hovertemplate: 'berry<br>%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}<extra></extra>',
    }});
  }}
  const planned = pathTrace(sc.planned_cup_path, 'cup planned', '#ff9f43', 6);
  if (planned) traces.push(planned);
  const executed = pathTrace(sc.executed_cup_path, 'cup executed', '#2ed3c6', 5);
  if (executed) traces.push(executed);
  const cam = pathTrace(sc.executed_cam_path, 'wrist cam', '#f6e58d', 4);
  if (cam) traces.push(cam);
  if (sc.tcp_start) {{
    traces.push({{
      type: 'scatter3d', mode: 'markers', name: 'cup start',
      x: [sc.tcp_start[0]], y: [sc.tcp_start[1]], z: [sc.tcp_start[2]],
      marker: {{ size: 6, color: '#7bed9f' }},
    }});
  }}
  if (sc.tcp_goal) {{
    traces.push({{
      type: 'scatter3d', mode: 'markers', name: 'cup goal',
      x: [sc.tcp_goal[0]], y: [sc.tcp_goal[1]], z: [sc.tcp_goal[2]],
      marker: {{ size: 6, color: '#ff6b6b' }},
    }});
  }}
  // Current motion frame tip marker
  const ex = sc.executed_cup_path;
  if (ex && ex.x && ex.x.length && motionIdx != null && motionIdx >= 0 && motionIdx < ex.x.length) {{
    traces.push({{
      type: 'scatter3d', mode: 'markers', name: 'cup now',
      x: [ex.x[motionIdx]], y: [ex.y[motionIdx]], z: [ex.z[motionIdx]],
      marker: {{ size: 9, color: '#ffffff', line: {{ width: 1, color: '#000' }} }},
    }});
  }}
  if (meta) {{
    meta.textContent =
      'step[' + sc.step + '] ' + (sc.control || '') + ' / ' + (sc.ik_method || sc.method || '') +
      ' · dist ' + num(sc.dist_cup) + '→' + num(sc.dist_cup_after) +
      'm · travel=' + num(sc.travel_m) + 'm · frame=base_link';
  }}
  const layout = {{
    paper_bgcolor: '#1c2030', plot_bgcolor: '#1c2030',
    font: {{ color: '#e8eaef', size: 11 }},
    margin: {{ l: 0, r: 0, t: 10, b: 0 }},
    showlegend: true,
    legend: {{ x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.25)' }},
    scene: {{
      xaxis: {{ title: 'X (m)', gridcolor: '#2a3144', backgroundcolor: '#161922' }},
      yaxis: {{ title: 'Y (m)', gridcolor: '#2a3144', backgroundcolor: '#161922' }},
      zaxis: {{ title: 'Z↑ (m)', gridcolor: '#2a3144', backgroundcolor: '#161922' }},
      aspectmode: 'data',
      camera: {{ eye: {{ x: 1.4, y: -1.6, z: 0.9 }} }},
    }},
  }};
  Plotly.react(el, traces, layout, {{ responsive: true, displayModeBar: true }});
}}

function setImg(el, src) {{
  if (!src) {{ el.removeAttribute('src'); el.alt = '(missing)'; return; }}
  try {{
    el.src = new URL(src, window.location.href).href;
  }} catch (e) {{
    el.src = src;
  }}
  el.alt = src;
  el.onerror = () => {{ el.alt = 'load failed: ' + src; }};
}}
function num(v) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  const n = Number(v);
  return Math.abs(n) >= 100 ? n.toFixed(1) : n.toFixed(3);
}}
function fmtUV(a) {{
  if (!a || a.length < 2) return '—';
  return '(' + num(a[0]) + ', ' + num(a[1]) + ')';
}}
function fmtXYZ(a) {{
  if (!a || a.length < 3) return '—';
  return '(' + [a[0], a[1], a[2]].map(num).join(', ') + ')';
}}
function lockImg(kind) {{
  return (DATA.lock_images && DATA.lock_images[kind]) ? DATA.lock_images[kind] : '';
}}
function buildFrameList() {{
  const frames = [];
  DATA.steps.forEach((s, stepI) => {{
    const mf = s.motion_frames || [];
    const after = s.images || {{}};
    mf.forEach((fr, mi) => {{
      frames.push({{
        stepI,
        motionIdx: mi,
        showAfter: false,
        wrist: fr.wrist || after.wrist || '',
        fine: fr.fine_viz || after.fine_viz || '',
        fixed: fr.fixed || after.fixed || lockImg('fixed') || '',
        tag: fr.tag || ('motion[' + mi + ']'),
      }});
    }});
    frames.push({{
      stepI,
      motionIdx: Math.max(0, mf.length - 1),
      showAfter: true,
      wrist: after.wrist || '',
      fine: after.fine_viz || '',
      fixed: after.fixed || lockImg('fixed') || '',
      tag: 'after',
    }});
  }});
  return frames;
}}
const ALL_FRAMES = buildFrameList();
let idx = 0, motionIdx = 0, frameIdx = 0, playTimer = null;
const EVENT_SPECS = [
  ['optical_fresh_lock', 'fresh optical lock（center_oneshot 后重新尝试 YOLO 锁）'],
  ['probe_tri_contact', 'probe tri / reproject fallback（YOLO 失败后的 3D 重投影）'],
  ['near_handoff_pending', 'near handoff pending（进入 WAIT_NEAR 前的接触前快照）'],
];
let currentEventTag = '';

function eventImg(tag) {{
  const imgs = (DATA.event_images && DATA.event_images[tag]) || {{}};
  return imgs.annotated || imgs.sidebyside || imgs.fine_viz || imgs.wrist || imgs.fixed || imgs.global_viz || '';
}}
function eventCaption(tag, label) {{
  const meta = (DATA.event_meta && DATA.event_meta[tag]) || {{}};
  const src = meta.source || '';
  const extra = [];
  if (src) extra.push('source=' + src);
  if (meta.track_id != null) extra.push('track_id=' + meta.track_id);
  if (meta.pix_off_reproject != null) extra.push('pix_off=' + num(meta.pix_off_reproject));
  if (meta.dist_cup_m != null) extra.push('dist_cup=' + num(meta.dist_cup_m) + 'm');
  if (meta.inferred) extra.push('(offline inferred)');
  return label + (extra.length ? ' · ' + extra.join(' · ') : '');
}}
function renderEvents() {{
  const available = EVENT_SPECS.filter(([tag]) => !!eventImg(tag));
  const section = document.getElementById('eventSection');
  const tabs = document.getElementById('eventTabs');
  const viewer = document.getElementById('eventViewer');
  section.hidden = available.length === 0;
  if (!available.length) return;
  if (!currentEventTag || !available.some(([tag]) => tag === currentEventTag)) {{
    currentEventTag = available[0][0];
  }}
  tabs.innerHTML = available.map(([tag, label]) =>
    '<button type="button" data-tag="' + tag + '" class="' + (tag === currentEventTag ? 'active' : '') + '">' + label + '</button>'
  ).join('');
  tabs.querySelectorAll('button').forEach(btn => {{
    btn.onclick = () => {{
      currentEventTag = btn.getAttribute('data-tag') || '';
      renderEvents();
    }};
  }});
  const current = available.find(([tag]) => tag === currentEventTag) || available[0];
  const src = eventImg(current[0]);
  viewer.hidden = !src;
  setImg(document.getElementById('imgEventMain'), src);
  document.getElementById('capEventMain').textContent = eventCaption(current[0], current[1]);
}}

function motionList() {{
  const s = DATA.steps[idx];
  return (s && s.motion_frames) ? s.motion_frames : [];
}}
function syncFromFrame(fi) {{
  frameIdx = Math.max(0, Math.min(ALL_FRAMES.length - 1, fi));
  const fr = ALL_FRAMES[frameIdx];
  idx = fr.stepI;
  motionIdx = fr.motionIdx;
  document.getElementById('slider').value = idx;
}}
function renderMotionUI() {{
  const mf = motionList();
  const ms = document.getElementById('motionSlider');
  ms.max = Math.max(0, mf.length);
  ms.value = Math.min(Number(ms.max), motionIdx);
  document.getElementById('motionLabel').textContent =
    mf.length ? ('motion[' + motionIdx + '] / ' + mf.length) : 'after only';
  document.getElementById('frameLabel').textContent =
    ALL_FRAMES.length ? ('frame ' + (frameIdx + 1) + ' / ' + ALL_FRAMES.length) : '';
}}
function render(stepI) {{
  if (ALL_FRAMES.length) syncFromFrame(frameIdx);
  else idx = stepI;
  const s = DATA.steps[idx];
  if (!s) return;
  const fr = ALL_FRAMES.length ? ALL_FRAMES[frameIdx] : null;
  const mode = fr ? fr.tag : 'after';
  document.getElementById('stepLabel').textContent =
    'servo[' + s.step + '] (' + (idx + 1) + '/' + DATA.steps.length + ')';
  document.getElementById('meta').textContent =
    DATA.session + ' · control=' + (s.control || '—') +
    ' · phase=' + (s.refine_phase || '—') +
    ' · motion=' + (s.n_motion_frames || (s.motion_frames || []).length || 0);
  renderMotionUI();

  const imgs = s.images || {{}};
  const centerOpenLoop = (
    s.refine_phase === 'center_oneshot' || s.control === 'center_oneshot_cart');
  const wrist = fr
    ? (centerOpenLoop
        ? (fr.wrist || imgs.wrist || '')
        : (fr.fine || fr.wrist || imgs.fine_viz || imgs.wrist || ''))
    : (centerOpenLoop
        ? (imgs.wrist || '')
        : (imgs.fine_viz || imgs.wrist || ''));
  const fixed = fr ? (fr.fixed || imgs.fixed || lockImg('fixed') || '') : (imgs.fixed || lockImg('fixed') || '');
  setImg(document.getElementById('imgWrist'), wrist);
  setImg(document.getElementById('imgFixed'), fixed);
  const wristCap = centerOpenLoop
    ? 'wrist RGB — center open-loop (lock cleared, no BoT-SORT)'
    : 'wrist detection viz';
  document.getElementById('capWrist').textContent = wristCap + ' (' + mode + ')';
  document.getElementById('capFixed').textContent = 'fixed mono RGB (' + mode + ')';
  const mi = fr && fr.motionIdx != null ? fr.motionIdx : 0;
  renderScene3d(s, mi);

  const lock = DATA.lock || {{}};
  const dj = s.delta_j_deg || {{}};
  const rows = [];
  if (lock.track_id != null) {{
    rows.push(['FSM lock track_id', lock.track_id]);
    rows.push(['lock conf / z_cam', num(lock.confidence) + ' / ' + num(lock.z_cam) + ' m']);
    rows.push(['lock berry_uv', fmtUV(lock.berry_uv)]);
  }}
  rows.push(
    ['track_id (step)', s.track_id],
    ['control / phase', (s.control || '—') + ' / ' + (s.refine_phase || '—')],
    ['berry_uv', fmtUV(s.berry_uv)],
    ['berry_uv_after', fmtUV(s.berry_uv_after) + (s.berry_uv_after_src ? ' (' + s.berry_uv_after_src + ')' : '')],
    ['optical center', fmtUV(s.optical_uv)],
    ['n_motion_frames', s.n_motion_frames || (s.motion_frames || []).length || 0],
    ['dist_cup -> after (m)', num(s.dist_cup) + ' -> ' + num(s.dist_cup_after)],
    ['berry_rel_cup', fmtXYZ(s.berry_rel_cup)],
    ['berry_rel_after', fmtXYZ(s.berry_rel_cup_after)],
    ['cmd_dxyz', fmtXYZ(s.cmd_dxyz)],
    ['achieved_dxyz', fmtXYZ(s.achieved_dxyz)],
    ['pix_err (px)', num(s.pix_err)],
    ['lat_m / pix_off', num(s.lat_m) + ' / ' + num(s.pix_off)],
    ['delta_j_deg', [dj.j1, dj.j2, dj.j3, dj.j5].map(num).join(' / ')],
    ['source', s.source || '—'],
  );
  document.getElementById('metrics').innerHTML = rows.map(([k, v]) =>
    '<tr><th>' + k + '</th><td>' + v + '</td></tr>').join('');
}}

const slider = document.getElementById('slider');
slider.max = Math.max(0, DATA.steps.length - 1);
slider.oninput = () => {{
  const stepI = Number(slider.value);
  const fi = ALL_FRAMES.findIndex(f => f.stepI === stepI);
  frameIdx = fi >= 0 ? fi : 0;
  render(stepI);
}};
document.getElementById('framePrev').onclick = () => {{
  frameIdx = Math.max(0, frameIdx - 1);
  render(idx);
}};
document.getElementById('frameNext').onclick = () => {{
  frameIdx = Math.min(Math.max(0, ALL_FRAMES.length - 1), frameIdx + 1);
  render(idx);
}};
document.getElementById('motionSlider').oninput = e => {{
  motionIdx = Number(e.target.value);
  const fi = ALL_FRAMES.findIndex(f =>
    f.stepI === idx && ((f.showAfter && motionIdx >= (motionList().length || 0)) || (!f.showAfter && f.motionIdx === motionIdx)));
  if (fi >= 0) frameIdx = fi;
  render(idx);
}};
document.getElementById('motionAfter').onclick = () => {{
  const fi = ALL_FRAMES.findIndex(f => f.stepI === idx && f.showAfter);
  if (fi >= 0) frameIdx = fi;
  render(idx);
}};
document.getElementById('motionPlay').onclick = () => {{
  if (playTimer) {{ clearInterval(playTimer); playTimer = null; return; }}
  if (!ALL_FRAMES.length) return;
  playTimer = setInterval(() => {{
    if (frameIdx >= ALL_FRAMES.length - 1) {{
      clearInterval(playTimer); playTimer = null; return;
    }}
    frameIdx += 1;
    render(idx);
  }}, 120);
}};
window.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') {{ e.preventDefault(); document.getElementById('framePrev').click(); }}
  if (e.key === 'ArrowRight') {{ e.preventDefault(); document.getElementById('frameNext').click(); }}
}});

render(0);
renderEvents();
</script>
</body>
</html>
"""
    (out_dir / 'replay.html').write_text(html, encoding='utf-8')
    serve_sh = out_dir / 'serve.sh'
    serve_sh.write_text(
        '#!/usr/bin/env bash\n'
        'cd "$(dirname "$0")"\n'
        'PORT=${1:-8765}\n'
        'echo "Open: http://127.0.0.1:${PORT}/replay.html"\n'
        'exec python3 -m http.server "$PORT"\n',
        encoding='utf-8')
    serve_sh.chmod(0o755)
    return timeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-dir', type=Path, required=True)
    parser.add_argument(
        '--fsm-log', type=Path,
        default=Path(__file__).resolve().parent.parent / 'log' / 'real_robot' / 'reach_fsm.log')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    qa_dir = args.qa_dir.resolve()
    if not qa_dir.is_dir():
        print(f'missing qa dir: {qa_dir}')
        return 2
    out_dir = args.out
    if out_dir is None:
        out_dir = qa_dir.parent.parent / 'refine_tune_viz' / f'replay_{qa_dir.name}'
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _scan_images(qa_dir)
    json_steps = _load_step_jsons(qa_dir)
    lock_json = _load_lock_json(qa_dir)
    lock_log, log_steps = _parse_fsm_log(args.fsm_log.resolve(), qa_dir.name)
    lock = lock_json or lock_log
    depth_qa = _load_depth_qa_json(qa_dir)
    depth_qa_images = _scan_depth_qa_images(qa_dir)

    timeline = _merge_timeline(
        qa_dir, images, json_steps, log_steps, lock,
        depth_qa=depth_qa, depth_qa_images=depth_qa_images)
    _write_plots(timeline, out_dir / 'plots.png')
    timeline = _write_html(timeline, out_dir)
    (out_dir / 'timeline.json').write_text(
        json.dumps(timeline, indent=2, ensure_ascii=True), encoding='utf-8')

    print(f'session={timeline["session"]} n_steps={timeline["n_steps"]}')
    if timeline.get('depth_qa'):
        dq = timeline['depth_qa']
        print(
            f'depth_qa source={dq.get("depth_source")} '
            f'z_cam={dq.get("z_cam_m")} travel={dq.get("travel_to_contact_m")}m')
    if lock:
        print(
            f'lock track_id={lock.get("track_id")} '
            f'base={lock.get("berry_base")} cup={lock.get("cup_dist")}')
    # Highlight freeze for this run review
    frozen = 0
    steps = timeline['steps']
    for i in range(1, len(steps)):
        a, b = steps[i - 1], steps[i]
        keys = ('z_cam', 'pix_err', 'err_yaw_deg', 'err_pitch_deg', 'dist_cup')
        if all(str(a.get(k)) == str(b.get(k)) for k in keys):
            tj_a = (a.get('target_j_deg') or {}).get('j1')
            tj_b = (b.get('target_j_deg') or {}).get('j1')
            if tj_a is not None and tj_b is not None and abs(float(tj_b) - float(tj_a)) > 0.05:
                frozen += 1
                if frozen <= 3:
                    print(f'freeze+j1-move: step {a["step"]}→{b["step"]}')
    if frozen:
        print(f'freeze_streak_transitions={frozen}')
    print(f'wrote {out_dir / "replay.html"}')
    print(f'wrote {out_dir / "timeline.json"}')
    print(f'wrote {out_dir / "plots.png"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
