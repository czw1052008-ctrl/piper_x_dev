#!/usr/bin/env python3
"""Visualize /planning/planner_input and optionally help record it.

**Authoritative data path (training / assembly check):**
  Topic:  /planning/planner_input   (picking_msgs/PlannerInput)
  Record: bash scripts/record_planner_bag.sh   → rosbag2

This script publishes overlays on:
  /planning/viz/fixed | /planning/viz/wrist | /planning/viz/hud | /planning/viz/status

Optional disk sidecars (NOT the training source of truth):
  --save-frames / --also-jsonl  → planner_qa/<session>/ for HTML QA

Usage:
  export PYTHONPATH="$(pwd)/scripts:${PYTHONPATH}"
  # 1) assembler must be running
  python3 scripts/planner_input_assembler.py
  # 2) live check
  ros2 topic echo /planning/planner_input --once
  # 3) viz
  python3 scripts/planner_input_recorder.py
  # 4) bag (separate terminal) — preferred for training
  bash scripts/record_planner_bag.sh
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from picking_msgs.msg import PlannerInput

FSM_NAMES = {
    0: 'IDLE',
    1: 'CLUSTER_ALIGN',
    2: 'APPROACH_FRUIT',
    3: 'RETRACT',
    4: 'HOLD',
}

FLAG_NAMES = (
    (1 << 0, 'no_clusters'),
    (1 << 1, 'no_berries'),
    (1 << 2, 'no_joints'),
    (1 << 3, 'cluster_id_unknown'),
    (1 << 4, 'fruit_id_unknown'),
    (1 << 5, 'no_occupancy'),
)

# Topics written into the training bag (must include planner_input).
BAG_TOPICS = (
    '/perception/scene_graph',
    '/perception/occupancy_esdf',
    '/perception/occupancy_local',
    '/perception/global/berries',
    '/perception/fine/berries',
    '/planning/planner_input',
    '/planning/task_context',
    '/planning/tool_trajectory_4s',
    '/planning/ik_joint_trajectory',
    '/planning/joint_cmd',
    '/planning/executor_status',
    '/feedback/joint_states',
    '/joint_states',
    '/pick/status',
    '/reach/status',
    '/planning/viz/fixed',
    '/planning/viz/wrist',
    '/planning/viz/occupancy_slice',
    '/planning/viz/hud',
    '/planning/viz/status',
)


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


def _image_to_bgr(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ('rgb8',):
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return rgb[:, :, ::-1].copy()
    if enc in ('bgr8',):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc in ('yuv422_yuy2', 'yuyv', 'yuyv422'):
        yuyv = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    raise ValueError(f'unsupported encoding {msg.encoding}')


def _bgr_to_imgmsg(bgr: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(bgr.shape[0]), int(bgr.shape[1])
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(bgr, dtype=np.uint8).tobytes()
    return msg


def planner_input_to_dict(msg: PlannerInput) -> Dict[str, Any]:
    flags = int(msg.input_flags)
    flag_list = [name for bit, name in FLAG_NAMES if flags & bit]
    clusters = []
    for c in msg.clusters:
        clusters.append({
            'id': int(c.id),
            'position': [float(v) for v in c.position],
            'bbox_uv': [float(c.bbox_u0), float(c.bbox_v0), float(c.bbox_u1), float(c.bbox_v1)],
            'confidence': float(c.confidence),
            'track_source': str(getattr(c, 'track_source', '') or 'live'),
        })
    berries = []
    for b in msg.berries:
        berries.append({
            'id': int(b.id),
            'position': [float(v) for v in b.position],
            'confidence': float(b.confidence),
            'visible_wrist': bool(b.visible_wrist),
            'visible_global': bool(b.visible_global),
            'pick_role': int(getattr(b, 'pick_role', 0)),
            'pick_role_name': {
                0: 'pending', 1: 'active', 2: 'done'
            }.get(int(getattr(b, 'pick_role', 0)), 'pending'),
            'bbox_uv': [float(b.bbox_u0), float(b.bbox_v0), float(b.bbox_u1), float(b.bbox_v1)],
            'image_uv': [float(b.image_u), float(b.image_v)],
            'center_depth_m': float(b.center_depth_m),
            'depth_mode': str(b.depth_mode),
            'track_source': str(getattr(b, 'track_source', '') or (
                'coast' if str(b.depth_mode) == 'coast' else 'live')),
            'z_depth_m': float(b.z_depth_m),
            'z_mono_m': float(b.z_mono_m),
            'surface_normal': [float(v) for v in b.surface_normal],
            'normal_valid': bool(b.normal_valid),
        })
    tc = msg.task_context
    ee = msg.end_effector
    ego = msg.ego
    return {
        'seq': int(msg.seq),
        'stamp': {
            'sec': int(msg.header.stamp.sec),
            'nanosec': int(msg.header.stamp.nanosec),
        },
        'frame_id': str(msg.header.frame_id),
        'input_flags': flags,
        'input_flag_names': flag_list,
        'clusters': clusters,
        'berries': berries,
        'obstacles_n': len(msg.obstacles),
        'occupancy_local_n': len(getattr(msg, 'occupancy_local', None).labels)
        if getattr(msg, 'occupancy_local', None) is not None
        and len(getattr(msg.occupancy_local, 'size_xyz', []) or []) >= 3
        and int(msg.occupancy_local.size_xyz[0]) > 0
        else 0,
        'obstacles': [
            {
                'id': int(o.id),
                'position': [float(v) for v in o.position],
                'radius_m': float(o.radius_m),
                'bbox_uv': [float(v) for v in o.bbox_uv],
                'source': str(o.source),
            }
            for o in msg.obstacles
        ],
        'ego': {
            'q_rad': [float(v) for v in ego.q_rad],
            'q_deg': [math.degrees(float(v)) for v in ego.q_rad],
            'qd_rad': [float(v) for v in ego.qd_rad],
            'tip_xyz': [float(v) for v in getattr(ego, 'tip_xyz', [0, 0, 0])],
            'approach_axis': [float(v) for v in getattr(ego, 'approach_axis', [0, 0, 1])],
            'ee_radius_m': float(getattr(ego, 'ee_radius_m', 0.02) or 0.02),
            'link_radius_m': float(getattr(ego, 'link_radius_m', 0.05) or 0.05),
            'workspace_aabb': [float(v) for v in getattr(ego, 'workspace_aabb', [])],
        },
        'task_context': {
            'fsm_state': int(tc.fsm_state),
            'fsm_name': FSM_NAMES.get(int(tc.fsm_state), f'UNKNOWN({tc.fsm_state})'),
            'cluster_id': int(tc.cluster_id),
            'fruit_id': int(tc.fruit_id),
        },
        'end_effector': {
            'tool_type_id': int(ee.tool_type_id),
            'profile_name': str(ee.profile_name),
            'tip_offset_link6': [float(v) for v in ee.tip_offset_link6],
            'approach_standoff_m': float(ee.approach_standoff_m),
            'cup_radius_m': float(ee.cup_radius_m),
        },
    }


def annotate_fixed(bgr: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
    vis = bgr.copy()
    for o in data.get('obstacles') or []:
        bb = o.get('bbox_uv') or [-1, -1, -1, -1]
        if min(bb) < 0:
            p = o.get('position') or [0, 0, 0]
            label = f'O{o.get("id")} r={float(o.get("radius_m", 0)):.2f}'
            cv2.putText(vis, label, (8, 20 + 14 * int(o.get('id', 0))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1, cv2.LINE_AA)
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in bb]
        color = (0, 140, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
        cv2.circle(vis, ((x0 + x1) // 2, (y0 + y1) // 2), 4, color, -1)
        cv2.putText(
            vis, f'O{o.get("id")}', (x0, max(y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    for c in data.get('clusters') or []:
        bb = c.get('bbox_uv') or [-1, -1, -1, -1]
        if min(bb) < 0:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in bb]
        cid = int(c.get('id', -1))
        src = str(c.get('track_source', 'live') or 'live')
        sel = cid == int(data.get('task_context', {}).get('cluster_id', -2))
        # live=cyan/red lock; coast=gray dashed-looking thinner box
        if src == 'coast':
            color = (160, 160, 160)
        else:
            color = (0, 0, 255) if sel else (0, 200, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2 if src == 'live' else 1)
        label = f'C{cid} {src} {float(c.get("confidence", 0)):.2f}'
        if sel:
            label += ' LOCK'
        cv2.putText(vis, label, (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        p = c.get('position') or [0, 0, 0]
        cv2.putText(
            vis, f'xyz=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})',
            (x0, min(y1 + 16, vis.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis


def annotate_wrist(bgr: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
    vis = bgr.copy()
    fruit_id = int(data.get('task_context', {}).get('fruit_id', -1))
    for b in data.get('berries') or []:
        bb = b.get('bbox_uv') or [-1, -1, -1, -1]
        if min(bb) >= 0:
            x0, y0, x1, y1 = [int(round(v)) for v in bb]
        else:
            uv = b.get('image_uv') or [-1, -1]
            if uv[0] < 0:
                continue
            x0 = int(uv[0]) - 12
            y0 = int(uv[1]) - 12
            x1 = int(uv[0]) + 12
            y1 = int(uv[1]) + 12
        bid = int(b.get('id', -1))
        role = int(b.get('pick_role', 0))
        role_name = str(b.get('pick_role_name') or 'pending')
        src = str(b.get('track_source', '') or (
            'coast' if str(b.get('depth_mode', '')) == 'coast' else 'live'))
        sel = role == 1 or (bid == fruit_id and fruit_id >= 0)
        if src == 'coast':
            color = (160, 160, 160)
        elif role == 1 or sel:
            color = (0, 255, 0)  # ACTIVE 采集
        elif role == 2:
            color = (180, 180, 80)  # DONE
        else:
            color = (255, 180, 0)  # PENDING 待采集
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2 if src == 'live' else 1)
        u = float((b.get('image_uv') or [0.5 * (x0 + x1), 0])[0])
        v = float((b.get('image_uv') or [0, 0.5 * (y0 + y1)])[1])
        if u >= 0 and v >= 0:
            cv2.circle(vis, (int(u), int(v)), 3, color, -1)
        z = float(b.get('center_depth_m', -1.0))
        nv = bool(b.get('normal_valid', False))
        n = b.get('surface_normal') or [0, 0, 0]
        label = f'B{bid} {role_name}'
        if z > 0 and src == 'live':
            label += f' z={z:.3f}'
        if role == 1 or sel:
            label += ' PICK'
        cv2.putText(vis, label, (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if nv and u >= 0 and v >= 0:
            tip = (int(u + 40 * float(n[0])), int(v - 40 * float(n[2])))
            cv2.arrowedLine(vis, (int(u), int(v)), tip, (255, 0, 255), 2, tipLength=0.3)
            cv2.putText(
                vis, f'n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})',
                (x0, min(y1 + 14, vis.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
        elif not nv:
            cv2.putText(
                vis, 'n=INVALID',
                (x0, min(y1 + 14, vis.shape[0] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 255), 1, cv2.LINE_AA)
    return vis


def hud_panel(data: Dict[str, Any], w: int = 640, h: int = 220) -> np.ndarray:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    tc = data.get('task_context') or {}
    ee = data.get('end_effector') or {}
    ego = data.get('ego') or {}
    flags = data.get('input_flag_names') or []
    qdeg = ego.get('q_deg') or [0] * 6
    lines = [
        f"TOPIC /planning/planner_input  seq={data.get('seq')}  "
        f"fsm={tc.get('fsm_name')}  cluster_id={tc.get('cluster_id')}  "
        f"fruit_id={tc.get('fruit_id')}",
        f"clusters={len(data.get('clusters') or [])}  "
        f"berries={len(data.get('berries') or [])}  "
        f"obstacles={len(data.get('obstacles') or [])}  "
        f"flags={','.join(flags) if flags else 'ok'}",
        f"tool={ee.get('profile_name')} tip={ego.get('tip_xyz')} "
        f"aabb={ego.get('workspace_aabb')}",
        f"q_deg=[{qdeg[0]:.1f},{qdeg[1]:.1f},{qdeg[2]:.1f},"
        f"{qdeg[3]:.1f},{qdeg[4]:.1f},{qdeg[5]:.1f}]",
    ]
    y = 22
    for line in lines:
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y += 22
    for b in (data.get('berries') or [])[:6]:
        n = b.get('surface_normal') or [0, 0, 0]
        src = str(b.get('track_source', 'live') or 'live')
        line = (
            f"B{b.get('id')}[{b.get('pick_role_name', 'pending')}/{src}]: "
            f"xyz=({b['position'][0]:.3f},{b['position'][1]:.3f},"
            f"{b['position'][2]:.3f}) z={b.get('center_depth_m', -1):.3f} "
            f"n_ok={b.get('normal_valid')} n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})"
        )
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 255, 180), 1, cv2.LINE_AA)
        y += 18
        if y >= h - 4:
            break
    return panel


def write_replay_html(session_dir: Path, rows: List[Dict[str, Any]]) -> None:
    cards = []
    for r in rows[-200:]:
        seq = r.get('seq', 0)
        fixed = f'frames/frame_{int(seq):05d}_fixed.jpg'
        wrist = f'frames/frame_{int(seq):05d}_wrist.jpg'
        tc = r.get('task_context') or {}
        flags = ','.join(r.get('input_flag_names') or []) or 'ok'
        cards.append(
            f'<div class="card">'
            f'<h3>seq {escape(str(seq))} · {escape(str(tc.get("fsm_name")))} · '
            f'flags={escape(flags)}</h3>'
            f'<div class="row">'
            f'<img src="{escape(fixed)}" alt="fixed"/>'
            f'<img src="{escape(wrist)}" alt="wrist"/>'
            f'</div>'
            f'<pre>{escape(json.dumps(r, indent=2)[:2500])}</pre>'
            f'</div>'
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>planner_input replay</title>
<style>
body{{font-family:sans-serif;background:#111;color:#eee;margin:16px}}
.card{{border:1px solid #333;margin:12px 0;padding:12px;border-radius:8px}}
.row{{display:flex;gap:8px;flex-wrap:wrap}}
img{{max-width:48%;height:auto;background:#000}}
pre{{font-size:11px;overflow:auto;max-height:220px;background:#1a1a1a;padding:8px}}
</style></head><body>
<h1>planner_input QA sidecar ({len(rows)} frames)</h1>
<p>Training source of truth is the rosbag topic <code>/planning/planner_input</code>, not this JSONL.</p>
{''.join(cards)}
</body></html>"""
    (session_dir / 'replay.html').write_text(html, encoding='utf-8')


class PlannerInputRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('planner_input_recorder')
        self._show = bool(args.show)
        self._jpeg_q = int(args.jpeg_quality)
        self._also_jsonl = bool(args.also_jsonl)
        self._save_frames = bool(args.save_frames) or self._also_jsonl
        self._last: Optional[Dict[str, Any]] = None
        self._fixed_bgr: Optional[np.ndarray] = None
        self._wrist_bgr: Optional[np.ndarray] = None
        self._rows: List[Dict[str, Any]] = []
        self._session_dir: Optional[Path] = None
        self._frames_dir: Optional[Path] = None
        self._jsonl = None
        self._bag_proc: Optional[subprocess.Popen] = None

        root = Path(args.qa_dir)
        sid = time.strftime('%Y%m%d_%H%M%S')
        if args.record_bag or self._save_frames or self._also_jsonl:
            root.mkdir(parents=True, exist_ok=True)
            self._session_dir = root / sid
            self._session_dir.mkdir(parents=True, exist_ok=True)

        if self._save_frames and self._session_dir is not None:
            self._frames_dir = self._session_dir / 'frames'
            self._frames_dir.mkdir(parents=True, exist_ok=True)

        if self._also_jsonl and self._session_dir is not None:
            self._jsonl = open(
                self._session_dir / 'planner_input_sidecar.jsonl', 'w', encoding='utf-8')
            self.get_logger().warn(
                'also-jsonl enabled: sidecar only — training must use rosbag '
                f'/planning/planner_input (session={self._session_dir})')

        if args.record_bag:
            if self._session_dir is None:
                root.mkdir(parents=True, exist_ok=True)
                self._session_dir = root / sid
                self._session_dir.mkdir(parents=True, exist_ok=True)
            bag_out = str(self._session_dir / 'bag')
            cmd = ['ros2', 'bag', 'record', '-o', bag_out, *BAG_TOPICS]
            self.get_logger().info(
                f'recording rosbag2 → {bag_out} topics={list(BAG_TOPICS)}')
            self._bag_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        self.create_subscription(PlannerInput, '/planning/planner_input', self._on_input, 10)
        self.create_subscription(
            Image, args.fixed_image_topic, self._on_fixed, qos_profile_sensor_data)
        self.create_subscription(
            Image, args.wrist_image_topic, self._on_wrist, qos_profile_sensor_data)

        self._pub_fixed = self.create_publisher(Image, '/planning/viz/fixed', 10)
        self._pub_wrist = self.create_publisher(Image, '/planning/viz/wrist', 10)
        self._pub_hud = self.create_publisher(Image, '/planning/viz/hud', 10)
        self._pub_status = self.create_publisher(String, '/planning/viz/status', 10)

        self.get_logger().info(
            'planner_input_recorder: source topic=/planning/planner_input '
            f'record_bag={bool(args.record_bag)} show={self._show}')

    def _on_fixed(self, msg: Image) -> None:
        try:
            self._fixed_bgr = _image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f'fixed decode: {exc}')

    def _on_wrist(self, msg: Image) -> None:
        try:
            self._wrist_bgr = _image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f'wrist decode: {exc}')

    def _on_input(self, msg: PlannerInput) -> None:
        data = planner_input_to_dict(msg)
        self._last = data
        stamp = msg.header.stamp

        fixed = self._fixed_bgr if self._fixed_bgr is not None else np.zeros((480, 640, 3), np.uint8)
        wrist = self._wrist_bgr if self._wrist_bgr is not None else np.zeros((480, 640, 3), np.uint8)
        fixed_a = annotate_fixed(fixed, data)
        wrist_a = annotate_wrist(wrist, data)
        hud = hud_panel(data, w=max(fixed_a.shape[1], 640))

        self._pub_fixed.publish(_bgr_to_imgmsg(fixed_a, stamp, 'camera_fixed'))
        self._pub_wrist.publish(_bgr_to_imgmsg(wrist_a, stamp, 'camera_wrist'))
        self._pub_hud.publish(_bgr_to_imgmsg(hud, stamp, 'planner_hud'))

        st = String()
        st.data = (
            f"/planning/planner_input seq={data['seq']} "
            f"fsm={data['task_context']['fsm_name']} "
            f"C={len(data['clusters'])} B={len(data['berries'])} "
            f"flags={','.join(data['input_flag_names']) or 'ok'}"
        )
        self._pub_status.publish(st)

        if self._show:
            h = max(fixed_a.shape[0], wrist_a.shape[0], 1)

            def _pad(img):
                if img.shape[0] == h:
                    return img
                out = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
                out[:img.shape[0]] = img
                return out

            canvas = np.hstack([_pad(fixed_a), _pad(wrist_a)])
            canvas = np.vstack([canvas, cv2.resize(hud, (canvas.shape[1], 180))])
            cv2.imshow('planner_input', canvas)
            cv2.waitKey(1)

        if self._save_frames and self._frames_dir is not None:
            seq = int(data['seq'])
            cv2.imwrite(
                str(self._frames_dir / f'frame_{seq:05d}_fixed.jpg'),
                fixed_a, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_q])
            cv2.imwrite(
                str(self._frames_dir / f'frame_{seq:05d}_wrist.jpg'),
                wrist_a, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_q])

        if self._also_jsonl and self._jsonl is not None:
            row = _json_safe(data)
            self._rows.append(row)
            self._jsonl.write(json.dumps(row, ensure_ascii=False) + '\n')
            self._jsonl.flush()

    def finalize(self) -> None:
        if self._bag_proc is not None:
            self._bag_proc.terminate()
            try:
                self._bag_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._bag_proc.kill()
            self._bag_proc = None
            if self._session_dir is not None:
                self.get_logger().info(
                    f'rosbag stopped → {self._session_dir / "bag"} '
                    '(read with: ros2 bag info … ; ros2 bag play …)')

        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None

        if self._session_dir is not None and (self._rows or self._save_frames):
            summary = {
                'source_topic': '/planning/planner_input',
                'msg_type': 'picking_msgs/msg/PlannerInput',
                'training_artifact': 'bag/' if (self._session_dir / 'bag').exists() else None,
                'sidecar_jsonl': 'planner_input_sidecar.jsonl' if self._rows else None,
                'n_sidecar_frames': len(self._rows),
                'session_dir': str(self._session_dir),
                'last': self._rows[-1] if self._rows else self._last,
            }
            (self._session_dir / 'summary.json').write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
            if self._rows:
                write_replay_html(self._session_dir, self._rows)

        if self._show:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ws = Path(__file__).resolve().parent.parent
    parser.add_argument('--qa-dir', default=str(ws / 'log' / 'real_robot' / 'planner_qa'))
    parser.add_argument(
        '--record-bag', action='store_true',
        help='ros2 bag record /planning/planner_input (+ joints/viz) into qa-dir/<session>/bag')
    parser.add_argument(
        '--also-jsonl', action='store_true',
        help='optional human-readable sidecar (NOT training source of truth)')
    parser.add_argument(
        '--save-frames', action='store_true',
        help='write annotated JPG frames for HTML QA')
    parser.add_argument('--show', action='store_true', help='OpenCV window')
    parser.add_argument('--jpeg-quality', type=int, default=85)
    parser.add_argument('--fixed-image-topic', default='/camera_fixed/color/image_raw')
    parser.add_argument('--wrist-image-topic', default='/camera_wrist/color/image_raw')
    args = parser.parse_args()

    rclpy.init()
    node = PlannerInputRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
