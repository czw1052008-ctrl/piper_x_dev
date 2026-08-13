"""Fine detection — wrist RGB-D YOLO+BoT-SORT+depth, streaming topics (FP optional)."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from picking_msgs.msg import DetectedBerry, DetectedBerryArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from picking_perception.berry_tracker import BerryTracker
from picking_perception.perception_utils import matrix_from_tf
from picking_perception.segmentation import filter_berry_mask, segment_blueberry_hsv
from picking_perception.yolo_berry_detector import TrackedYoloDetection, YoloBerryDetector


class _NoFoundationPose:
    """Stub so depth helpers can run without FoundationPose."""

    ready = False

    def detect_all(self, *_args, **_kwargs):
        return []


@dataclass
class _CamDetection:
    pose_cam: np.ndarray
    confidence: float
    track_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    mode: str
    z_depth: Optional[float] = None
    z_mono: Optional[float] = None


def _image_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ('rgb8',):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc in ('bgr8',):
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return bgr[:, :, ::-1].copy()
    raise ValueError(f'unsupported color encoding: {msg.encoding}')


def _image_to_depth_m(msg: Image) -> np.ndarray:
    enc = msg.encoding
    if enc in ('32FC1', '32FC'):
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    if enc in ('16UC1', 'mono16'):
        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return depth_mm.astype(np.float32) / 1000.0
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return arr.astype(np.float32)


def _pose_to_msg(T: np.ndarray, header) -> PoseStamped:
    ps = PoseStamped()
    ps.header = header
    ps.pose.position.x = float(T[0, 3])
    ps.pose.position.y = float(T[1, 3])
    ps.pose.position.z = float(T[2, 3])
    ps.pose.orientation.w = 1.0
    return ps


def _rgb_to_imgmsg(rgb: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(rgb.shape[0]), int(rgb.shape[1])
    msg.encoding = 'rgb8'
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return msg


class FineDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('fine_detector_node')
        self.declare_parameter('score_threshold', 0.3)
        self.declare_parameter('max_detections', 8)
        self.declare_parameter('mask_source', 'yolo')
        self.declare_parameter('yolo_model', 'yoloe-11s-seg-pf.pt')
        self.declare_parameter('yolo_classes', 'blueberry,blueberries,berry')
        self.declare_parameter('yolo_conf_threshold', 0.22)
        # Only while FSM lock is cleared (entry / after-center re-lock). Does not
        # change BoT-SORT or locked mid-range track() conf.
        self.declare_parameter('yolo_unlocked_conf_threshold', 0.10)
        self.declare_parameter('yolo_min_area_px', 120)
        self.declare_parameter('yolo_max_area_frac', 0.18)
        self.declare_parameter('yolo_min_circularity', 0.45)
        self.declare_parameter('yolo_max_aspect_ratio', 2.0)
        self.declare_parameter('yolo_open_vocab', True)
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('tracker_config', '')
        self.declare_parameter('berry_diameter_m', 0.015)
        self.declare_parameter('depth_min_m', 0.03)
        self.declare_parameter('depth_max_m', 1.8)
        # Pose Z: depth (wrist RGB-D mask median) | mono (bbox size) | fused.
        # Wrist Gemini: prefer depth when pixels exist; mono only if source=mono.
        self.declare_parameter('depth_pose_source', 'depth')
        self.declare_parameter('track_lost_max_frames', 25)
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter(
            'lock_bbox_file',
            'log/real_robot/target_bbox_lock.json',
        )
        self.declare_parameter('enable_foundation_pose', False)
        self.declare_parameter('mesh_path', '')
        self.declare_parameter('foundation_pose_root', '')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_wrist_color_optical_frame')
        self.declare_parameter('color_topic', '/camera_wrist/color/image_raw')
        self.declare_parameter('depth_topic', '/camera_wrist/depth/image_raw')
        self.declare_parameter('info_topic', '/camera_wrist/color/camera_info')
        self.declare_parameter('viz_topic', '/perception/fine/detection_viz')
        self.declare_parameter('publish_viz', True)

        self._fp: Any = _NoFoundationPose()
        if bool(self.get_parameter('enable_foundation_pose').value):
            try:
                from picking_perception.foundation_pose_wrapper import FoundationPoseWrapper
                mesh_path = self.get_parameter('mesh_path').value
                if not mesh_path:
                    from ament_index_python.packages import get_package_share_directory
                    mesh_path = os.path.join(
                        get_package_share_directory('picking_perception'), 'meshes', 'blueberry.obj')
                fp_root = self.get_parameter('foundation_pose_root').value or \
                    '/home/user/codes/piper_x_dev/FoundationPose'
                self._fp = FoundationPoseWrapper(
                    mesh_path=mesh_path, foundation_pose_root=fp_root)
                if self._fp.ready:
                    self.get_logger().info(f'FoundationPose ready (mesh={mesh_path})')
                else:
                    self.get_logger().warn('FoundationPose requested but not ready — YOLO+depth only')
                    self._fp = _NoFoundationPose()
            except Exception as exc:
                self.get_logger().warn(f'FoundationPose init failed ({exc}) — YOLO+depth only')
                self._fp = _NoFoundationPose()
        else:
            self.get_logger().info('FoundationPose disabled — YOLO + BoT-SORT + depth')

        self._mask_source = str(self.get_parameter('mask_source').value).lower()
        self._enable_tracking = bool(self.get_parameter('enable_tracking').value)
        tracker_cfg = str(self.get_parameter('tracker_config').value).strip()
        if not tracker_cfg:
            from ament_index_python.packages import get_package_share_directory
            tracker_cfg = os.path.join(
                get_package_share_directory('picking_perception'),
                'config', 'botsort_wrist.yaml')

        self._yolo: Optional[YoloBerryDetector] = None
        if self._mask_source == 'yolo':
            prompts = [
                p.strip() for p in str(self.get_parameter('yolo_classes').value).split(',') if p.strip()]
            self._yolo = YoloBerryDetector(
                model_name=str(self.get_parameter('yolo_model').value),
                class_prompts=prompts,
                conf_threshold=float(self.get_parameter('yolo_conf_threshold').value),
                min_area_px=int(self.get_parameter('yolo_min_area_px').value),
                max_area_frac=float(self.get_parameter('yolo_max_area_frac').value),
                min_circularity=float(self.get_parameter('yolo_min_circularity').value),
                max_aspect_ratio=float(self.get_parameter('yolo_max_aspect_ratio').value),
                max_detections=int(self.get_parameter('max_detections').value),
                open_vocab=bool(self.get_parameter('yolo_open_vocab').value),
                tracker_config=tracker_cfg,
            )
            self._yolo_unlocked_conf = float(
                self.get_parameter('yolo_unlocked_conf_threshold').value)
            if self._yolo.ready:
                self.get_logger().info(
                    f'YOLO ready prompts={prompts} tracker={tracker_cfg} '
                    f'conf={float(self.get_parameter("yolo_conf_threshold").value):.2f} '
                    f'unlocked_conf={self._yolo_unlocked_conf:.2f}')
            else:
                self.get_logger().error(
                    'YOLO unavailable — publishing empty fine detections')
        else:
            self._yolo_unlocked_conf = 0.10

        self._depth_util = BerryTracker(
            berry_diameter_m=float(self.get_parameter('berry_diameter_m').value),
            depth_min_m=float(self.get_parameter('depth_min_m').value),
            depth_max_m=float(self.get_parameter('depth_max_m').value),
            track_lost_max_frames=int(self.get_parameter('track_lost_max_frames').value),
        )
        self._track_lost_max = int(self.get_parameter('track_lost_max_frames').value)
        self._depth_pose_source = str(
            self.get_parameter('depth_pose_source').value or 'mono').strip().lower()
        if self._depth_pose_source not in ('mono', 'depth', 'fused'):
            self.get_logger().warn(
                f'unknown depth_pose_source={self._depth_pose_source!r}; using mono')
            self._depth_pose_source = 'mono'
        self.get_logger().info(f'depth_pose_source={self._depth_pose_source}')

        # Depth Anything V2 fuser — lazy-loaded on first use (pbvs-vlm-reach-v2).
        self._depth_fuser = None
        if self._depth_pose_source == 'fused':
            try:
                _scripts_dir = os.path.join(
                    os.path.dirname(__file__), '..', '..', '..', '..', 'scripts')
                if _scripts_dir not in sys.path:
                    sys.path.insert(0, os.path.abspath(_scripts_dir))
                from depth_fusion import DepthFuser
                self._depth_fuser = DepthFuser(use_da2=True)
                self.get_logger().info('DepthFuser (RGB-D + DA2) initialised.')
            except Exception as exc:
                self.get_logger().warn(
                    f'DepthFuser init failed ({exc}); falling back to mono')

        self._base_frame = self.get_parameter('base_frame').value
        self._camera_frame = self.get_parameter('camera_frame').value
        self._rgb = None
        self._depth = None
        self._k = None
        self._target_lock: Optional[DetectedBerry] = None
        self._fsm_locked_track_id: Optional[int] = None
        self._track_pose_cache: Dict[int, _CamDetection] = {}
        self._track_cache_t: Dict[int, float] = {}
        # World-fixed coast (base_link xyz) — never re-project stale cam pose (pixel glue).
        self._track_base_xyz: Dict[int, np.ndarray] = {}
        self._track_base_t: Dict[int, float] = {}
        self._lock_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
        self._lock_crop_xyxy: Optional[Tuple[int, int, int, int]] = None
        self._lock_bbox_file = str(
            self.get_parameter('lock_bbox_file').value or '').strip()
        self._lock_bbox_file_mtime = 0.0
        self._file_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
        self._lock_iou_min = 0.15
        self._file_bbox_center_max_px = 55.0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(
            Image, self.get_parameter('color_topic').value, self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter('info_topic').value, self._on_info, qos_profile_sensor_data)
        self.create_subscription(DetectedBerry, '/perception/target_lock', self._on_lock, 10)

        self._pub = self.create_publisher(DetectedBerryArray, '/perception/fine/berries', 10)
        self._publish_viz = bool(self.get_parameter('publish_viz').value)
        self._viz_pub = None
        if self._publish_viz:
            self._viz_pub = self.create_publisher(
                Image, str(self.get_parameter('viz_topic').value), qos_profile_sensor_data)
        self._last_viz_boxes: List[Tuple[Tuple[int, int, int, int], float, int]] = []
        hz = max(float(self.get_parameter('publish_hz').value), 0.2)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(f'fine_detector streaming /perception/fine/berries @ {hz:.1f} Hz')

    def _on_rgb(self, msg: Image) -> None:
        try:
            self._rgb = _image_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _on_depth(self, msg: Image) -> None:
        try:
            self._depth = _image_to_depth_m(msg)
        except Exception as exc:
            self.get_logger().warn(f'depth decode failed: {exc}')

    def _on_info(self, msg: CameraInfo) -> None:
        if self._k is None:
            self._k = np.array(msg.k).reshape(3, 3)

    def _on_lock(self, msg: DetectedBerry) -> None:
        self._target_lock = msg
        tid = int(msg.track_id)
        if tid >= 0:
            if self._fsm_locked_track_id is not None and self._fsm_locked_track_id != tid:
                if self._yolo is not None:
                    self._yolo.reset_tracker()
                self._lock_bbox_xyxy = None
            self._fsm_locked_track_id = tid
            try:
                u = float(msg.image_u)
                v = float(msg.image_v)
            except (TypeError, ValueError):
                u, v = -1.0, -1.0
            if u >= 0.0 and v >= 0.0 and self._rgb is not None:
                h, w = self._rgb.shape[:2]
                side = 56
                z = max(float(getattr(msg, 'z_mono_m', -1.0)), float(getattr(msg, 'z_depth_m', -1.0)))
                if z > 1e-4 and self._k is not None:
                    fx = max(1.0, float(self._k[0, 0]))
                    est = int(round(
                        fx * float(self.get_parameter('berry_diameter_m').value) / z))
                    side = max(24, min(96, est * 3))
                half = side // 2
                x0 = max(0, min(w - 1, int(round(u)) - half))
                y0 = max(0, min(h - 1, int(round(v)) - half))
                x1 = max(x0 + 1, min(w, int(round(u)) + half))
                y1 = max(y0 + 1, min(h, int(round(v)) + half))
                self._lock_bbox_xyxy = (x0, y0, x1, y1)
            # Seed base cache from lock message when already in base_link.
            bf = (msg.pose.header.frame_id or msg.header.frame_id or '')
            if bf.endswith('base_link') or bf == self._base_frame:
                p = msg.pose.pose.position
                self._track_base_xyz[tid] = np.array(
                    [float(p.x), float(p.y), float(p.z)], dtype=np.float64)
                self._track_base_t[tid] = time.time()
            self.get_logger().info(
                f'FSM target_lock track_id={tid}'
                f'{f" seed_bbox={self._lock_bbox_xyxy}" if self._lock_bbox_xyxy is not None else ""}')
        else:
            old = self._fsm_locked_track_id
            self._fsm_locked_track_id = None
            self._lock_bbox_xyxy = None
            # Full reset: clear every coast/cache, not only the previous lock id.
            self._track_base_xyz.clear()
            self._track_base_t.clear()
            self._track_pose_cache.clear()
            self._last_viz_boxes = []
            if self._yolo is not None:
                self._yolo.reset_tracker()
            self.get_logger().info(
                f'FSM target_lock cleared + tracker reset'
                f'{f" (was id={old})" if old is not None else ""}')

    def _resolve_lock_bbox_path(self) -> Optional[Path]:
        raw = self._lock_bbox_file
        if not raw:
            return None
        p = Path(raw)
        if p.is_file():
            return p
        for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
            cand = base / raw
            if cand.is_file():
                return cand
        return None

    def _maybe_load_bbox_file(self) -> None:
        path = self._resolve_lock_bbox_path()
        if path is None:
            return
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime <= self._lock_bbox_file_mtime:
            return
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            bb = data.get('bbox_xyxy')
            if not bb or len(bb) != 4:
                return
            self._file_bbox_xyxy = tuple(int(v) for v in bb)
            self._lock_bbox_file_mtime = mtime
            self.get_logger().info(
                f'loaded lock_bbox_file {path} bbox={self._file_bbox_xyxy}')
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.get_logger().warn(f'lock_bbox_file read failed ({path}): {exc}')

    def _pin_bbox_xyxy(self) -> Optional[Tuple[int, int, int, int]]:
        return self._lock_bbox_xyxy or self._file_bbox_xyxy

    def _depth_at_uv(
        self,
        depth: np.ndarray,
        u: int,
        v: int,
        *,
        radius_px: int = 4,
    ) -> Optional[float]:
        h, w = depth.shape[:2]
        u0 = max(0, u - radius_px)
        u1 = min(w, u + radius_px + 1)
        v0 = max(0, v - radius_px)
        v1 = min(h, v + radius_px + 1)
        patch = depth[v0:v1, u0:u1]
        dmin = float(self.get_parameter('depth_min_m').value)
        dmax = float(self.get_parameter('depth_max_m').value)
        valid = patch[(patch >= dmin) & (patch <= dmax)]
        if valid.size < 3:
            return None
        return float(np.median(valid))

    def _synthetic_file_bbox_detection(
        self,
        pin_bbox: Tuple[int, int, int, int],
        depth: np.ndarray,
        k: np.ndarray,
    ) -> Optional[_CamDetection]:
        """When YOLO misses the user-pinned berry, depth at bbox center."""
        x0, y0, x1, y1 = pin_bbox
        u = int(round(0.5 * (x0 + x1)))
        v = int(round(0.5 * (y0 + y1)))
        z_depth = self._depth_at_uv(depth, u, v)
        side = max(x1 - x0, y1 - y0, 12)
        mask = np.zeros(depth.shape[:2], dtype=np.uint8)
        half = max(side // 2, 6)
        y0m = max(0, v - half)
        y1m = min(depth.shape[0], v + half)
        x0m = max(0, u - half)
        x1m = min(depth.shape[1], u + half)
        mask[y0m:y1m, x0m:x1m] = 255
        z_mono = self._depth_util.mono_depth_from_mask(mask, k)
        if self._depth_pose_source == 'depth':
            if z_depth is not None:
                z = float(z_depth)
                mode = 'depth_raw'
            else:
                z = float(z_mono)
                mode = 'mono'
        elif z_depth is not None:
            z = float(z_depth)
            mode = 'depth_raw'
        else:
            z = float(z_mono)
            mode = 'mono'
        if z <= 1e-4:
            return None
        x, y, zz = self._depth_util.uv_to_cam_xyz(u, v, z, k)
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = x, y, zz
        tid = 100000 + (u // 8) * 1000 + (v // 8)
        return _CamDetection(
            T, 0.35, tid, pin_bbox, mode, z_depth, float(z_mono))

    @staticmethod
    def _bbox_iou(
        a: Tuple[int, int, int, int], b: Tuple[int, int, int, int],
    ) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = float(iw * ih)
        if inter <= 0.0:
            return 0.0
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = float(area_a + area_b - inter)
        return inter / union if union > 1e-6 else 0.0

    def _coast_locked_base(self, tid: int) -> Optional[_CamDetection]:
        """Coast locked fruit at last *base* XYZ (identity in world, not glued to pixel).

        While FSM holds this lock_id, keep publishing indefinitely so REFINING can
        freeze→oneshot without BoT-SORT / YOLO frame limits killing the stream.
        """
        if tid not in self._track_base_xyz:
            return None
        # Unlocked / other tracks: still bound by track_lost window.
        if tid != self._fsm_locked_track_id:
            age = time.time() - self._track_base_t.get(tid, 0.0)
            max_age = self._track_lost_max / max(
                float(self.get_parameter('publish_hz').value), 1.0)
            if age > max_age:
                return None
        # Pose in base; _tick will skip cam→base when mode=='base_coast'.
        T = np.eye(4)
        T[0:3, 3] = self._track_base_xyz[tid]
        bbox = self._pin_bbox_xyxy() or (0, 0, 1, 1)
        conf = 0.35
        if tid in self._track_pose_cache:
            conf = max(0.25, float(self._track_pose_cache[tid].confidence) * 0.5)
        return _CamDetection(T, conf, tid, bbox, 'base_coast', None, None)

    def _det_to_cam_pose(
        self,
        det: TrackedYoloDetection,
        depth: np.ndarray,
        k: np.ndarray,
        *,
        prior_z: Optional[float] = None,
    ) -> Tuple[np.ndarray, str, Optional[float], float]:
        z_depth = self._depth_util.depth_in_mask(depth, det.mask)
        z_mono = self._depth_util.mono_depth_from_mask(det.mask, k)
        if self._depth_pose_source == 'fused' and self._depth_fuser is not None:
            # Priority: RGB-D → Depth Anything V2 → mono.
            rgb = getattr(self, '_rgb', None)
            fused_z, fused_src = self._depth_fuser.get_depth(
                rgb_img=rgb if rgb is not None else np.zeros((1, 1, 3), dtype=np.uint8),
                raw_depth_m=depth,
                mask=det.mask.astype(bool),
            )
            if fused_z is not None:
                z = float(fused_z)
                mode = f'fused_{fused_src}'
                if prior_z is not None:
                    z = 0.90 * z + 0.10 * float(prior_z)
                    mode = f'fused_{fused_src}_coast'
            else:
                z = float(z_mono)
                mode = 'mono_fallback'
        elif self._depth_pose_source == 'mono':
            # Hardware RGB-D often samples background through mask holes.
            z = float(z_mono)
            mode = 'mono'
            if prior_z is not None:
                z = 0.85 * z + 0.15 * float(prior_z)
                mode = 'coast'
        elif self._depth_pose_source == 'depth':
            if z_depth is not None:
                z = float(z_depth)
                mode = 'depth_raw'
            elif prior_z is not None:
                z = float(prior_z)
                mode = 'depth_coast'
            else:
                z = float(z_mono)
                mode = 'depth_missing'
        elif z_depth is not None:
            z = float(z_depth)
            mode = 'depth_raw'
        elif prior_z is not None:
            z = 0.75 * z_mono + 0.25 * prior_z
            mode = 'coast'
        else:
            z = z_mono
            mode = 'mono'
        ys, xs = np.where(det.mask > 0)
        u, v = int(xs.mean()), int(ys.mean())
        x, y, zz = self._depth_util.uv_to_cam_xyz(u, v, z, k)
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = x, y, zz
        return T, mode, z_depth, float(z_mono)

    def _detect_cam_poses(self) -> Tuple[List[_CamDetection], str]:
        rgb, depth, k = self._rgb, self._depth, self._k
        assert rgb is not None and depth is not None and k is not None
        self._last_viz_boxes = []

        if self._mask_source == 'yolo' and self._enable_tracking and self._yolo and self._yolo.ready:
            lock_id = self._fsm_locked_track_id

            # Unlocked window (entry lock / after-center fresh lock): lower-conf
            # detect() only — do NOT weaken BoT-SORT for mid-range probe.
            if lock_id is None:
                unlocked_conf = float(self._yolo_unlocked_conf)
                dets = self._yolo.detect(rgb, conf=unlocked_conf)
                out: List[_CamDetection] = []
                for det in dets:
                    T, mode, z_depth, z_mono = self._det_to_cam_pose(det, depth, k)
                    x0, y0, x1, y1 = det.bbox_xyxy
                    u = int(round(0.5 * (x0 + x1)))
                    v = int(round(0.5 * (y0 + y1)))
                    # Stable positive synthetic id so FSM can pin; IoU-pin
                    # remaps onto BoT-SORT once lock is set.
                    tid = 100000 + (u // 8) * 1000 + (v // 8)
                    cd = _CamDetection(
                        T, float(det.confidence), tid, det.bbox_xyxy, mode,
                        z_depth, z_mono)
                    out.append(cd)
                    self._track_pose_cache[tid] = cd
                    self._track_cache_t[tid] = time.time()
                self._last_viz_boxes = [
                    (d.bbox_xyxy, d.confidence, d.track_id, d.mode,
                     d.z_depth, d.z_mono) for d in out]
                pin_bbox = self._file_bbox_xyxy
                if pin_bbox is not None:
                    if out:
                        best_i, best_iou = -1, 0.0
                        for i, cd in enumerate(out):
                            iou = self._bbox_iou(pin_bbox, cd.bbox_xyxy)
                            if iou > best_iou:
                                best_iou, best_i = iou, i
                        if best_i >= 0 and best_iou >= self._lock_iou_min:
                            out = [out[best_i]]
                            ids = str(out[0].track_id)
                            return out, (
                                f'YOLO file-bbox pin n=1 id={ids} '
                                f'IoU={best_iou:.2f} conf>={unlocked_conf:.2f}')
                        # Dense cluster: YOLO box may miss user pin — center match.
                        bx = 0.5 * (pin_bbox[0] + pin_bbox[2])
                        by = 0.5 * (pin_bbox[1] + pin_bbox[3])
                        best_i, best_d = -1, 1e9
                        for i, cd in enumerate(out):
                            x0, y0, x1, y1 = cd.bbox_xyxy
                            cx = 0.5 * (x0 + x1)
                            cy = 0.5 * (y1 + y0)
                            d = math.hypot(cx - bx, cy - by)
                            if d < best_d:
                                best_d, best_i = d, i
                        if (
                            best_i >= 0
                            and best_d <= self._file_bbox_center_max_px
                        ):
                            out = [out[best_i]]
                            ids = str(out[0].track_id)
                            return out, (
                                f'YOLO file-bbox center-pin n=1 id={ids} '
                                f'd={best_d:.1f}px conf>={unlocked_conf:.2f}')
                    synth = self._synthetic_file_bbox_detection(
                        pin_bbox, depth, k)
                    if synth is not None:
                        self._last_viz_boxes = [
                            (synth.bbox_xyxy, synth.confidence, synth.track_id,
                             synth.mode, synth.z_depth, synth.z_mono)]
                        return [synth], (
                            f'file-bbox depth@center id={synth.track_id} '
                            f'mode={synth.mode}')
                    return [], (
                        f'YOLO file-bbox: no IoU/center/depth pin '
                        f'(n_yolo={len(out)})')
                if not out:
                    return [], f'YOLO unlocked detect: empty (conf>={unlocked_conf:.2f})'
                ids = ','.join(str(d.track_id) for d in out[:6])
                return out, (
                    f'YOLO unlocked-detect n={len(out)} ids=[{ids}] '
                    f'conf>={unlocked_conf:.2f}')

            # After FSM lock: BoT-SORT IDs are not the synthetic detect() id
            # (156022 etc). Pin by reprojecting frozen P, not track().
            if lock_id is not None:
                return self._detect_locked_uv_pin(rgb, depth, k, int(lock_id))

            tracked = self._yolo.track(rgb)

            if not tracked:
                coast = self._coast_locked_base(lock_id)
                if coast is not None:
                    self._last_viz_boxes = [
                        (coast.bbox_xyxy, coast.confidence, lock_id,
                         coast.mode, coast.z_depth, coast.z_mono)]
                    return [coast], f'BoT-SORT base_coast id={lock_id}'
                return [], 'BoT-SORT: no tracks'

            out = []
            seen_ids = set()
            for det in tracked:
                tid = int(det.track_id)
                if tid < 0:
                    continue
                prior_z = None
                if tid in self._track_pose_cache:
                    prior_z = float(self._track_pose_cache[tid].pose_cam[2, 3])
                T, mode, z_depth, z_mono = self._det_to_cam_pose(
                    det, depth, k, prior_z=prior_z)
                cd = _CamDetection(
                    T, float(det.confidence), tid, det.bbox_xyxy, mode,
                    z_depth, z_mono)
                out.append(cd)
                self._track_pose_cache[tid] = cd
                self._track_cache_t[tid] = time.time()
                seen_ids.add(tid)

            # Pin: if lock ID missing, re-attach best IoU overlap with last lock bbox.
            if (
                lock_id is not None
                and lock_id not in seen_ids
                and self._pin_bbox_xyxy() is not None
                and out
            ):
                pin_bbox = self._pin_bbox_xyxy()
                assert pin_bbox is not None
                best_i, best_iou = -1, 0.0
                for i, cd in enumerate(out):
                    iou = self._bbox_iou(pin_bbox, cd.bbox_xyxy)
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_i >= 0 and best_iou >= self._lock_iou_min:
                    pinned = out[best_i]
                    # Keep depth diagnostics from the overlapping detection.
                    pin_mode = 'iou_pin' if pinned.z_depth is not None else 'iou_pin_mono'
                    pinned = _CamDetection(
                        pinned.pose_cam, pinned.confidence, lock_id,
                        pinned.bbox_xyxy, pin_mode,
                        pinned.z_depth, pinned.z_mono)
                    out[best_i] = pinned
                    self._track_pose_cache[lock_id] = pinned
                    self._track_cache_t[lock_id] = time.time()
                    seen_ids.add(lock_id)
                    self.get_logger().info(
                        f'pin lock id={lock_id} via IoU={best_iou:.2f} '
                        f'mode={pin_mode} z_d={pinned.z_depth} z_m={pinned.z_mono} '
                        f'(was T{tracked[best_i].track_id if best_i < len(tracked) else "?"})')

            if lock_id is not None and lock_id not in seen_ids:
                coast = self._coast_locked_base(lock_id)
                if coast is not None:
                    out.append(coast)
                    seen_ids.add(lock_id)

            # After lock: prefer the pinned fruit; if it is only a base_coast
            # ghost, also publish live tracks so FSM can re-pin (otherwise the
            # real berries never leave this node and tracking looks "broken").
            viz_all = list(out)
            if lock_id is not None:
                locked = [d for d in out if d.track_id == lock_id]
                live = [d for d in out if d.mode != 'base_coast']
                if locked and locked[0].mode != 'base_coast':
                    out = locked
                    self._lock_bbox_xyxy = locked[0].bbox_xyxy
                elif locked and locked[0].mode == 'base_coast':
                    # Ghost first (compat), then live candidates for FSM re-pin.
                    out = locked + [d for d in live if d.track_id != lock_id]
                    self.get_logger().warn(
                        f'lock id={lock_id} is base_coast; publishing '
                        f'{len(out) - 1} live track(s) for FSM re-pin')
                elif live:
                    out = live
                else:
                    out = []

            self._last_viz_boxes = [
                (d.bbox_xyxy, d.confidence, d.track_id, d.mode,
                 d.z_depth, d.z_mono) for d in viz_all]
            ids = ','.join(str(d.track_id) for d in viz_all[:6])
            return out, f'BoT-SORT n={len(viz_all)} ids=[{ids}] lock={lock_id}'

        if self._mask_source == 'yolo' and self._yolo and self._yolo.ready:
            detections: List[_CamDetection] = []
            for det in self._yolo.detect(rgb):
                T, mode, z_depth, z_mono = self._det_to_cam_pose(det, depth, k)
                detections.append(_CamDetection(
                    T, float(det.confidence), -1, det.bbox_xyxy, mode,
                    z_depth, z_mono))
                self._last_viz_boxes.append(
                    (det.bbox_xyxy, float(det.confidence), -1, mode, z_depth, z_mono))
            if not detections:
                return [], 'YOLO found no blueberries'
            return detections, 'YOLO depth (no MOT)'

        if self._mask_source == 'yolo':
            return [], 'YOLO not ready — empty (no HSV fallback)'

        mask = filter_berry_mask(segment_blueberry_hsv(rgb))
        if mask.sum() < 100:
            return [], 'HSV mask too small'
        z_depth = self._depth_util.depth_in_mask(depth, mask)
        z_mono = self._depth_util.mono_depth_from_mask(mask, k)
        if self._depth_pose_source == 'mono':
            z = float(z_mono)
            mode = 'mono'
        elif self._depth_pose_source == 'depth':
            if z_depth is not None:
                z = float(z_depth)
                mode = 'depth_raw'
            else:
                z = float(z_mono)
                mode = 'depth_missing'
        elif z_depth is not None:
            z = float(z_depth)
            mode = 'depth_raw'
        else:
            z = z_mono
            mode = 'mono'
        ys, xs = np.where(mask > 0)
        u, v = int(xs.mean()), int(ys.mean())
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        self._last_viz_boxes = [((x0, y0, x1, y1), 0.4, -1, mode, z_depth, z_mono)]
        x, y, zz = self._depth_util.uv_to_cam_xyz(u, v, z, k)
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = x, y, zz
        return [_CamDetection(T, 0.4, -1, (x0, y0, x1, y1), mode, z_depth, z_mono)], (
            f'HSV {mode} (explicit)')

    @staticmethod
    def _viz_draw_label_panel(
        vis: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        lines: List[str],
        *,
        border_rgb: Tuple[int, int, int],
        prefer_above: bool = True,
    ) -> None:
        """High-contrast label strip (black bg, white text) — readable in QA replay."""
        import cv2  # type: ignore

        if not lines:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.58, 2
        pad_x, pad_y = 6, 5
        line_gap = 20
        max_tw = 0
        for ln in lines:
            tw, _ = cv2.getTextSize(ln, font, fs, th)[0]
            max_tw = max(max_tw, int(tw))
        panel_w = max(max_tw + 2 * pad_x, x1 - x0, 88)
        panel_h = len(lines) * line_gap + 2 * pad_y
        h_img, w_img = vis.shape[:2]
        px0 = int(max(0, min(w_img - panel_w, x0)))
        if prefer_above and y0 - panel_h - 4 >= 0:
            py0 = y0 - panel_h - 4
        elif y1 + panel_h + 4 < h_img:
            py0 = y1 + 4
        else:
            py0 = max(0, min(h_img - panel_h, y0))
        px1, py1 = px0 + panel_w, py0 + panel_h
        cv2.rectangle(vis, (px0, py0), (px1, py1), (12, 12, 12), -1)
        cv2.rectangle(vis, (px0, py0), (px1, py1), border_rgb, 2)
        for i, ln in enumerate(lines):
            ty = py0 + pad_y + (i + 1) * line_gap - 6
            cv2.putText(
                vis, ln, (px0 + pad_x, ty), font, fs, (255, 255, 255), th,
                cv2.LINE_AA)

    @staticmethod
    def _viz_draw_detection_box(
        vis: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        border_rgb: Tuple[int, int, int],
        thickness: int,
        corner_len: Optional[int] = None,
    ) -> None:
        """Corner brackets + thin full rect — box interior stays visible."""
        import cv2  # type: ignore

        cv2.rectangle(vis, (x0, y0), (x1, y1), border_rgb, max(1, thickness - 1))
        cl = corner_len or max(10, min(22, (x1 - x0) // 3, (y1 - y0) // 3))
        for cx, cy, dx, dy in (
            (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1),
        ):
            cv2.line(vis, (cx, cy), (cx + dx * cl, cy), border_rgb, thickness)
            cv2.line(vis, (cx, cy), (cx, cy + dy * cl), border_rgb, thickness)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        cv2.circle(vis, (cx, cy), 4, border_rgb, -1)
        cv2.circle(vis, (cx, cy), 4, (255, 255, 255), 1)

    def _viz_box_half_px(self, z_cam_m: float) -> int:
        if z_cam_m <= 1e-4 or self._k is None:
            return 12
        fx = max(1.0, float(self._k[0, 0]))
        diam = float(self.get_parameter('berry_diameter_m').value)
        est = int(round(fx * diam / z_cam_m))
        return max(8, min(48, est * 2))

    def _uv_to_bbox_xyxy(self, u: float, v: float, half: int) -> Tuple[int, int, int, int]:
        h_img, w_img = self._rgb.shape[:2]
        u_i, v_i = int(round(u)), int(round(v))
        x0 = max(0, min(w_img - 1, u_i - half))
        y0 = max(0, min(h_img - 1, v_i - half))
        x1 = max(x0 + 1, min(w_img, u_i + half))
        y1 = max(y0 + 1, min(h_img, v_i + half))
        return x0, y0, x1, y1

    def _cd_mask_uv(
        self, cd: _CamDetection, k: np.ndarray,
    ) -> Tuple[float, float]:
        z = float(cd.pose_cam[2, 3])
        if z > 1e-6:
            fx, fy = float(k[0, 0]), float(k[1, 1])
            cx, cy = float(k[0, 2]), float(k[1, 2])
            return (
                fx * float(cd.pose_cam[0, 3]) / z + cx,
                fy * float(cd.pose_cam[1, 3]) / z + cy,
            )
        x0, y0, x1, y1 = cd.bbox_xyxy
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    def _dets_to_cam(
        self,
        dets,
        depth: np.ndarray,
        k: np.ndarray,
    ) -> Tuple[List[_CamDetection], List[Tuple[float, float]]]:
        out: List[_CamDetection] = []
        uvs: List[Tuple[float, float]] = []
        for det in dets:
            T, mode, z_depth, z_mono = self._det_to_cam_pose(det, depth, k)
            x0, y0, x1, y1 = det.bbox_xyxy
            u = int(round(0.5 * (x0 + x1)))
            v = int(round(0.5 * (y0 + y1)))
            tid = 100000 + (u // 8) * 1000 + (v // 8)
            cd = _CamDetection(
                T, float(det.confidence), tid, det.bbox_xyxy, mode,
                z_depth, z_mono)
            out.append(cd)
            uvs.append(self._cd_mask_uv(cd, k))
        return out, uvs

    def _detect_locked_uv_pin(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        k: np.ndarray,
        lock_id: int,
    ) -> Tuple[List[_CamDetection], str]:
        """Crop-YOLO around reprojected P0. No BoT-SORT / full-frame top-N pin."""
        unlocked_conf = float(self._yolo_unlocked_conf)
        full_dets = self._yolo.detect(rgb, conf=unlocked_conf) if self._yolo else []
        out, _ = self._dets_to_cam(full_dets, depth, k)
        self._last_viz_boxes = [
            (d.bbox_xyxy, d.confidence, d.track_id, d.mode,
             d.z_depth, d.z_mono) for d in out]
        self._lock_crop_xyxy = None

        pred = None
        xyz = self._track_base_xyz.get(int(lock_id))
        if xyz is not None:
            pred = self._reproject_base_xyz_uv(xyz)

        coast = self._coast_locked_base(lock_id)
        h_img, w_img = rgb.shape[:2]
        margin = 8.0

        def _miss(why: str) -> Tuple[List[_CamDetection], str]:
            if coast is not None:
                return [coast] + out, f'uv_pin miss {why} lock={lock_id} n={len(out)} -> coast'
            return out, f'uv_pin miss {why} lock={lock_id} n={len(out)} no-coast'

        if pred is None:
            return _miss('no-P0')
        u, v, zc = pred
        if not (margin <= u < w_img - margin and margin <= v < h_img - margin):
            return _miss(f'P0-off ({u:.0f},{v:.0f})')

        half = 96
        self._lock_crop_xyxy = (
            max(0, int(round(u)) - half),
            max(0, int(round(v)) - half),
            min(w_img, int(round(u)) + half),
            min(h_img, int(round(v)) + half),
        )
        crop_dets = self._yolo.detect_near_uv(
            rgb, (u, v), half_px=half, conf=0.05, min_area_px=40,
        ) if self._yolo else []
        pin_cds, pin_uvs = self._dets_to_cam(crop_dets, depth, k)

        fx = float(k[0, 0])
        max_px = float(np.clip(0.008 * fx / max(0.08, float(zc)), 8.0, 20.0))
        pinned = None
        duv = None
        best = None
        if pin_uvs:
            dists = [
                math.hypot(uv[0] - u, uv[1] - v) for uv in pin_uvs]
            order = sorted(range(len(dists)), key=lambda i: dists[i])
            best = order[0]
            if dists[best] <= max_px:
                amb = (
                    len(order) >= 2
                    and (dists[order[1]] - dists[best]) < 6.0
                    and dists[best] > 8.0)
                if not amb:
                    src = pin_cds[best]
                    pinned = _CamDetection(
                        src.pose_cam, src.confidence, lock_id, src.bbox_xyxy,
                        src.mode if src.mode != 'base_coast' else 'depth_raw',
                        src.z_depth, src.z_mono)
                    duv = dists[best]
                    self._lock_bbox_xyxy = src.bbox_xyxy
                    self._track_pose_cache[lock_id] = pinned
                    self._track_cache_t[lock_id] = time.time()

        if pinned is not None:
            viz = [pinned] + out
            self._last_viz_boxes = [
                (d.bbox_xyxy, d.confidence, d.track_id, d.mode,
                 d.z_depth, d.z_mono) for d in viz]
            return [pinned], (
                f'uv_pin crop lock={lock_id} duv={duv:.1f}px '
                f'max={max_px:.0f} n_crop={len(pin_cds)}')
        n_crop = len(pin_cds)
        nearest = ''
        if pin_uvs and best is not None:
            nearest = f' nearest={dists[best]:.0f}px'
        return _miss(f'crop n={n_crop}{nearest}')

    def _reproject_base_xyz_uv(
        self, xyz_base: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """Frozen lock point in base_link → optical (u, v, z_cam)."""
        if self._k is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self._camera_frame, self._base_frame, rclpy.time.Time())
            T_base_cam = matrix_from_tf(tf)
            T_cam_base = np.linalg.inv(T_base_cam)
            p_b = np.asarray(xyz_base, dtype=np.float64).reshape(3)
            p_c = T_cam_base @ np.array([p_b[0], p_b[1], p_b[2], 1.0], dtype=np.float64)
            zc = float(p_c[2])
            if zc <= 0.05:
                return None
            fx, fy = float(self._k[0, 0]), float(self._k[1, 1])
            cx, cy = float(self._k[0, 2]), float(self._k[1, 2])
            u = fx * float(p_c[0]) / zc + cx
            v = fy * float(p_c[1]) / zc + cy
            return u, v, zc
        except Exception:
            return None

    def _viz_draw_frozen_p(self, vis: np.ndarray, lock_id: int) -> Optional[Tuple[float, float]]:
        """Draw frozen lock P reprojected to this wrist frame. Returns UV or None."""
        import cv2  # type: ignore

        xyz = self._track_base_xyz.get(int(lock_id))
        if xyz is None:
            return None
        uvz = self._reproject_base_xyz_uv(xyz)
        if uvz is None:
            return None
        u, v, zc = uvz
        h_img, w_img = vis.shape[:2]
        if not (0 <= u < w_img and 0 <= v < h_img):
            cv2.putText(
                vis, f'P0 off-image ({u:.0f},{v:.0f})', (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 255), 2, cv2.LINE_AA)
            return (float(u), float(v))
        ui, vi = int(round(u)), int(round(v))
        half = self._viz_box_half_px(zc)
        col = (255, 80, 255)
        cv2.circle(vis, (ui, vi), half, col, 2, cv2.LINE_AA)
        cv2.circle(vis, (ui, vi), 5, col, -1, cv2.LINE_AA)
        cv2.circle(vis, (ui, vi), 5, (255, 255, 255), 1, cv2.LINE_AA)
        arm = max(12, half + 6)
        cv2.line(vis, (ui - arm, vi), (ui + arm, vi), col, 2, cv2.LINE_AA)
        cv2.line(vis, (ui, vi - arm), (ui, vi + arm), col, 2, cv2.LINE_AA)
        tag = f'P0  ({ui},{vi})  z={zc*1000:.0f}mm'
        cv2.putText(
            vis, tag, (min(w_img - 280, max(8, ui + half + 6)), max(48, vi - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        return (float(u), float(v))

    def _viz_draw_one_box(
        self,
        vis: np.ndarray,
        bbox: Tuple[int, int, int, int],
        tid: int,
        conf: float,
        mode: str,
        z_d: Optional[float],
        z_m: Optional[float],
        *,
        locked: bool,
    ) -> None:
        x0, y0, x1, y1 = bbox
        is_coast = mode in ('base_coast', 'frozen_reproj')
        if locked:
            color = (255, 80, 255) if is_coast else (40, 255, 80)
        else:
            color = (40, 180, 255)
        self._viz_draw_detection_box(
            vis, x0, y0, x1, y1, border_rgb=color, thickness=3 if locked else 2)
        z_mm = None
        if z_d is not None and float(z_d) > 1e-4:
            z_mm = float(z_d) * 1000.0
        elif z_m is not None and float(z_m) > 1e-4:
            z_mm = float(z_m) * 1000.0
        if locked:
            tag = f'LIVE  conf={conf:.2f}'
            if is_coast:
                tag = f'P0  conf={conf:.2f}'
            if mode == 'base_coast':
                tag = f'P0  conf={conf:.2f}'
        else:
            tag = f'CAND id={tid}  conf={conf:.2f}'
        if z_mm is not None:
            tag += f'  z={z_mm:.0f}mm'
        if mode and mode not in ('', 'depth_raw', 'mono'):
            tag += f'  {mode}'
        self._viz_draw_label_panel(
            vis, x0, y0, x1, y1, [tag], border_rgb=color)

    def _publish_detection_viz(self, stamp, label: str) -> None:
        if self._viz_pub is None or self._rgb is None:
            return
        import cv2  # type: ignore

        vis = self._rgb.copy()
        lock_id = self._fsm_locked_track_id
        h_img, w_img = vis.shape[:2]
        # Status banner (top) — always readable.
        banner = f'fine  {label}'[:96]
        cv2.rectangle(vis, (0, 0), (w_img, 30), (12, 12, 12), -1)
        cv2.putText(
            vis, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (120, 230, 255), 2, cv2.LINE_AA)
        if lock_id is not None and lock_id >= 0:
            cv2.putText(
                vis, 'P0 reproj', (w_img - 130, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 255), 2, cv2.LINE_AA)

        if lock_id is not None and lock_id >= 0:
            p0_uv = self._viz_draw_frozen_p(vis, int(lock_id))
            if self._lock_crop_xyxy is not None:
                x0, y0, x1, y1 = self._lock_crop_xyxy
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 80, 255), 1, cv2.LINE_AA)
            live_box = None
            for box in self._last_viz_boxes:
                if int(box[2]) != int(lock_id):
                    continue
                mode = str(box[3] if len(box) > 3 else '')
                if mode == 'base_coast':
                    continue
                live_box = box
                break
            for box in self._last_viz_boxes[:8]:
                if int(box[2]) == int(lock_id):
                    continue
                mode = str(box[3] if len(box) > 3 else '')
                if mode == 'base_coast':
                    continue
                self._viz_draw_one_box(
                    vis,
                    tuple(int(v) for v in box[0]),
                    int(box[2]),
                    float(box[1]),
                    mode,
                    box[4] if len(box) > 4 else None,
                    box[5] if len(box) > 5 else None,
                    locked=False,
                )
            if live_box is not None:
                bbox = tuple(int(v) for v in live_box[0])
                self._viz_draw_one_box(
                    vis, bbox, int(lock_id), float(live_box[1]),
                    str(live_box[3] if len(live_box) > 3 else 'uv_pin'),
                    live_box[4] if len(live_box) > 4 else None,
                    live_box[5] if len(live_box) > 5 else None,
                    locked=True)
                if p0_uv is not None:
                    lu = int(round(0.5 * (bbox[0] + bbox[2])))
                    lv = int(round(0.5 * (bbox[1] + bbox[3])))
                    cv2.line(
                        vis,
                        (int(round(p0_uv[0])), int(round(p0_uv[1]))),
                        (lu, lv), (80, 255, 120), 2, cv2.LINE_AA)
        else:
            # Pre-lock: show YOLO/BoT-SORT candidates (FSM has not pinned yet).
            for box in self._last_viz_boxes[:8]:
                bbox, conf, tid = box[0], box[1], box[2]
                mode = box[3] if len(box) > 3 else ''
                z_d = box[4] if len(box) > 4 else None
                z_m = box[5] if len(box) > 5 else None
                self._viz_draw_one_box(
                    vis,
                    tuple(int(v) for v in bbox),
                    int(tid),
                    float(conf),
                    str(mode),
                    z_d,
                    z_m,
                    locked=False,
                )
        # Optical principal point (cx, cy).
        if self._k is not None:
            cx = int(round(float(self._k[0, 2])))
            cy = int(round(float(self._k[1, 2])))
            arm = 14
            col = (255, 60, 0)
            cv2.line(vis, (cx - arm, cy), (cx + arm, cy), col, 2, cv2.LINE_AA)
            cv2.line(vis, (cx, cy - arm), (cx, cy + arm), col, 2, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 3, col, -1, cv2.LINE_AA)
            cv2.putText(
                vis, f'pp({cx},{cy})', (cx + arm + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        self._viz_pub.publish(_rgb_to_imgmsg(vis, stamp, self._camera_frame))

    def _tick(self) -> None:
        self._maybe_load_bbox_file()
        stamp = self.get_clock().now().to_msg()
        if self._rgb is None or self._depth is None or self._k is None:
            arr = DetectedBerryArray()
            arr.header.stamp = stamp
            arr.header.frame_id = self._camera_frame
            self._pub.publish(arr)
            return

        detections, msg = self._detect_cam_poses()
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._camera_frame, rclpy.time.Time())
            T_base_cam = matrix_from_tf(tf)
            out_frame = self._base_frame
        except Exception as exc:
            self.get_logger().warn(
                f'TF {self._base_frame}->{self._camera_frame} failed ({exc})')
            T_base_cam = None
            out_frame = self._camera_frame

        header = PoseStamped().header
        header.frame_id = out_frame
        header.stamp = stamp

        berries: List[DetectedBerry] = []
        for det in detections:
            if det.mode == 'base_coast':
                T_out = det.pose_cam  # already base xyz in T
            else:
                T_out = T_base_cam @ det.pose_cam if T_base_cam is not None else det.pose_cam
            b = DetectedBerry()
            b.header = header
            # base_coast must advertise base_link even if TF failed this tick.
            if det.mode == 'base_coast':
                b.header.frame_id = self._base_frame
            b.confidence = float(det.confidence)
            b.track_id = int(det.track_id)
            b.depth_mode = str(det.mode or '')
            b.z_depth_m = float(det.z_depth) if det.z_depth is not None else -1.0
            b.z_mono_m = float(det.z_mono) if det.z_mono is not None else -1.0
            x0, y0, x1, y1 = [int(v) for v in det.bbox_xyxy]
            # Live tracks: bbox center. base_coast: NEVER reuse frozen pixels
            # (camera moved → old box sticks to wall). Reproject base XYZ → UV.
            if det.mode == 'base_coast' and T_base_cam is not None and self._k is not None:
                try:
                    T_cam_base = np.linalg.inv(T_base_cam)
                    p_b = np.array(
                        [float(T_out[0, 3]), float(T_out[1, 3]),
                         float(T_out[2, 3]), 1.0],
                        dtype=np.float64)
                    p_c = T_cam_base @ p_b
                    zc = float(p_c[2])
                    if zc > 0.05:
                        fx, fy = float(self._k[0, 0]), float(self._k[1, 1])
                        cx, cy = float(self._k[0, 2]), float(self._k[1, 2])
                        b.image_u = fx * float(p_c[0]) / zc + cx
                        b.image_v = fy * float(p_c[1]) / zc + cy
                        # Tiny viz box at reprojected UV (not glued pixels).
                        u_i, v_i = int(round(b.image_u)), int(round(b.image_v))
                        half = 8
                        x0, y0 = u_i - half, v_i - half
                        x1, y1 = u_i + half, v_i + half
                    else:
                        b.image_u = -1.0
                        b.image_v = -1.0
                except Exception:
                    b.image_u = -1.0
                    b.image_v = -1.0
            else:
                # Locked live: mask-centroid UV (same as pose), not bbox center.
                if (
                    self._fsm_locked_track_id is not None
                    and int(det.track_id) == int(self._fsm_locked_track_id)
                    and self._k is not None
                    and float(det.pose_cam[2, 3]) > 1e-6
                ):
                    uu, vv = self._cd_mask_uv(det, self._k)
                    b.image_u = float(uu)
                    b.image_v = float(vv)
                else:
                    b.image_u = 0.5 * float(x0 + x1)
                    b.image_v = 0.5 * float(y0 + y1)
            pose_header = b.header
            b.pose = _pose_to_msg(T_out, pose_header)
            berries.append(b)
            if T_base_cam is not None or det.mode == 'base_coast':
                tid = int(det.track_id)
                # Seed/update world coast from live 3D (mono or RGB-D pose).
                if (
                    tid >= 0
                    and det.mode != 'base_coast'
                    and tid != self._fsm_locked_track_id
                ):
                    # Frozen lock P stays put — live depth must not move the
                    # reproject pin (that was the off-axis z error).
                    self._track_base_xyz[tid] = np.array(
                        [float(T_out[0, 3]), float(T_out[1, 3]), float(T_out[2, 3])],
                        dtype=np.float64)
                    self._track_base_t[tid] = time.time()
                if (
                    self._fsm_locked_track_id is not None
                    and tid == self._fsm_locked_track_id
                    and det.mode != 'base_coast'
                ):
                    self._lock_bbox_xyxy = det.bbox_xyxy

        arr = DetectedBerryArray()
        arr.header = header
        arr.berries = berries
        self._pub.publish(arr)
        self._publish_detection_viz(stamp, msg or '')
        if msg:
            self.get_logger().debug(msg)


def main() -> None:
    rclpy.init()
    node = FineDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
