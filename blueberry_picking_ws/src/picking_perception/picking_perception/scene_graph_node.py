"""Fuse global clusters + fine berries into /perception/scene_graph.

Authoritative perception topic for planner / bag.

World-frame tracking policy (moving camera, static targets):
  1. Associate detections primarily by base_link 3D distance
  2. On YOLO miss: coast last xyz for max_misses frames (same track_id)
  3. On re-detect near coasted xyz: reuse id

Raw /perception/global|fine/berries remain available for debug.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from picking_msgs.msg import (
    DetectedBerry,
    DetectedBerryArray,
    PerceptionScene,
    SceneBerry,
    SceneCluster,
)
from picking_perception.stable_id_tracker import (
    StableIdTracker,
    TrackObservation,
    TrackUpdate,
)

FLAG_NO_CLUSTERS = 1 << 0
FLAG_NO_BERRIES = 1 << 1
FLAG_GLOBAL_STALE = 1 << 2
FLAG_FINE_STALE = 1 << 3
FLAG_HAS_COAST = 1 << 4

MAX_CLUSTERS = 8
MAX_BERRIES = 10


def _xyz(b: DetectedBerry) -> Optional[Tuple[float, float, float]]:
    try:
        p = b.pose.pose.position
        x, y, z = float(p.x), float(p.y), float(p.z)
        if all(math.isfinite(v) for v in (x, y, z)):
            return x, y, z
    except Exception:
        return None
    return None


class SceneGraphNode(Node):
    def __init__(self) -> None:
        super().__init__('scene_graph_node')
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('stale_s', 0.5)
        self.declare_parameter('global_topic', '/perception/global/berries')
        self.declare_parameter('fine_topic', '/perception/fine/berries')
        self.declare_parameter('scene_topic', '/perception/scene_graph')
        self.declare_parameter('enable_coast', True)
        # ~2 s coast @ 10 Hz
        self.declare_parameter('cluster_max_misses', 20)
        self.declare_parameter('berry_max_misses', 25)
        self.declare_parameter('cluster_max_dist_m', 0.25)
        self.declare_parameter('berry_max_dist_m', 0.12)

        self._global: Optional[DetectedBerryArray] = None
        self._fine: Optional[DetectedBerryArray] = None
        self._global_t = 0.0
        self._fine_t = 0.0

        self._cluster_tracker = StableIdTracker(
            iou_thresh=0.25,
            max_misses=int(self.get_parameter('cluster_max_misses').value),
            max_dist_m=float(self.get_parameter('cluster_max_dist_m').value),
            use_xyz=True,
            prefer_xyz=True,
        )
        self._berry_tracker = StableIdTracker(
            iou_thresh=0.20,
            max_misses=int(self.get_parameter('berry_max_misses').value),
            max_dist_m=float(self.get_parameter('berry_max_dist_m').value),
            use_xyz=True,
            prefer_xyz=True,
        )

        self.create_subscription(
            DetectedBerryArray, str(self.get_parameter('global_topic').value),
            self._on_global, 10)
        self.create_subscription(
            DetectedBerryArray, str(self.get_parameter('fine_topic').value),
            self._on_fine, 10)
        self._pub = self.create_publisher(
            PerceptionScene, str(self.get_parameter('scene_topic').value), 10)

        hz = max(float(self.get_parameter('publish_hz').value), 0.5)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f'scene_graph → {self.get_parameter("scene_topic").value} @ {hz:.1f} Hz '
            f'coast={bool(self.get_parameter("enable_coast").value)}')

    def _on_global(self, msg: DetectedBerryArray) -> None:
        self._global = msg
        self._global_t = time.time()

    def _on_fine(self, msg: DetectedBerryArray) -> None:
        self._fine = msg
        self._fine_t = time.time()

    def _obs_from_detected(self, berries: List[DetectedBerry]) -> List[TrackObservation]:
        obs: List[TrackObservation] = []
        for b in berries:
            xyz = _xyz(b)
            if xyz is None:
                continue
            u0 = float(getattr(b, 'bbox_u0', -1.0))
            v0 = float(getattr(b, 'bbox_v0', -1.0))
            u1 = float(getattr(b, 'bbox_u1', -1.0))
            v1 = float(getattr(b, 'bbox_v1', -1.0))
            if min(u0, v0, u1, v1) < 0:
                # Synthetic tiny box around image_u/v if present.
                iu = float(getattr(b, 'image_u', -1.0))
                iv = float(getattr(b, 'image_v', -1.0))
                if iu < 0 or iv < 0:
                    continue
                u0, v0, u1, v1 = iu - 8, iv - 8, iu + 8, iv + 8
            obs.append(TrackObservation(
                bbox_xyxy=(u0, v0, u1, v1),
                confidence=float(b.confidence),
                xyz=xyz,
                payload=b,
            ))
        return obs

    def _cluster_from_update(self, u: TrackUpdate) -> Optional[SceneCluster]:
        if u.xyz is None:
            return None
        c = SceneCluster()
        c.id = int(u.track_id)
        c.position = [float(u.xyz[0]), float(u.xyz[1]), float(u.xyz[2])]
        c.confidence = float(u.confidence)
        c.bbox_u0 = float(u.bbox_xyxy[0])
        c.bbox_v0 = float(u.bbox_xyxy[1])
        c.bbox_u1 = float(u.bbox_xyxy[2])
        c.bbox_v1 = float(u.bbox_xyxy[3])
        c.track_source = 'coast' if u.coast else 'live'
        return c

    def _berry_from_update(self, u: TrackUpdate) -> Optional[SceneBerry]:
        if u.xyz is None:
            return None
        sb = SceneBerry()
        sb.id = int(u.track_id)
        sb.position = [float(u.xyz[0]), float(u.xyz[1]), float(u.xyz[2])]
        sb.confidence = float(u.confidence)
        sb.visible_wrist = True
        sb.visible_global = False
        sb.bbox_u0 = float(u.bbox_xyxy[0])
        sb.bbox_v0 = float(u.bbox_xyxy[1])
        sb.bbox_u1 = float(u.bbox_xyxy[2])
        sb.bbox_v1 = float(u.bbox_xyxy[3])
        sb.image_u = 0.5 * (sb.bbox_u0 + sb.bbox_u1)
        sb.image_v = 0.5 * (sb.bbox_v0 + sb.bbox_v1)
        sb.surface_normal = [0.0, 0.0, 0.0]
        sb.normal_valid = False
        sb.z_depth_m = -1.0
        sb.z_mono_m = -1.0
        sb.center_depth_m = -1.0
        sb.track_source = 'coast' if u.coast else 'live'
        sb.pick_role = SceneBerry.PICK_ROLE_PENDING

        if u.coast:
            sb.depth_mode = 'coast'
            # Keep last live depth/normal if payload was a DetectedBerry.
            b = u.payload
            if isinstance(b, DetectedBerry):
                sb.z_depth_m = float(getattr(b, 'z_depth_m', -1.0))
                sb.z_mono_m = float(getattr(b, 'z_mono_m', -1.0))
                sb.center_depth_m = float(getattr(b, 'center_depth_m', -1.0))
                raw_n = getattr(b, 'surface_normal_base', None)
                if raw_n is not None:
                    try:
                        sb.surface_normal = [
                            float(raw_n[0]), float(raw_n[1]), float(raw_n[2])]
                        sb.normal_valid = bool(getattr(b, 'normal_valid', False))
                    except (TypeError, IndexError, ValueError):
                        pass
            return sb

        b = u.payload
        if isinstance(b, DetectedBerry):
            sb.depth_mode = str(getattr(b, 'depth_mode', '') or 'live')
            sb.z_depth_m = float(getattr(b, 'z_depth_m', -1.0))
            sb.z_mono_m = float(getattr(b, 'z_mono_m', -1.0))
            cd = float(getattr(b, 'center_depth_m', -1.0))
            if cd <= 0.05:
                cd = sb.z_depth_m if sb.z_depth_m > 0.05 else (
                    sb.z_mono_m if sb.z_mono_m > 0.05 else -1.0)
            sb.center_depth_m = cd
            iu = float(getattr(b, 'image_u', -1.0))
            iv = float(getattr(b, 'image_v', -1.0))
            if iu >= 0 and iv >= 0:
                sb.image_u, sb.image_v = iu, iv
            raw_n = getattr(b, 'surface_normal_base', None)
            if raw_n is None:
                n = [0.0, 0.0, 0.0]
            else:
                try:
                    n = [float(raw_n[0]), float(raw_n[1]), float(raw_n[2])]
                except (TypeError, IndexError, ValueError):
                    n = [0.0, 0.0, 0.0]
            if len(n) >= 3:
                sb.surface_normal = [float(n[0]), float(n[1]), float(n[2])]
                sb.normal_valid = bool(getattr(b, 'normal_valid', False))
            # Prefer detector bbox if valid.
            u0 = float(getattr(b, 'bbox_u0', -1.0))
            if u0 >= 0:
                sb.bbox_u0 = u0
                sb.bbox_v0 = float(b.bbox_v0)
                sb.bbox_u1 = float(b.bbox_u1)
                sb.bbox_v1 = float(b.bbox_v1)
        else:
            sb.depth_mode = 'live'
        return sb

    def _track_clusters(self) -> Tuple[List[SceneCluster], bool]:
        raw = list(self._global.berries[:MAX_CLUSTERS]) if self._global else []
        obs = self._obs_from_detected(raw)
        enable_coast = bool(self.get_parameter('enable_coast').value)
        updates = self._cluster_tracker.update_full(obs)
        out: List[SceneCluster] = []
        has_coast = False
        for u in updates:
            if u.coast and not enable_coast:
                continue
            if u.coast:
                has_coast = True
            c = self._cluster_from_update(u)
            if c is not None:
                out.append(c)
            if len(out) >= MAX_CLUSTERS:
                break
        return out, has_coast

    def _track_berries(self) -> Tuple[List[SceneBerry], bool]:
        raw = list(self._fine.berries[:MAX_BERRIES]) if self._fine else []
        obs = self._obs_from_detected(raw)
        enable_coast = bool(self.get_parameter('enable_coast').value)
        updates = self._berry_tracker.update_full(obs)
        out: List[SceneBerry] = []
        has_coast = False
        for u in updates:
            if u.coast and not enable_coast:
                continue
            if u.coast:
                has_coast = True
            sb = self._berry_from_update(u)
            if sb is not None:
                out.append(sb)
            if len(out) >= MAX_BERRIES:
                break
        return out, has_coast

    def _tick(self) -> None:
        now = time.time()
        stale_s = float(self.get_parameter('stale_s').value)
        flags = 0
        if self._global is None or (now - self._global_t) > stale_s:
            flags |= FLAG_GLOBAL_STALE
        if self._fine is None or (now - self._fine_t) > stale_s:
            flags |= FLAG_FINE_STALE

        # Even if source is stale/empty: still tick trackers with [] to advance coast.
        clusters, c_coast = self._track_clusters()
        berries, b_coast = self._track_berries()
        if c_coast or b_coast:
            flags |= FLAG_HAS_COAST
        if not clusters:
            flags |= FLAG_NO_CLUSTERS
        if not berries:
            flags |= FLAG_NO_BERRIES

        msg = PerceptionScene()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.clusters = clusters
        msg.berries = berries
        msg.flags = flags
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = SceneGraphNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
