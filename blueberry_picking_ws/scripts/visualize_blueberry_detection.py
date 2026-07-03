#!/usr/bin/env python3
"""Visualize FoundationPose blueberry detections on DaBai raw RGB.

Only draws berries returned by /trigger_fine_detection (no HSV preview boxes).
Publishes to /camera_wrist/color/detection_viz.

Examples:
  bash scripts/run_detection_viz.sh --show
  ros2 run rqt_image_view rqt_image_view /camera_wrist/color/detection_viz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from picking_msgs.srv import TriggerFineDetection
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

_WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _setup_ros_pythonpath() -> None:
    for rel in (
        'install/picking_msgs/lib/python3.12/site-packages',
        'install/picking_perception/lib/python3.12/site-packages',
    ):
        path = os.path.join(_WS_ROOT, rel)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    ros_site = '/opt/ros/jazzy/lib/python3.12/site-packages'
    if os.path.isdir(ros_site) and ros_site not in sys.path:
        sys.path.insert(0, ros_site)


def _image_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc == 'rgb8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc == 'bgr8':
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return bgr[:, :, ::-1].copy()
    raise ValueError(f'unsupported color encoding: {msg.encoding}')


def _rgb_to_imgmsg(rgb: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = 'rgb8'
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = rgb.tobytes()
    return msg


@dataclass
class BerryDetection:
    index: int
    uv: Optional[Tuple[int, int]]
    radius_px: int
    frame_id: str
    xyz: Tuple[float, float, float]
    confidence: float
    message: str
    locked: bool = False


def _selection_metric(
    frame_id: str,
    xyz: Tuple[float, float, float],
    confidence: float,
    *,
    mode: str,
    nearest_frame: str,
) -> float:
    x, y, z = xyz
    if mode == 'nearest':
        if frame_id == nearest_frame or not nearest_frame:
            return z
        return float((x * x + y * y + z * z) ** 0.5)
    return -confidence


def _pick_locked_index(
    berries: List[BerryDetection],
    *,
    mode: str = 'nearest',
    nearest_frame: str,
) -> int:
    if not berries:
        return -1
    metrics = [
        _selection_metric(b.frame_id, b.xyz, b.confidence, mode=mode, nearest_frame=nearest_frame)
        for b in berries
    ]
    return int(min(range(len(berries)), key=lambda i: metrics[i]))


def _berry_radius_px(z_m: float, k: np.ndarray, diameter_m: float = 0.015) -> int:
    if z_m <= 0.05:
        return 18
    fx = k[0, 0]
    return max(int(fx * (diameter_m * 0.5) / z_m), 10)


def _draw_detections(
    rgb: np.ndarray,
    berries: List[BerryDetection],
    locked_index: int = -1,
    grasp_cup_uv: Optional[Tuple[int, int]] = None,
    grasp_eef_uv: Optional[Tuple[int, int]] = None,
    pick_lock_uv: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    vis = rgb.copy()
    other_color = (0, 180, 220)
    locked_color = (80, 255, 80)

    for i, b in enumerate(berries):
        is_locked = (i == locked_index)
        color = locked_color if is_locked else other_color
        thickness = 3 if is_locked else 1
        if b.uv is None:
            label = f'LOCK #{b.index}' if is_locked else f'#{b.index} off-screen'
            cv2.putText(
                vis, label, (12, 40 + b.index * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)
            continue
        u, v = b.uv
        cv2.circle(vis, (u, v), b.radius_px, color, thickness)
        if is_locked:
            cv2.circle(vis, (u, v), b.radius_px + 8, locked_color, 2)
            cv2.drawMarker(vis, (u, v), locked_color, cv2.MARKER_DIAMOND, 14, 2)
        else:
            cv2.drawMarker(vis, (u, v), color, cv2.MARKER_CROSS, 8, 1)
        x, y, z = b.xyz
        prefix = 'LOCK ' if is_locked else ''
        label = f'{prefix}#{b.index} ({x:.2f},{y:.2f},{z:.2f})m c={b.confidence:.2f}'
        cv2.putText(
            vis, label, (u + b.radius_px + 4, max(v - 4, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, thickness, cv2.LINE_AA)

    n = len(berries)
    if locked_index >= 0 and locked_index < n:
        lb = berries[locked_index]
        status = f'{n} berries | LOCK #{lb.index} z={lb.xyz[2]:.2f}m (nearest pick)'
    else:
        status = f'FoundationPose: {n} berries'
    if grasp_cup_uv is not None:
        u, v = grasp_cup_uv
        cv2.rectangle(vis, (u - 10, v - 10), (u + 10, v + 10), (255, 80, 255), 2)
        cv2.putText(
            vis, 'GRASP cup', (u + 12, v + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 80, 255), 1, cv2.LINE_AA)
        status += ' | magenta=planned cup contact'
    if grasp_eef_uv is not None:
        u, v = grasp_eef_uv
        cv2.circle(vis, (u, v), 6, (255, 200, 80), 2)
    if pick_lock_uv is not None:
        u, v = pick_lock_uv
        cv2.circle(vis, (u, v), 22, locked_color, 2)
        cv2.drawMarker(vis, (u, v), locked_color, cv2.MARKER_DIAMOND, 16, 2)
        cv2.putText(
            vis, 'PICK lock', (u + 14, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, locked_color, 2, cv2.LINE_AA)
        status += ' | green=PICK lock (grasp plan)'
    cv2.putText(vis, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


class BlueberryDetectionVisualizer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('blueberry_detection_visualizer')
        self._args = args
        self._rgb: Optional[np.ndarray] = None
        self._k: Optional[np.ndarray] = None
        self._camera_frame = args.camera_frame
        self._berries: List[BerryDetection] = []
        self._locked_index = -1
        self._last_fp_time = 0.0
        self._fp_busy = False
        self._fp_started = 0.0
        self._fp_gen = 0
        self._status = 'waiting for camera'
        self._grasp_cup_xyz: Optional[Tuple[float, float, float]] = None
        self._grasp_cup_frame = 'base_link'
        self._grasp_eef_xyz: Optional[Tuple[float, float, float]] = None
        self._grasp_eef_frame = 'base_link'
        self._pick_locked_xyz: Optional[Tuple[float, float, float]] = None
        self._pick_locked_frame = 'base_link'
        self._pick_locked_time = 0.0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Image, args.color_topic, self._on_rgb, 1)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._on_info, 1)
        self.create_subscription(
            PointStamped, '/pick/suction_cup_contact', self._on_grasp_cup, 1)
        self.create_subscription(
            PoseStamped, '/pick/suction_grasp_target', self._on_grasp_eef, 1)
        self.create_subscription(
            PointStamped, '/pick/locked_berry', self._on_pick_locked, 1)
        self._pub = self.create_publisher(Image, args.publish_topic, 1)
        self.create_timer(1.0 / max(args.rate_hz, 1.0), self._on_timer)

        self._fp_client = self.create_client(TriggerFineDetection, 'trigger_fine_detection')
        if not self._fp_client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error(
                'trigger_fine_detection unavailable — start: bash scripts/run_fine_detector_node.sh')

    def _on_rgb(self, msg: Image) -> None:
        try:
            self._rgb = _image_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)

    def _on_info(self, msg: CameraInfo) -> None:
        if self._k is None:
            self._k = np.array(msg.k).reshape(3, 3)
            if msg.header.frame_id:
                self._camera_frame = msg.header.frame_id

    def _on_grasp_cup(self, msg: PointStamped) -> None:
        p = msg.point
        self._grasp_cup_xyz = (float(p.x), float(p.y), float(p.z))
        self._grasp_cup_frame = msg.header.frame_id or 'base_link'

    def _on_grasp_eef(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._grasp_eef_xyz = (float(p.x), float(p.y), float(p.z))
        self._grasp_eef_frame = msg.header.frame_id or 'base_link'

    def _on_pick_locked(self, msg: PointStamped) -> None:
        p = msg.point
        self._pick_locked_xyz = (float(p.x), float(p.y), float(p.z))
        self._pick_locked_frame = msg.header.frame_id or 'base_link'
        self._pick_locked_time = time.monotonic()

    def _active_lock_xyz(self) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        """Prefer pick loop locked berry (same 3D point as grasp plan)."""
        if (self._pick_locked_xyz is not None
                and time.monotonic() - self._pick_locked_time < 120.0):
            return self._pick_locked_frame, self._pick_locked_xyz
        if self._locked_index < 0 or self._locked_index >= len(self._berries):
            return None
        lb = self._berries[self._locked_index]
        return lb.frame_id, lb.xyz

    def _project_to_uv(self, frame_id: str, xyz: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
        if self._k is None:
            return None
        if frame_id == self._camera_frame:
            x, y, z = xyz
            if z <= 0.05:
                return None
            fx, fy = self._k[0, 0], self._k[1, 1]
            cx, cy = self._k[0, 2], self._k[1, 2]
            return int(fx * x / z + cx), int(fy * y / z + cy)
        try:
            tf = self._tf_buffer.lookup_transform(
                self._camera_frame, frame_id, rclpy.time.Time())
        except Exception:
            return None
        pt = PointStamped()
        pt.header.frame_id = frame_id
        pt.point.x, pt.point.y, pt.point.z = xyz
        cam = do_transform_point(pt, tf).point
        if cam.z <= 0.05:
            return None
        fx, fy = self._k[0, 0], self._k[1, 1]
        cx, cy = self._k[0, 2], self._k[1, 2]
        return int(fx * cam.x / cam.z + cx), int(fy * cam.y / cam.z + cy)

    def _depth_for_radius(self, frame_id: str, xyz: Tuple[float, float, float]) -> float:
        if frame_id == self._camera_frame:
            return max(xyz[2], 0.05)
        uv = self._project_to_uv(frame_id, xyz)
        if uv is None or self._k is None:
            return 0.5
        try:
            tf = self._tf_buffer.lookup_transform(
                self._camera_frame, frame_id, rclpy.time.Time())
        except Exception:
            return 0.5
        pt = PointStamped()
        pt.header.frame_id = frame_id
        pt.point.x, pt.point.y, pt.point.z = xyz
        return max(float(do_transform_point(pt, tf).point.z), 0.05)

    def _call_fp(self) -> None:
        if self._fp_busy or not self._fp_client.service_is_ready():
            return
        self._fp_busy = True
        self._fp_started = time.monotonic()
        self._fp_gen += 1
        gen = self._fp_gen
        req = TriggerFineDetection.Request()
        future = self._fp_client.call_async(req)
        future.add_done_callback(lambda f: self._on_fp_done(f, gen))

    def _on_fp_done(self, future, gen: int) -> None:
        if gen != self._fp_gen:
            return
        self._fp_busy = False
        self._last_fp_time = time.monotonic()
        try:
            res = future.result()
        except Exception as exc:
            self._status = f'FP error: {exc}'
            self.get_logger().error(f'trigger_fine_detection failed: {exc}')
            return
        self._status = res.message
        self._berries = []
        for i, berry in enumerate(res.detected_berries):
            p = berry.pose.pose.position
            frame = berry.pose.header.frame_id or berry.header.frame_id or 'base_link'
            xyz = (p.x, p.y, p.z)
            uv = self._project_to_uv(frame, xyz)
            z_cam = self._depth_for_radius(frame, xyz)
            radius = _berry_radius_px(z_cam, self._k) if self._k is not None else 18
            self._berries.append(BerryDetection(
                index=i + 1, uv=uv, radius_px=radius, frame_id=frame,
                xyz=xyz, confidence=float(berry.confidence), message=res.message))

        locked = int(getattr(res, 'locked_berry_index', -1))
        if locked < 0 or locked >= len(self._berries):
            locked = _pick_locked_index(
                self._berries,
                mode=self._args.selection_mode,
                nearest_frame=self._args.nearest_frame,
            )
        self._locked_index = locked
        for i, b in enumerate(self._berries):
            b.locked = (i == self._locked_index)
        n = len(self._berries)
        lock_msg = ''
        if self._locked_index >= 0:
            lb = self._berries[self._locked_index]
            lock_msg = f' lock=#{lb.index} z={lb.xyz[2]:.2f}m'
        self.get_logger().info(f'FP: success={res.success} n={n}{lock_msg} msg={res.message}')

    def _filter_display_berries(self) -> List[BerryDetection]:
        """Always show locked target; hide low-confidence false positives."""
        if self._locked_index < 0:
            return [b for b in self._berries if b.confidence >= self._args.min_display_conf]
        out: List[BerryDetection] = []
        for i, b in enumerate(self._berries):
            if i == self._locked_index or b.confidence >= self._args.min_display_conf:
                out.append(b)
        return out

    def _display_lock_index(self) -> int:
        shown = self._filter_display_berries()
        if self._locked_index < 0 or not shown:
            return -1
        locked = self._berries[self._locked_index]
        for i, b in enumerate(shown):
            if b.index == locked.index:
                return i
        return -1

    def _on_timer(self) -> None:
        if self._rgb is None:
            return
        if self._fp_busy and (time.monotonic() - self._fp_started) >= self._args.fp_timeout:
            self._fp_gen += 1
            self._fp_busy = False
            self._last_fp_time = time.monotonic()
            self._status = 'FP timeout'
            self.get_logger().error('trigger_fine_detection timed out')
        elif self._k is not None and not self._fp_busy:
            if self._last_fp_time == 0.0 or (
                    time.monotonic() - self._last_fp_time) >= self._args.fp_interval:
                self._call_fp()

        pick_lock_uv = None
        lock_xyz = self._active_lock_xyz()
        if lock_xyz is not None:
            frame_id, xyz = lock_xyz
            pick_lock_uv = self._project_to_uv(frame_id, xyz)

        vis = _draw_detections(
            self._rgb,
            self._filter_display_berries(),
            self._display_lock_index(),
            self._project_to_uv(self._grasp_cup_frame, self._grasp_cup_xyz)
            if self._grasp_cup_xyz is not None else None,
            self._project_to_uv(self._grasp_eef_frame, self._grasp_eef_xyz)
            if self._grasp_eef_xyz is not None else None,
            pick_lock_uv=pick_lock_uv,
        )
        if self._status:
            cv2.putText(
                vis, self._status[:80], (8, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        stamp = self.get_clock().now().to_msg()
        self._pub.publish(_rgb_to_imgmsg(vis, stamp, self._camera_frame))

        if self._args.show:
            cv2.imshow('blueberry_detection', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        if self._args.save_dir:
            os.makedirs(self._args.save_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(self._args.save_dir, 'latest_detection_viz.png'),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main() -> int:
    _setup_ros_pythonpath()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--color-topic', default='/camera_wrist/color/image_raw')
    parser.add_argument('--camera-info-topic', default='/camera_wrist/color/camera_info')
    parser.add_argument('--publish-topic', default='/camera_wrist/color/detection_viz')
    parser.add_argument('--camera-frame', default='camera_wrist_color_optical_frame')
    parser.add_argument('--rate-hz', type=float, default=10.0)
    parser.add_argument('--fp-interval', type=float, default=8.0,
                        help='Seconds between FoundationPose detections')
    parser.add_argument('--fp-timeout', type=float, default=120.0)
    parser.add_argument('--min-display-conf', type=float, default=0.18,
                        help='Hide detections below this confidence in viz')
    parser.add_argument('--selection-mode', default='nearest',
                        choices=('nearest', 'highest_confidence'))
    parser.add_argument('--nearest-frame', default='camera_wrist_color_optical_frame')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save-dir', default='')
    args = parser.parse_args()

    rclpy.init()
    node = BlueberryDetectionVisualizer(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
