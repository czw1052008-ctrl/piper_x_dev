"""Fine detection — wrist RGB-D YOLO+depth, streaming topics (FP optional)."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from picking_msgs.msg import DetectedBerry, DetectedBerryArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from picking_perception.berry_tracker import BerryTracker
from picking_perception.segmentation import filter_berry_mask, segment_blueberry_hsv
from picking_perception.yolo_berry_detector import YoloBerryDetector


class _NoFoundationPose:
    """Stub so BerryTracker can run without FoundationPose."""

    ready = False

    def detect_all(self, *_args, **_kwargs):
        return []


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


def _matrix_from_tf(transform) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


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
        self.declare_parameter('yolo_min_area_px', 120)
        self.declare_parameter('yolo_max_area_frac', 0.04)
        self.declare_parameter('yolo_min_circularity', 0.45)
        self.declare_parameter('yolo_max_aspect_ratio', 2.0)
        self.declare_parameter('yolo_open_vocab', True)
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('berry_diameter_m', 0.015)
        self.declare_parameter('depth_min_m', 0.03)
        self.declare_parameter('depth_max_m', 1.8)
        self.declare_parameter('track_lost_max_frames', 25)
        self.declare_parameter('publish_hz', 3.0)
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
            self.get_logger().info('FoundationPose disabled — YOLO + depth / mono prior')

        self._mask_source = str(self.get_parameter('mask_source').value).lower()
        self._enable_tracking = bool(self.get_parameter('enable_tracking').value)
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
            )
            if self._yolo.ready:
                self.get_logger().info(f'YOLO ready prompts={prompts}')
            else:
                self.get_logger().warn(
                    'YOLO unavailable — falling back to HSV masks '
                    f'(model={self.get_parameter("yolo_model").value})')
                self._mask_source = 'hsv'

        self._tracker = BerryTracker(
            berry_diameter_m=float(self.get_parameter('berry_diameter_m').value),
            depth_min_m=float(self.get_parameter('depth_min_m').value),
            depth_max_m=float(self.get_parameter('depth_max_m').value),
            track_lost_max_frames=int(self.get_parameter('track_lost_max_frames').value),
        )

        self._base_frame = self.get_parameter('base_frame').value
        self._camera_frame = self.get_parameter('camera_frame').value
        self._rgb = None
        self._depth = None
        self._k = None
        self._target_lock: Optional[DetectedBerry] = None

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
        self._last_viz_boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
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

    def _detect_cam_poses(self) -> Tuple[List[Tuple[np.ndarray, float]], str]:
        rgb, depth, k = self._rgb, self._depth, self._k
        assert rgb is not None and depth is not None and k is not None
        self._last_viz_boxes = []

        if self._mask_source == 'yolo' and self._enable_tracking and self._yolo and self._yolo.ready:
            yolo_dets = self._yolo.detect(rgb)
            self._last_viz_boxes = [(d.bbox_xyxy, float(d.confidence)) for d in yolo_dets]
            tracked, _lock_idx, track_msg = self._tracker.process(
                yolo_dets, rgb, depth, k, self._fp, reset_lock=False)
            if not tracked:
                return [], track_msg
            self._last_viz_boxes = [
                (tuple(int(v) for v in t.bbox), float(t.confidence)) for t in tracked]
            return [(t.pose_cam, t.confidence) for t in tracked], track_msg

        if self._mask_source == 'yolo' and self._yolo and self._yolo.ready:
            detections = []
            for det in self._yolo.detect(rgb):
                self._last_viz_boxes.append((det.bbox_xyxy, float(det.confidence)))
                z = self._tracker.depth_in_mask(depth, det.mask)
                if z is None:
                    z = self._tracker.mono_depth_from_mask(det.mask, k)
                ys, xs = np.where(det.mask > 0)
                u, v = int(xs.mean()), int(ys.mean())
                x, y, zz = self._tracker.uv_to_cam_xyz(u, v, z, k)
                T = np.eye(4)
                T[0, 3], T[1, 3], T[2, 3] = x, y, zz
                detections.append((T, float(det.confidence)))
            if not detections:
                return [], 'YOLO found no blueberries'
            return detections, 'YOLO depth'

        mask = filter_berry_mask(segment_blueberry_hsv(rgb))
        if mask.sum() < 100:
            return [], 'HSV mask too small'
        z = self._tracker.depth_in_mask(depth, mask)
        if z is None:
            z = self._tracker.mono_depth_from_mask(mask, k)
        ys, xs = np.where(mask > 0)
        u, v = int(xs.mean()), int(ys.mean())
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        self._last_viz_boxes = [((x0, y0, x1, y1), 0.4)]
        x, y, zz = self._tracker.uv_to_cam_xyz(u, v, z, k)
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = x, y, zz
        return [(T, 0.4)], 'HSV mono/depth'

    def _publish_detection_viz(self, stamp, label: str) -> None:
        if self._viz_pub is None or self._rgb is None:
            return
        # Lazy import keeps node import light when viz disabled.
        import cv2  # type: ignore

        vis = self._rgb.copy()
        for i, (bbox, conf) in enumerate(self._last_viz_boxes):
            x0, y0, x1, y1 = [int(v) for v in bbox]
            color = (0, 255, 0) if i == 0 else (255, 200, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                vis, f'{i}:{conf:.2f}', (x0, max(y0 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(
            vis, f'YOLO/fine n={len(self._last_viz_boxes)} {label}'[:60],
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2, cv2.LINE_AA)
        self._viz_pub.publish(_rgb_to_imgmsg(vis, stamp, self._camera_frame))

    def _tick(self) -> None:
        stamp = self.get_clock().now().to_msg()
        # Always publish (even empty) so bringup readiness and FSM can proceed.
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
            T_base_cam = _matrix_from_tf(tf)
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
        for pose_cam, score in detections:
            T_out = T_base_cam @ pose_cam if T_base_cam is not None else pose_cam
            b = DetectedBerry()
            b.header = header
            b.confidence = float(score)
            b.pose = _pose_to_msg(T_out, header)
            berries.append(b)

        # Prefer berry closest to locked global target when available
        if self._target_lock is not None and berries and \
                self._target_lock.pose.header.frame_id == out_frame:
            lx = self._target_lock.pose.pose.position.x
            ly = self._target_lock.pose.pose.position.y
            lz = self._target_lock.pose.pose.position.z

            def _dist2(b: DetectedBerry) -> float:
                dx = b.pose.pose.position.x - lx
                dy = b.pose.pose.position.y - ly
                dz = b.pose.pose.position.z - lz
                return dx * dx + dy * dy + dz * dz

            berries = sorted(berries, key=_dist2)

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
