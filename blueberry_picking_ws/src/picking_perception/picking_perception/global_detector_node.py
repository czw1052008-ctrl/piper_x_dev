"""Global coarse detection — fixed DaBai RGB-D + YOLO, topic stream.

Pose uses registered depth (not mono diameter). Mono is diagnostics-only.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from picking_msgs.msg import DetectedBerry, DetectedBerryArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from picking_perception.berry_tracker import BerryTracker
from picking_perception.perception_utils import matrix_from_tf, transform_xyz
from picking_perception.segmentation import filter_berry_mask, segment_blueberry_hsv
from picking_perception.stable_id_tracker import StableIdTracker, TrackObservation
from picking_perception.yolo_berry_detector import (
    TrackedYoloDetection,
    YoloBerryDetector,
    YoloDetection,
)


def _image_to_rgb(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ('rgb8',):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc in ('bgr8',):
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return bgr[:, :, ::-1].copy()
    if enc in ('yuv422_yuy2', 'yuyv', 'yuyv422'):
        import cv2  # type: ignore
        yuyv = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
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


class GlobalDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('global_detector_node')
        self.declare_parameter('berry_diameter_m', 0.08)
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('image_topic', '/camera_fixed/color/image_raw')
        self.declare_parameter('depth_topic', '/camera_fixed/depth/image_raw')
        self.declare_parameter('info_topic', '/camera_fixed/color/camera_info')
        self.declare_parameter('camera_frame', 'camera_fixed_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_mask_px', 100)
        self.declare_parameter('max_mask_px', 25000)
        self.declare_parameter('viz_topic', '/perception/global/detection_viz')
        self.declare_parameter('publish_viz', True)
        self.declare_parameter('mask_source', 'yolo')
        self.declare_parameter('yolo_model', 'yoloe-11s-seg-pf.pt')
        self.declare_parameter('yolo_classes', 'blueberry,blueberries,berry')
        self.declare_parameter('yolo_conf_threshold', 0.22)
        self.declare_parameter('yolo_min_area_px', 120)
        self.declare_parameter('yolo_max_area_frac', 0.08)
        self.declare_parameter('yolo_min_circularity', 0.35)
        self.declare_parameter('yolo_max_aspect_ratio', 2.5)
        self.declare_parameter('yolo_open_vocab', True)
        self.declare_parameter('yolo_target_class_ids', '')
        self.declare_parameter('max_cluster_detections', 8)
        self.declare_parameter('enable_tracking', True)
        self.declare_parameter('tracker_config', '')
        self.declare_parameter('track_iou_thresh', 0.30)
        self.declare_parameter('track_max_misses', 20)
        self.declare_parameter('track_max_dist_m', 0.25)
        # Pose from RGB-D only (DaBai). No mono diameter pose.
        self.declare_parameter('depth_pose_source', 'depth')
        self.declare_parameter('depth_min_m', 0.20)
        self.declare_parameter('depth_max_m', 2.5)
        self.declare_parameter('min_valid_depth_px', 30)

        self._rgb = None
        self._depth = None
        self._k = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._mask_source = str(self.get_parameter('mask_source').value).lower()
        self._yolo: Optional[YoloBerryDetector] = None
        self._det_mode = 'hsv'
        self._enable_tracking = bool(self.get_parameter('enable_tracking').value)
        self._depth_pose_source = str(
            self.get_parameter('depth_pose_source').value or 'depth').strip().lower()
        self._depth_util = BerryTracker(
            berry_diameter_m=float(self.get_parameter('berry_diameter_m').value),
            depth_min_m=float(self.get_parameter('depth_min_m').value),
            depth_max_m=float(self.get_parameter('depth_max_m').value),
            min_valid_depth_px=int(self.get_parameter('min_valid_depth_px').value),
        )
        self._id_tracker = StableIdTracker(
            iou_thresh=float(self.get_parameter('track_iou_thresh').value),
            max_misses=int(self.get_parameter('track_max_misses').value),
            max_dist_m=float(self.get_parameter('track_max_dist_m').value),
            use_xyz=True,
            prefer_xyz=True,
        )

        if self._mask_source == 'yolo':
            prompts = [
                p.strip()
                for p in str(self.get_parameter('yolo_classes').value).split(',')
                if p.strip()
            ]
            target_ids = [
                int(p.strip()) for p in str(self.get_parameter('yolo_target_class_ids').value).split(',')
                if p.strip()
            ]
            tracker_cfg = str(self.get_parameter('tracker_config').value).strip()
            if not tracker_cfg:
                from ament_index_python.packages import get_package_share_directory
                tracker_cfg = os.path.join(
                    get_package_share_directory('picking_perception'),
                    'config', 'botsort_fixed.yaml')
            self._yolo = YoloBerryDetector(
                model_name=str(self.get_parameter('yolo_model').value),
                class_prompts=prompts,
                conf_threshold=float(self.get_parameter('yolo_conf_threshold').value),
                min_area_px=int(self.get_parameter('yolo_min_area_px').value),
                max_area_frac=float(self.get_parameter('yolo_max_area_frac').value),
                min_circularity=float(self.get_parameter('yolo_min_circularity').value),
                max_aspect_ratio=float(self.get_parameter('yolo_max_aspect_ratio').value),
                max_detections=int(self.get_parameter('max_cluster_detections').value),
                open_vocab=bool(self.get_parameter('yolo_open_vocab').value),
                target_class_ids=target_ids or None,
                tracker_config=tracker_cfg,
            )
            if self._yolo.ready:
                self._det_mode = 'yolo'
                self.get_logger().info(
                    f'global YOLO ready model={self.get_parameter("yolo_model").value} '
                    f'tracking={self._enable_tracking} depth_pose={self._depth_pose_source}')
            else:
                self.get_logger().error(
                    'global YOLO unavailable — publishing empty (HSV fallback removed)')
                self._det_mode = 'none'
        else:
            self._det_mode = 'hsv'
            self.get_logger().warn('global mask_source=hsv (explicit debug; not approved reach path)')

        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter('info_topic').value, self._on_info, qos_profile_sensor_data)
        self._pub = self.create_publisher(DetectedBerryArray, '/perception/global/berries', 10)
        self._publish_viz = bool(self.get_parameter('publish_viz').value)
        self._viz_pub = None
        if self._publish_viz:
            self._viz_pub = self.create_publisher(
                Image, str(self.get_parameter('viz_topic').value), qos_profile_sensor_data)

        hz = max(float(self.get_parameter('publish_hz').value), 0.2)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f'global_detector RGB-D @ {hz:.1f} Hz '
            f'color={self.get_parameter("image_topic").value} '
            f'depth={self.get_parameter("depth_topic").value}')

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

    def _detect(self) -> Tuple[Optional[np.ndarray], List[YoloDetection], str]:
        assert self._rgb is not None
        if self._det_mode == 'yolo' and self._yolo is not None and self._yolo.ready:
            if self._enable_tracking:
                dets: List[YoloDetection] = list(self._yolo.track(self._rgb))
                mode = 'yolo_track'
            else:
                dets = list(self._yolo.detect(self._rgb))
                mode = 'yolo'
            if dets:
                mask = np.zeros(self._rgb.shape[:2], dtype=np.uint8)
                for d in dets:
                    mask = np.maximum(mask, d.mask)
                return mask, dets, mode
            return None, [], f'{mode}_empty'
        if self._det_mode != 'hsv':
            return None, [], 'yolo_unavailable'
        mask = segment_blueberry_hsv(self._rgb)
        min_px = int(self.get_parameter('min_mask_px').value)
        max_px = int(self.get_parameter('max_mask_px').value)
        mask = filter_berry_mask(mask, min_area=min_px, max_area=max_px)
        return mask, [], 'hsv'

    def _depth_in_roi(self, mask: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> Optional[float]:
        assert self._depth is not None
        z = self._depth_util.depth_in_mask(self._depth, mask)
        if z is not None:
            return z
        # Fallback: median depth inside bbox (cluster masks can be sparse).
        x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
        h, w = self._depth.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        patch = self._depth[y0:y1, x0:x1]
        lo = float(self.get_parameter('depth_min_m').value)
        hi = float(self.get_parameter('depth_max_m').value)
        valid = patch[(patch >= lo) & (patch <= hi)]
        if valid.size < int(self.get_parameter('min_valid_depth_px').value):
            return None
        return float(np.median(valid))

    def _berry_from_detection(
        self,
        det: YoloDetection,
        stamp,
        cam_frame: str,
        base_frame: str,
        *,
        track_id: int = -1,
    ) -> Optional[DetectedBerry]:
        if self._depth is None or self._k is None:
            return None
        x0, y0, x1, y1 = det.bbox_xyxy
        ys, xs = np.where(det.mask > 0)
        if xs.size > 0:
            cx, cy = float(xs.mean()), float(ys.mean())
        else:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)

        z_depth = self._depth_in_roi(det.mask, (x0, y0, x1, y1))
        z_mono = -1.0
        if self._k is not None and det.mask is not None:
            try:
                z_mono = float(self._depth_util.mono_depth_from_mask(det.mask, self._k))
            except Exception:
                z_mono = -1.0

        if self._depth_pose_source != 'depth':
            self.get_logger().warn(
                f'depth_pose_source={self._depth_pose_source!r} ignored; global uses RGB-D only')
        if z_depth is None:
            return None

        px, py, pz = self._depth_util.uv_to_cam_xyz(
            int(round(cx)), int(round(cy)), float(z_depth), self._k)

        pose = PoseStamped()
        pose.header.frame_id = cam_frame
        pose.header.stamp = stamp
        pose.pose.position.x = px
        pose.pose.position.y = py
        pose.pose.position.z = pz
        pose.pose.orientation.w = 1.0

        try:
            tf = self._tf_buffer.lookup_transform(base_frame, cam_frame, rclpy.time.Time())
            bx, by, bz = transform_xyz(matrix_from_tf(tf), px, py, pz)
            pose.header.frame_id = base_frame
            pose.pose.position.x = bx
            pose.pose.position.y = by
            pose.pose.position.z = bz
        except Exception as exc:
            self.get_logger().warn(f'TF {base_frame}<-{cam_frame} failed: {exc}')

        if track_id < 0 and isinstance(det, TrackedYoloDetection):
            track_id = int(det.track_id)

        berry = DetectedBerry()
        berry.header = pose.header
        berry.pose = pose
        berry.confidence = float(det.confidence)
        berry.track_id = int(track_id)
        berry.depth_mode = 'depth'
        berry.z_depth_m = float(z_depth)
        berry.z_mono_m = float(z_mono) if z_mono > 0 else -1.0
        berry.image_u = float(cx)
        berry.image_v = float(cy)
        berry.class_id = 1
        berry.bbox_u0 = float(x0)
        berry.bbox_v0 = float(y0)
        berry.bbox_u1 = float(x1)
        berry.bbox_v1 = float(y1)
        berry.center_depth_m = float(z_depth)
        berry.surface_normal_base = [0.0, 0.0, 0.0]
        berry.normal_valid = False
        return berry

    def _assign_stable_ids(self, berries: List[DetectedBerry]) -> None:
        if not self._enable_tracking:
            for i, b in enumerate(berries):
                if int(b.track_id) < 0:
                    b.track_id = i
            return
        obs: List[TrackObservation] = []
        for b in berries:
            p = b.pose.pose.position
            obs.append(TrackObservation(
                bbox_xyxy=(float(b.bbox_u0), float(b.bbox_v0),
                           float(b.bbox_u1), float(b.bbox_v1)),
                confidence=float(b.confidence),
                xyz=(float(p.x), float(p.y), float(p.z)),
                payload=b,
            ))
        for tid, o in self._id_tracker.update(obs):
            b = o.payload
            assert isinstance(b, DetectedBerry)
            b.track_id = int(tid)

    def _tick(self) -> None:
        if self._rgb is None:
            return
        if self._depth is None or self._k is None:
            # Wait for RGB-D; do not invent mono poses.
            return

        stamp = self.get_clock().now().to_msg()
        cam_frame = self.get_parameter('camera_frame').value
        base_frame = self.get_parameter('base_frame').value
        min_px = int(self.get_parameter('min_mask_px').value)
        max_n = max(1, int(self.get_parameter('max_cluster_detections').value))

        mask, dets, mode = self._detect()
        if mask is None or int(mask.sum()) < min_px:
            out = DetectedBerryArray()
            out.header.stamp = stamp
            out.header.frame_id = base_frame
            self._pub.publish(out)
            self._publish_detection_viz(stamp, cam_frame, mask, dets, mode, None, None)
            return

        berries: List[DetectedBerry] = []
        viz_dets = dets
        cx, cy, radius = 0, 0, 8
        if dets:
            ranked = sorted(dets, key=lambda d: -float(d.confidence))[:max_n]
            for det in ranked:
                b = self._berry_from_detection(det, stamp, cam_frame, base_frame)
                if b is not None:
                    berries.append(b)
            if berries:
                self._assign_stable_ids(berries)
                best = ranked[0]
                x0, y0, x1, y1 = best.bbox_xyxy
                cx = int(0.5 * (x0 + x1))
                cy = int(0.5 * (y0 + y1))
                radius = int(max(x1 - x0, y1 - y0) * 0.5)
        else:
            # HSV debug path: still require depth.
            ys, xs = np.where(mask > 0)
            cx, cy = int(xs.mean()), int(ys.mean())
            radius = int(max(2.0 * math.sqrt(max(float(mask.sum()), 1.0) / math.pi) * 0.5, 8))
            z = self._depth_util.depth_in_mask(self._depth, mask)
            if z is not None:
                px, py, pz = self._depth_util.uv_to_cam_xyz(cx, cy, float(z), self._k)
                pose = PoseStamped()
                pose.header.frame_id = cam_frame
                pose.header.stamp = stamp
                pose.pose.position.x = px
                pose.pose.position.y = py
                pose.pose.position.z = pz
                pose.pose.orientation.w = 1.0
                try:
                    tf = self._tf_buffer.lookup_transform(base_frame, cam_frame, rclpy.time.Time())
                    bx, by, bz = transform_xyz(matrix_from_tf(tf), px, py, pz)
                    pose.header.frame_id = base_frame
                    pose.pose.position.x = bx
                    pose.pose.position.y = by
                    pose.pose.position.z = bz
                except Exception as exc:
                    self.get_logger().warn(f'TF {base_frame}<-{cam_frame} failed: {exc}')
                berry = DetectedBerry()
                berry.header = pose.header
                berry.pose = pose
                berry.confidence = 0.5
                berry.track_id = -1
                berry.depth_mode = 'depth'
                berry.z_depth_m = float(z)
                berry.class_id = 1
                berry.image_u = float(cx)
                berry.image_v = float(cy)
                berry.center_depth_m = float(z)
                berries = [berry]
                self._assign_stable_ids(berries)
            viz_dets = []

        out_frame = berries[0].pose.header.frame_id if berries else base_frame
        arr = DetectedBerryArray()
        arr.header.stamp = stamp
        arr.header.frame_id = out_frame
        arr.berries = berries
        self._pub.publish(arr)
        self._publish_detection_viz(
            stamp, cam_frame, mask, viz_dets, mode,
            (int(cx), int(cy)) if berries else None,
            int(max(radius, 8)) if berries else None)

    def _publish_detection_viz(
        self, stamp, frame_id: str, mask: Optional[np.ndarray],
        dets: List[YoloDetection], mode: str,
        center: Optional[Tuple[int, int]], radius: Optional[int],
    ) -> None:
        if self._viz_pub is None or self._rgb is None:
            return
        import cv2  # type: ignore

        vis = self._rgb.copy()
        if mask is not None and mask.any():
            tint = vis.copy()
            tint[mask > 0] = (0, 220, 220)
            vis = cv2.addWeighted(vis, 0.65, tint, 0.35, 0)
        for d in dets:
            x0, y0, x1, y1 = d.bbox_xyxy
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                vis, f'{d.confidence:.2f}', (x0, max(y0 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if center is not None and radius is not None:
            cv2.circle(vis, center, radius, (0, 255, 0), 2)
            cv2.circle(vis, center, 3, (0, 255, 0), -1)
        label = f'global {mode} depth'
        if center is not None:
            label += f' @ {center}'
        else:
            label += ': no cluster'
        cv2.putText(
            vis, label, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2, cv2.LINE_AA)
        self._viz_pub.publish(_rgb_to_imgmsg(vis, stamp, frame_id))


def main() -> None:
    rclpy.init()
    node = GlobalDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
