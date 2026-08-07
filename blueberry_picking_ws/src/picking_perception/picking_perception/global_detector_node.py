"""Global coarse detection — fixed mono camera, YOLO (no HSV auto-fallback), topic stream."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from picking_msgs.msg import DetectedBerry, DetectedBerryArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from picking_perception.perception_utils import matrix_from_tf, transform_xyz
from picking_perception.segmentation import filter_berry_mask, segment_blueberry_hsv
from picking_perception.yolo_berry_detector import YoloBerryDetector, YoloDetection


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
        self.declare_parameter('berry_diameter_m', 0.015)
        self.declare_parameter('camera_focal_px', 500.0)
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('image_topic', '/camera_fixed/image_raw')
        self.declare_parameter('camera_frame', 'camera_fixed_optical_frame')
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

        self._rgb = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._mask_source = str(self.get_parameter('mask_source').value).lower()
        self._yolo: Optional[YoloBerryDetector] = None
        self._det_mode = 'hsv'

        if self._mask_source == 'yolo':
            prompts = [
                p.strip()
                for p in str(self.get_parameter('yolo_classes').value).split(',')
                if p.strip()
            ]
            self._yolo = YoloBerryDetector(
                model_name=str(self.get_parameter('yolo_model').value),
                class_prompts=prompts,
                conf_threshold=float(self.get_parameter('yolo_conf_threshold').value),
                min_area_px=int(self.get_parameter('yolo_min_area_px').value),
                max_area_frac=float(self.get_parameter('yolo_max_area_frac').value),
                min_circularity=float(self.get_parameter('yolo_min_circularity').value),
                max_aspect_ratio=float(self.get_parameter('yolo_max_aspect_ratio').value),
                open_vocab=bool(self.get_parameter('yolo_open_vocab').value),
            )
            if self._yolo.ready:
                self._det_mode = 'yolo'
                self.get_logger().info(
                    f'global YOLO ready model={self.get_parameter("yolo_model").value}')
            else:
                self.get_logger().error(
                    'global YOLO unavailable — publishing empty (HSV fallback removed)')
                self._det_mode = 'none'
        else:
            # Explicit hsv only for offline debug; approved reach path is YOLO.
            self._det_mode = 'hsv'
            self.get_logger().warn('global mask_source=hsv (explicit debug; not approved reach path)')

        image_topic = self.get_parameter('image_topic').value
        self.create_subscription(
            Image, image_topic, self._on_rgb, qos_profile_sensor_data)
        self._pub = self.create_publisher(DetectedBerryArray, '/perception/global/berries', 10)
        self._publish_viz = bool(self.get_parameter('publish_viz').value)
        self._viz_pub = None
        if self._publish_viz:
            self._viz_pub = self.create_publisher(
                Image, str(self.get_parameter('viz_topic').value), qos_profile_sensor_data)

        hz = max(float(self.get_parameter('publish_hz').value), 0.2)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f'global_detector streaming /perception/global/berries @ {hz:.1f} Hz '
            f'(image={image_topic}, mode={self._det_mode})')

    def _on_rgb(self, msg: Image) -> None:
        try:
            self._rgb = _image_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _detect(self) -> Tuple[Optional[np.ndarray], List[YoloDetection], str]:
        assert self._rgb is not None
        if self._det_mode == 'yolo' and self._yolo is not None and self._yolo.ready:
            dets = self._yolo.detect(self._rgb)
            if dets:
                mask = np.zeros(self._rgb.shape[:2], dtype=np.uint8)
                for d in dets:
                    mask = np.maximum(mask, d.mask)
                return mask, dets, 'yolo'
            return None, [], 'yolo_empty'
        if self._det_mode != 'hsv':
            return None, [], 'yolo_unavailable'
        mask = segment_blueberry_hsv(self._rgb)
        min_px = int(self.get_parameter('min_mask_px').value)
        max_px = int(self.get_parameter('max_mask_px').value)
        mask = filter_berry_mask(mask, min_area=min_px, max_area=max_px)
        return mask, [], 'hsv'

    def _tick(self) -> None:
        if self._rgb is None:
            return

        stamp = self.get_clock().now().to_msg()
        cam_frame = self.get_parameter('camera_frame').value
        base_frame = self.get_parameter('base_frame').value
        min_px = int(self.get_parameter('min_mask_px').value)

        mask, dets, mode = self._detect()
        if mask is None or int(mask.sum()) < min_px:
            out = DetectedBerryArray()
            out.header.stamp = stamp
            out.header.frame_id = base_frame
            self._pub.publish(out)
            self._publish_detection_viz(stamp, cam_frame, mask, dets, mode, None, None)
            return

        if dets:
            best = max(dets, key=lambda d: d.confidence)
            x0, y0, x1, y1 = best.bbox_xyxy
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            diameter_px = float(max(x1 - x0, y1 - y0))
            conf = float(best.confidence)
            viz_dets = dets
        else:
            ys, xs = np.where(mask > 0)
            cx, cy = float(xs.mean()), float(ys.mean())
            area = max(float(mask.sum()), 1.0)
            diameter_px = 2.0 * math.sqrt(area / math.pi)
            conf = 0.5
            viz_dets = []

        focal = float(self.get_parameter('camera_focal_px').value)
        berry_d = float(self.get_parameter('berry_diameter_m').value)
        distance = focal * berry_d / max(diameter_px, 1.0)

        pose = PoseStamped()
        pose.header.frame_id = cam_frame
        pose.header.stamp = stamp
        pose.pose.position.x = (cx - self._rgb.shape[1] / 2) * distance / focal
        pose.pose.position.y = (cy - self._rgb.shape[0] / 2) * distance / focal
        pose.pose.position.z = distance
        pose.pose.orientation.w = 1.0

        out_frame = cam_frame
        try:
            tf = self._tf_buffer.lookup_transform(base_frame, cam_frame, rclpy.time.Time())
            bx, by, bz = transform_xyz(
                matrix_from_tf(tf),
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            pose.header.frame_id = base_frame
            pose.pose.position.x = bx
            pose.pose.position.y = by
            pose.pose.position.z = bz
            out_frame = base_frame
        except Exception as exc:
            self.get_logger().warn(f'TF {base_frame}<-{cam_frame} failed: {exc}')

        berry = DetectedBerry()
        berry.header = pose.header
        berry.pose = pose
        berry.confidence = conf
        berry.track_id = -1
        berry.depth_mode = ''
        berry.z_depth_m = -1.0
        berry.z_mono_m = -1.0

        arr = DetectedBerryArray()
        arr.header.stamp = stamp
        arr.header.frame_id = out_frame
        arr.berries = [berry]
        self._pub.publish(arr)
        self._publish_detection_viz(
            stamp, cam_frame, mask, viz_dets, mode,
            (int(cx), int(cy)), int(max(diameter_px * 0.5, 8)))

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
        label = f'global {mode}'
        if center is not None:
            label += f' @ {center}'
        else:
            label += ': no berry'
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
