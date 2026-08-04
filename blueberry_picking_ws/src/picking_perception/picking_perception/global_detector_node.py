"""Global coarse detection — fixed camera, topic stream (no business service)."""

from __future__ import annotations

import math

from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from picking_msgs.msg import DetectedBerry, DetectedBerryArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from picking_perception.segmentation import segment_blueberry_hsv


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
        self.declare_parameter('publish_hz', 3.0)
        self.declare_parameter('image_topic', '/camera_fixed/image_raw')
        self.declare_parameter('camera_frame', 'camera_fixed_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_mask_px', 100)
        self.declare_parameter('viz_topic', '/perception/global/detection_viz')
        self.declare_parameter('publish_viz', True)

        self._rgb = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        image_topic = self.get_parameter('image_topic').value
        # Match v4l2_camera / rqt (BEST_EFFORT sensor QoS).
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
            f'(image={image_topic})')

    def _on_rgb(self, msg: Image) -> None:
        try:
            self._rgb = _image_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc))

    def _tick(self) -> None:
        if self._rgb is None:
            return

        mask = segment_blueberry_hsv(self._rgb)
        min_px = int(self.get_parameter('min_mask_px').value)
        stamp = self.get_clock().now().to_msg()
        cam_frame = self.get_parameter('camera_frame').value
        base_frame = self.get_parameter('base_frame').value

        if mask.sum() < min_px:
            out = DetectedBerryArray()
            out.header.stamp = stamp
            out.header.frame_id = base_frame
            self._pub.publish(out)
            self._publish_detection_viz(stamp, cam_frame, mask, None, None)
            return

        ys, xs = np.where(mask > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        area = max(float(mask.sum()), 1.0)
        diameter_px = 2.0 * math.sqrt(area / math.pi)
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
            t = tf.transform.translation
            pose.header.frame_id = base_frame
            pose.pose.position.x += t.x
            pose.pose.position.y += t.y
            pose.pose.position.z += t.z
            out_frame = base_frame
        except Exception as exc:
            self.get_logger().warn(f'TF {base_frame}<-{cam_frame} failed: {exc}')

        berry = DetectedBerry()
        berry.header = pose.header
        berry.pose = pose
        berry.confidence = 0.5

        arr = DetectedBerryArray()
        arr.header.stamp = stamp
        arr.header.frame_id = out_frame
        arr.berries = [berry]
        self._pub.publish(arr)
        self._publish_detection_viz(
            stamp, cam_frame, mask, (int(cx), int(cy)), int(max(diameter_px * 0.5, 8)))

    def _publish_detection_viz(
        self, stamp, frame_id: str, mask: np.ndarray,
        center: Optional[Tuple[int, int]], radius: Optional[int],
    ) -> None:
        if self._viz_pub is None or self._rgb is None:
            return
        import cv2  # type: ignore

        vis = self._rgb.copy()
        # HSV mask tint (cyan)
        tint = vis.copy()
        tint[mask > 0] = (0, 220, 220)
        vis = cv2.addWeighted(vis, 0.65, tint, 0.35, 0)
        if center is not None and radius is not None:
            cv2.circle(vis, center, radius, (0, 255, 0), 2)
            cv2.circle(vis, center, 3, (0, 255, 0), -1)
            cv2.putText(
                vis, f'global HSV @ {center}', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(
                vis, 'global: no berry', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2, cv2.LINE_AA)
        self._viz_pub.publish(_rgb_to_imgmsg(vis, stamp, frame_id))


def main() -> None:
    rclpy.init()
    node = GlobalDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
