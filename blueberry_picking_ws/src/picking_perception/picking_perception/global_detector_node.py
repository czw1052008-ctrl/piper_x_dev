"""Global coarse detection — fixed camera, topic stream (no business service)."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from picking_msgs.msg import DetectedBerry, DetectedBerryArray
from rclpy.node import Node
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
    raise ValueError(f'unsupported color encoding: {msg.encoding}')


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

        self._rgb = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        image_topic = self.get_parameter('image_topic').value
        self.create_subscription(Image, image_topic, self._on_rgb, 1)
        self._pub = self.create_publisher(DetectedBerryArray, '/perception/global/berries', 10)

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
        if mask.sum() < min_px:
            out = DetectedBerryArray()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = self.get_parameter('base_frame').value
            self._pub.publish(out)
            return

        ys, xs = np.where(mask > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        area = max(float(mask.sum()), 1.0)
        diameter_px = 2.0 * math.sqrt(area / math.pi)
        focal = float(self.get_parameter('camera_focal_px').value)
        berry_d = float(self.get_parameter('berry_diameter_m').value)
        distance = focal * berry_d / max(diameter_px, 1.0)

        cam_frame = self.get_parameter('camera_frame').value
        base_frame = self.get_parameter('base_frame').value
        stamp = self.get_clock().now().to_msg()

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


def main() -> None:
    rclpy.init()
    node = GlobalDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
