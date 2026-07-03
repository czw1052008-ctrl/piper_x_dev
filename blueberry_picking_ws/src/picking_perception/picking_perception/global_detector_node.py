"""Global coarse detection using fixed camera (HSV + size prior)."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from picking_msgs.srv import TriggerGlobalDetection
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from picking_perception.segmentation import segment_blueberry_hsv


class GlobalDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('global_detector_node')
        self.declare_parameter('berry_diameter_m', 0.015)
        self.declare_parameter('camera_focal_px', 500.0)

        self._bridge = CvBridge()
        self._rgb = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Image, '/camera_fixed/image_raw', self._on_rgb, 1)
        self.create_service(
            TriggerGlobalDetection, 'trigger_global_detection', self._on_detect)

    def _on_rgb(self, msg: Image) -> None:
        self._rgb = self._bridge.imgmsg_to_cv2(msg, 'rgb8')

    def _on_detect(self, _req, res):
        if self._rgb is None:
            res.success = False
            res.message = 'No fixed camera image'
            return res

        mask = segment_blueberry_hsv(self._rgb)
        if mask.sum() < 100:
            res.success = False
            res.message = 'No blueberry colored region'
            return res

        ys, xs = np.where(mask > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        area = max(float(mask.sum()), 1.0)
        diameter_px = 2.0 * math.sqrt(area / math.pi)
        focal = self.get_parameter('camera_focal_px').value
        berry_d = self.get_parameter('berry_diameter_m').value
        distance = focal * berry_d / max(diameter_px, 1.0)

        pose = PoseStamped()
        pose.header.frame_id = 'camera_fixed_optical_frame'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = (cx - self._rgb.shape[1] / 2) * distance / focal
        pose.pose.position.y = (cy - self._rgb.shape[0] / 2) * distance / focal
        pose.pose.position.z = distance
        pose.pose.orientation.w = 1.0

        try:
            tf = self._tf_buffer.lookup_transform(
                'base_link', pose.header.frame_id, rclpy.time.Time())
            # Simplified: use translation only
            pose.header.frame_id = 'base_link'
            pose.pose.position.x += tf.transform.translation.x
            pose.pose.position.y += tf.transform.translation.y
            pose.pose.position.z += tf.transform.translation.z
        except Exception as exc:
            self.get_logger().warn('TF lookup failed: %s', exc)

        res.cluster_pose = pose
        res.success = True
        res.message = 'Global detection OK'
        return res


def main() -> None:
    rclpy.init()
    node = GlobalDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
