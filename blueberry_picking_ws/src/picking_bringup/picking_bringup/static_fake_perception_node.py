"""Static fake berries for BT testing without Gazebo."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from picking_msgs.msg import DetectedBerry
from picking_msgs.srv import TriggerFineDetection, TriggerGlobalDetection
from picking_perception.perception_utils import branch1_static_cluster
from rclpy.node import Node


class StaticFakePerceptionNode(Node):
  def __init__(self) -> None:
    super().__init__('static_fake_perception_node')
    self.declare_parameter('end_effector_mode', 'suction')
    self.create_service(TriggerGlobalDetection, 'trigger_global_detection', self._on_global)
    self.create_service(TriggerFineDetection, 'trigger_fine_detection', self._on_fine)
    self.get_logger().info(
      'Static fake perception (tip cluster + branch contact, no PCA)')

  def _cluster(self):
    stamp = self.get_clock().now().to_msg()
    fruit_poses, contact, stem = branch1_static_cluster(stamp, 'base_link')
    berries = []
    for ps in fruit_poses:
      b = DetectedBerry()
      b.header = ps.header
      b.pose = ps
      b.confidence = 0.9
      b.track_id = -1
      berries.append(b)
    return berries, contact, stem

  def _on_global(self, _req, res):
    berries, contact, _stem = self._cluster()
    c = PoseStamped()
    c.header = contact.header
    c.pose = contact.pose
    res.cluster_pose = c
    res.success = True
    res.message = 'static global'
    return res

  def _on_fine(self, _req, res):
    berries, contact, stem = self._cluster()
    mode = self.get_parameter('end_effector_mode').value
    if mode == 'suction':
      berries = [berries[0]]
    res.detected_berries = berries
    res.contact_pose = contact
    res.contact_pose_valid = True
    res.stem_direction = stem
    res.stem_direction_valid = True
    res.success = len(berries) >= (1 if mode == 'suction' else 2)
    res.message = f'{len(berries)} static berries + contact + stem'
    return res


def main():
  rclpy.init()
  node = StaticFakePerceptionNode()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
