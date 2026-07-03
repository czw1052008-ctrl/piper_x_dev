"""Publish robot_description for gz_ros2_control controller_manager."""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


class RobotDescriptionPublisher(Node):
    def __init__(self) -> None:
        super().__init__('robot_description_publisher')
        self.declare_parameter('robot_description', '')
        description = self.get_parameter('robot_description').get_parameter_value().string_value
        if not description.strip():
            raise RuntimeError('robot_description parameter is empty')

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(String, '/robot_description', qos)
        self._msg = String()
        self._msg.data = description
        self._pub.publish(self._msg)
        self.get_logger().info(
            f'Publishing /robot_description once ({len(description)} bytes, transient_local)',
        )


def main() -> None:
    rclpy.init()
    node = RobotDescriptionPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
