"""Move Gazebo contact marker to match /perception/contact_pose_gz (sim world coords)."""

from __future__ import annotations

import subprocess

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class ContactGzVisualizer(Node):
    def __init__(self) -> None:
        super().__init__('contact_gz_visualizer')
        self.declare_parameter('marker_model_name', 'contact_marker')
        self.declare_parameter('world_name', 'blueberry_picking')
        self.declare_parameter('contact_topic', '/perception/contact_pose_gz')

        self._marker = self.get_parameter('marker_model_name').value
        self._world_name = self.get_parameter('world_name').value
        self._gz_service = f'/world/{self._world_name}/set_pose'

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        topic = self.get_parameter('contact_topic').value
        self.create_subscription(PoseStamped, topic, self._on_contact, qos)

        self._update_count = 0
        self._last_pose: tuple[float, float, float] | None = None
        self.get_logger().info(
            f'contact_gz_visualizer: marker={self._marker} topic={topic} '
            f'(direct Gazebo world coords from perception)')

    def _set_gz_pose(self, x: float, y: float, z: float) -> bool:
        req = (
            f'name: "{self._marker}", '
            f'position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{w: 1.0}}'
        )
        try:
            proc = subprocess.run(
                [
                    'gz', 'service', '-s', self._gz_service,
                    '--reqtype', 'gz.msgs.Pose',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '2000',
                    '--req', req,
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            ok = 'data: true' in proc.stdout
            if not ok and proc.stderr:
                self.get_logger().warn(
                    f'set_pose failed: {proc.stderr.strip()}', throttle_duration_sec=5.0)
            return ok
        except Exception as exc:
            self.get_logger().warn(
                f'gz service error: {exc}', throttle_duration_sec=5.0)
            return False

    def _on_contact(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        pose_key = (round(p.x, 4), round(p.y, 4), round(p.z, 4))
        if pose_key == self._last_pose:
            return
        if self._set_gz_pose(p.x, p.y, p.z):
            self._last_pose = pose_key
            self._update_count += 1
            if self._update_count == 1 or self._update_count % 10 == 0:
                self.get_logger().info(
                    f'contact marker gz=({p.x:.3f},{p.y:.3f},{p.z:.3f})')


def main() -> None:
    rclpy.init()
    node = ContactGzVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
