"""Gazebo ground-truth perception: stream topics from pose_static (no service pull)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Vector3, Vector3Stamped
from picking_msgs.msg import DetectedBerry, DetectedBerryArray, PerceptionStatus
from picking_msgs.srv import TriggerFineDetection, TriggerGlobalDetection
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from picking_perception.perception_utils import (
    berries_from_pose_info,
    branch_cluster_contact_world,
    compose_pose_static_to_world,
    pose_with_noise,
    stem_direction_from_branch,
    transform_pose_to_frame,
    highest_branch_with_berries,
    yaw_to_quat,
)

PIPELINE = 'pick_pipeline'


class FakePerceptionNode(Node):
    """GT berries + branch contact + stem; streams /perception/* for teleop / IL."""

    def __init__(self) -> None:
        super().__init__('fake_perception_node')
        self.declare_parameter('suction_noise_m', 0.003)
        self.declare_parameter('vibration_noise_m', 0.015)
        self.declare_parameter('end_effector_mode', 'suction')
        self.declare_parameter('provide_fine_detection', True)
        self.declare_parameter('stream_topics', True)
        self.declare_parameter('stream_rate_hz', 10.0)
        self.declare_parameter(
            'gz_pose_topic', '/model/blueberry_plant/pose_static')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('pick_branch_id', 1)
        self.declare_parameter('pick_highest_branch', True)
        self.declare_parameter('branch_clamp_back_m', 0.03)
        self.declare_parameter('eef_link', 'link6')

        self._gz_transforms: list = []
        self._gz_world_frame = 'world'
        self._pose_msg_count = 0
        self._last_status = PerceptionStatus()

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub_cluster = self.create_publisher(PoseStamped, '/perception/cluster_pose', qos)
        self._pub_berries = self.create_publisher(
            DetectedBerryArray, '/perception/berries', qos)
        self._pub_contact = self.create_publisher(
            PoseStamped, '/perception/contact_pose', qos)
        self._pub_contact_gz = self.create_publisher(
            PoseStamped, '/perception/contact_pose_gz', qos)
        self._pub_stem = self.create_publisher(
            Vector3Stamped, '/perception/stem_direction', qos)
        self._pub_eef = self.create_publisher(PoseStamped, '/perception/eef_pose', qos)
        self._pub_status = self.create_publisher(PerceptionStatus, '/perception/status', qos)

        gz_pose_topic = self.get_parameter('gz_pose_topic').value
        self.create_subscription(TFMessage, gz_pose_topic, self._on_gz_poses, 10)

        if self.get_parameter('provide_fine_detection').value:
            self.create_service(
                TriggerFineDetection, 'trigger_fine_detection', self._on_fine)
        self.create_service(
            TriggerGlobalDetection, 'trigger_global_detection', self._on_global)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        if self.get_parameter('stream_topics').value:
            rate = max(1.0, float(self.get_parameter('stream_rate_hz').value))
            self.create_timer(1.0 / rate, self._publish_stream)

        self.get_logger().info(
            f'[{PIPELINE}:perception] stream topics on /perception/* '
            f'(GT from {gz_pose_topic}, contact+stem from branch geometry)')

    def _on_gz_poses(self, msg: TFMessage) -> None:
        self._gz_transforms = compose_pose_static_to_world(msg.transforms)
        if self._gz_transforms:
            self._gz_world_frame = self._gz_transforms[0].header.frame_id or 'world'
        self._pose_msg_count += 1
        if self._pose_msg_count == 1 or self._pose_msg_count % 50 == 0:
            self.get_logger().info(
                f'[{PIPELINE}:perception] cached {len(self._gz_transforms)} plant links (world)')

    def _to_target_frame(self, pose: PoseStamped) -> Optional[PoseStamped]:
        target = self.get_parameter('target_frame').value
        src = pose.header.frame_id or self._gz_world_frame
        for frame in (src, self._gz_world_frame, 'world', 'blueberry_picking'):
            trial = PoseStamped()
            trial.header = pose.header
            trial.header.frame_id = frame
            trial.pose = pose.pose
            out = transform_pose_to_frame(
                trial, target, self._tf_buffer, self.get_logger())
            if out is not None:
                return out
        return None

    def _lookup_eef_pose(self, stamp) -> Optional[PoseStamped]:
        target = self.get_parameter('target_frame').value
        eef = self.get_parameter('eef_link').value
        try:
            tf = self._tf_buffer.lookup_transform(
                target, eef, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            return None
        out = PoseStamped()
        out.header.stamp = stamp
        out.header.frame_id = target
        out.pose.position.x = tf.transform.translation.x
        out.pose.position.y = tf.transform.translation.y
        out.pose.position.z = tf.transform.translation.z
        out.pose.orientation = tf.transform.rotation
        return out

    def _resolve_branch_id(self) -> int:
        if self.get_parameter('pick_highest_branch').value:
            return highest_branch_with_berries(self._gz_transforms)
        return int(self.get_parameter('pick_branch_id').value)

    def _analyze_cluster(
        self, mode: str,
    ) -> Tuple[List[DetectedBerry], Optional[PoseStamped], Optional[PoseStamped],
               Optional[Vector3], bool, bool]:
        branch_id = self._resolve_branch_id()
        if not self._gz_transforms:
            return [], None, None, None, False, False

        stamp = self.get_clock().now().to_msg()
        world_berries = berries_from_pose_info(
            self._gz_transforms, stamp, 'world', branch_id)
        clamp_m = float(self.get_parameter('branch_clamp_back_m').value)
        contact_w = branch_cluster_contact_world(
            self._gz_transforms, branch_id, clamp_back_from_pedicel_m=clamp_m)

        if not world_berries or contact_w is None:
            return [], None, None, None, False, False

        cx, cy, cz = contact_w
        stem_w = stem_direction_from_branch(self._gz_transforms, branch_id, contact_w)
        if stem_w is None:
            return [], None, None, None, False, False

        noise = self.get_parameter(
            'suction_noise_m' if mode == 'suction' else 'vibration_noise_m').value
        berries: List[DetectedBerry] = []
        for pose_w in world_berries:
            transformed = self._to_target_frame(pose_w)
            if transformed is None:
                continue
            transformed.header.stamp = stamp
            transformed.pose = pose_with_noise(transformed.pose, noise)
            b = DetectedBerry()
            b.header.stamp = stamp
            b.header.frame_id = transformed.header.frame_id
            b.pose = PoseStamped()
            b.pose.header.stamp = stamp
            b.pose.header.frame_id = transformed.header.frame_id
            b.pose.pose = transformed.pose
            b.confidence = 0.95
            berries.append(b)

        contact_gz = PoseStamped()
        contact_gz.header.frame_id = self._gz_world_frame
        contact_gz.header.stamp = stamp
        contact_gz.pose.position.x = cx
        contact_gz.pose.position.y = cy
        contact_gz.pose.position.z = cz
        contact_gz.pose.orientation.w = 1.0

        contact_ps = PoseStamped()
        contact_ps.header.stamp = stamp
        contact_ps.header.frame_id = 'world'
        contact_ps.pose = contact_gz.pose
        contact_bl = self._to_target_frame(contact_ps)
        if contact_bl is not None:
            contact_bl.header.stamp = stamp
        if contact_bl is None:
            return berries, None, contact_gz, None, len(berries) >= 2, False

        if not hasattr(self, '_branch_log_count'):
            self._branch_log_count = 0
        self._branch_log_count += 1
        if self._branch_log_count == 1 or self._branch_log_count % 50 == 0:
            self.get_logger().info(
                f'[{PIPELINE}:perception] branch={branch_id} '
                f'contact_gz=({cx:.3f},{cy:.3f},{cz:.3f}) '
                f'contact_bl=({contact_bl.pose.position.x:.3f},'
                f'{contact_bl.pose.position.y:.3f},'
                f'{contact_bl.pose.position.z:.3f})')

        return berries, contact_bl, contact_gz, stem_w, len(berries) >= 2, True

    def _publish_stream(self) -> None:
        mode = self.get_parameter('end_effector_mode').value
        stamp = self.get_clock().now().to_msg()
        berries, contact, contact_gz, stem, berries_ok, contact_ok = self._analyze_cluster(mode)

        status = PerceptionStatus()
        status.header.stamp = stamp
        status.header.frame_id = self.get_parameter('target_frame').value
        status.berry_count = len(berries)
        status.berries_valid = berries_ok
        status.contact_valid = berries_ok and contact_ok and contact is not None
        status.stem_valid = berries_ok and stem is not None
        status.cluster_valid = status.contact_valid

        eef = self._lookup_eef_pose(stamp)
        status.eef_valid = eef is not None

        if contact_gz is not None:
            self._pub_contact_gz.publish(contact_gz)

        if status.contact_valid and contact is not None:
            cluster = PoseStamped()
            cluster.header.stamp = stamp
            cluster.header.frame_id = contact.header.frame_id
            cluster.pose = contact.pose
            cluster.pose.orientation = yaw_to_quat(0.3)
            self._pub_cluster.publish(cluster)
            contact.header.stamp = stamp
            self._pub_contact.publish(contact)

        if status.stem_valid and stem is not None:
            stem_msg = Vector3Stamped()
            stem_msg.header.stamp = stamp
            stem_msg.header.frame_id = status.header.frame_id
            stem_msg.vector = stem
            self._pub_stem.publish(stem_msg)

        if berries:
            arr = DetectedBerryArray()
            arr.header.stamp = stamp
            arr.header.frame_id = status.header.frame_id
            arr.berries = berries if mode != 'suction' else [
                max(berries, key=lambda b: b.confidence)]
            self._pub_berries.publish(arr)

        if eef is not None:
            self._pub_eef.publish(eef)

        if status.contact_valid:
            p = contact.pose.position
            status.message = (
                f'OK berries={len(berries)} contact=({p.x:.3f},{p.y:.3f},{p.z:.3f})')
        else:
            status.message = (
                f'invalid links={len(self._gz_transforms)} berries={len(berries)}')

        self._pub_status.publish(status)

        if not hasattr(self, '_stream_log_count'):
            self._stream_log_count = 0
        self._stream_log_count += 1
        if self._stream_log_count == 1 or self._stream_log_count % 50 == 0:
            if status.contact_valid and contact is not None and stem is not None:
                p = contact.pose.position
                self.get_logger().info(
                    f'[{PIPELINE}:perception_stream] berries={len(berries)} '
                    f'contact=({p.x:.3f},{p.y:.3f},{p.z:.3f}) '
                    f'stem=({stem.x:.3f},{stem.y:.3f},{stem.z:.3f}) '
                    f'eef_ok={status.eef_valid}')
            else:
                self.get_logger().warn(
                    f'[{PIPELINE}:perception_stream] {status.message}')

        self._last_status = status

    def _on_global(self, _req, res):
        berries, contact, _contact_gz, _stem, ok, contact_ok = self._analyze_cluster('vibration')
        if not ok or not contact_ok or contact is None:
            res.success = False
            res.message = (
                f'No cluster/contact ({len(self._gz_transforms)} gz links, '
                f'berries={len(berries)})')
            return res
        cluster = PoseStamped()
        cluster.header = contact.header
        cluster.pose = contact.pose
        cluster.pose.orientation = yaw_to_quat(0.3)
        res.cluster_pose = cluster
        res.success = True
        res.message = 'OK'
        return res

    def _on_fine(self, _req, res):
        mode = self.get_parameter('end_effector_mode').value
        berries, contact, contact_gz, stem, berries_ok, contact_ok = self._analyze_cluster(mode)
        res.detected_berries = berries
        res.stem_direction_valid = berries_ok and stem is not None
        res.contact_pose_valid = berries_ok and contact_ok and contact is not None
        res.success = (
            berries_ok and res.stem_direction_valid and res.contact_pose_valid)
        if res.success:
            res.stem_direction = stem
            res.contact_pose = contact
            if mode == 'suction':
                res.detected_berries = [max(berries, key=lambda b: b.confidence)]
            res.message = f'{len(res.detected_berries)} berries + contact + stem'
        else:
            res.message = (
                f'Fine detect failed berries={len(berries)} '
                f'stem={res.stem_direction_valid} contact={res.contact_pose_valid}')
        return res


def main() -> None:
    rclpy.init()
    node = FakePerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
