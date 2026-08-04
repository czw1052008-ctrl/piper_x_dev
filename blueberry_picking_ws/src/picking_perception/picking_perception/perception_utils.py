"""Shared helpers for perception nodes."""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, TransformStamped, Vector3
from tf2_geometry_msgs import do_transform_pose

# static_transform_publisher world -> base_link (must match gz_sim.launch.py)
WORLD_TO_BASE_Z = 0.1
# Gazebo branch cylinder length in blueberry_plant SDF
BRANCH_CYLINDER_HALF_LENGTH_M = 0.09
# Gazebo link names: berry_{branch}_{index}
_BERRY_LINK_RE = re.compile(r'berry_(\d+)_(\d+)')
_BRANCH_LINK_RE = re.compile(r'branch_(\d+)')


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def matrix_from_tf(transform: TransformStamped) -> np.ndarray:
    """4x4 SE3 from geometry_msgs TransformStamped (translation + quat xyzw)."""
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def transform_xyz(T: np.ndarray, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Apply 4x4 SE3 to a point."""
    p = T @ np.array([x, y, z, 1.0], dtype=np.float64)
    return float(p[0]), float(p[1]), float(p[2])


def pose_from_transform(transform: TransformStamped) -> PoseStamped:
    ps = PoseStamped()
    ps.header = transform.header
    ps.pose.position.x = transform.transform.translation.x
    ps.pose.position.y = transform.transform.translation.y
    ps.pose.position.z = transform.transform.translation.z
    ps.pose.orientation = transform.transform.rotation
    return ps


def _pose_to_matrix(pose: Pose) -> np.ndarray:
    q = pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return mat


def quat_rotate_vector(q: Quaternion, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    x, y, z, w = q.x, q.y, q.z, q.w
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return rx, ry, rz


def _is_plant_model_root(transform: TransformStamped) -> bool:
    child = transform.child_frame_id or ''
    if '::' in child:
        child = child.split('::')[-1]
    return child == 'blueberry_plant'


def compose_pose_static_to_world(
    transforms: Iterable[TransformStamped],
) -> List[TransformStamped]:
    """PosePublisher gives link poses in model frame; compose to world."""
    raw = list(transforms)
    model_tf = next((t for t in raw if _is_plant_model_root(t)), None)
    if model_tf is None:
        return []

    world_frame = model_tf.header.frame_id or 'world'
    stamp = model_tf.header.stamp
    model_frame = model_tf.child_frame_id

    world_transforms: List[TransformStamped] = []
    for transform in raw:
        if _is_plant_model_root(transform):
            continue
        if transform.header.frame_id not in (model_frame, 'blueberry_plant'):
            continue
        link_pose = Pose()
        link_pose.position = transform.transform.translation
        link_pose.orientation = transform.transform.rotation
        world_pose = do_transform_pose(link_pose, model_tf)
        out = TransformStamped()
        out.header.stamp = stamp
        out.header.frame_id = world_frame
        out.child_frame_id = _link_name(transform)
        out.transform.translation.x = world_pose.position.x
        out.transform.translation.y = world_pose.position.y
        out.transform.translation.z = world_pose.position.z
        out.transform.rotation = world_pose.orientation
        world_transforms.append(out)
    return world_transforms


def _link_name(transform: TransformStamped) -> str:
    name = transform.child_frame_id or transform.header.frame_id
    if '::' in name:
        name = name.split('::')[-1]
    if '/' in name:
        name = name.split('/')[-1]
    return name.lower()


def _find_branch_pose(
    transforms: Iterable[TransformStamped], branch_id: int,
) -> Optional[Pose]:
    target = f'branch_{branch_id}'
    for transform in transforms:
        if _link_name(transform) == target:
            return pose_from_transform(transform).pose
    return None


def branch_axis_world(branch_pose: Pose) -> np.ndarray:
    """SDF cylinder length is along link +Z."""
    ax, ay, az = quat_rotate_vector(branch_pose.orientation, 0.0, 0.0, 1.0)
    axis = np.array([ax, ay, az], dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def _branch_ids_with_berries(transforms: Iterable[TransformStamped]) -> List[int]:
    ids: set[int] = set()
    for transform in transforms:
        m = _BERRY_LINK_RE.match(_link_name(transform))
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def vibration_pick_branch_id(
    transforms: Iterable[TransformStamped],
) -> int:
    """Top arm-facing branch: horizontal side branch with fruit toward the robot."""
    best_id = 1
    best_score = -float('inf')
    for bid in _branch_ids_with_berries(transforms):
        branch_pose = _find_branch_pose(transforms, bid)
        if branch_pose is None:
            continue
        berries = berries_from_pose_info(transforms, None, 'world', bid)
        if not berries:
            continue
        axis = branch_axis_world(branch_pose)
        horiz = 1.0 - min(1.0, abs(float(axis[2])))
        cz = max(b.pose.position.z for b in berries)
        cx = sum(b.pose.position.x for b in berries) / len(berries)
        toward_arm = -cx
        score = 2.0 * horiz + 0.4 * cz + 0.2 * toward_arm
        if score > best_score:
            best_score = score
            best_id = bid
    return best_id


def highest_branch_with_berries(
    transforms: Iterable[TransformStamped],
) -> int:
    """Branch whose fruit cluster has the highest world z."""
    best_id = 1
    best_z = -float('inf')
    for bid in _branch_ids_with_berries(transforms):
        berries = berries_from_pose_info(transforms, None, 'world', bid)
        if not berries:
            continue
        cz = max(b.pose.position.z for b in berries)
        if cz > best_z:
            best_z = cz
            best_id = bid
    return best_id


def _pick_branch_id(transforms: Iterable[TransformStamped]) -> int:
    return vibration_pick_branch_id(transforms)


def _branch_axis_toward_fruit(
    center: np.ndarray,
    axis: np.ndarray,
    berry_pts: np.ndarray,
) -> np.ndarray:
    if berry_pts.size == 0:
        return axis
    centroid = berry_pts.mean(axis=0)
    if float(np.dot(axis, centroid - center)) < 0.0:
        axis = -axis
    return axis


def _fruit_branch_junction_on_axis(
    center: np.ndarray,
    axis: np.ndarray,
    berry_pts: np.ndarray,
) -> np.ndarray:
    """Branch cylinder +Z tip where the fruit cluster meets the stem."""
    axis = _branch_axis_toward_fruit(center, axis, berry_pts)
    return center + BRANCH_CYLINDER_HALF_LENGTH_M * axis


def branch_cluster_contact_world(
    transforms: Iterable[TransformStamped],
    branch_id: int,
    clamp_back_from_pedicel_m: float = 0.03,
) -> Optional[Tuple[float, float, float]]:
    """Slot target: fruit–branch junction on axis, offset toward trunk (not branch root)."""
    branch_pose = _find_branch_pose(transforms, branch_id)
    if branch_pose is None:
        return None
    axis = branch_axis_world(branch_pose)
    center = np.array([
        branch_pose.position.x,
        branch_pose.position.y,
        branch_pose.position.z,
    ])

    berries = berries_from_pose_info(transforms, None, 'world', branch_id)
    if berries:
        berry_pts = np.array([
            [b.pose.position.x, b.pose.position.y, b.pose.position.z]
            for b in berries
        ])
    else:
        berry_pts = np.empty((0, 3))

    axis = _branch_axis_toward_fruit(center, axis, berry_pts)
    if berry_pts.size:
        # Trunk-side edge of fruit cluster on branch axis (pedicel / 果–枝连接处).
        t_junction = min(float(np.dot(p - center, axis)) for p in berry_pts)
    else:
        t_junction = BRANCH_CYLINDER_HALF_LENGTH_M
    t_contact = t_junction - clamp_back_from_pedicel_m
    t_contact = max(
        -BRANCH_CYLINDER_HALF_LENGTH_M + 0.015,
        min(BRANCH_CYLINDER_HALF_LENGTH_M - 0.01, t_contact),
    )
    contact = center + t_contact * axis
    return float(contact[0]), float(contact[1]), float(contact[2])


def stem_direction_from_branch(
    transforms: Iterable[TransformStamped],
    branch_id: int,
    cluster_hint: Tuple[float, float, float],
) -> Optional[Vector3]:
    """Branch cylinder axis; flip so it points from contact toward arm base."""
    branch_pose = _find_branch_pose(transforms, branch_id)
    if branch_pose is None:
        return None
    axis = branch_axis_world(branch_pose)
    center = np.array([
        branch_pose.position.x,
        branch_pose.position.y,
        branch_pose.position.z,
    ])
    hint = np.array(cluster_hint)
    to_cluster = hint - center
    if float(np.dot(axis, to_cluster)) < 0.0:
        axis = -axis
    # Rod axis points from tip toward motor (toward base / away from cluster tip).
    to_base = -hint
    if float(np.dot(axis, to_base)) < 0.0:
        axis = -axis
    stem = Vector3()
    stem.x = float(axis[0])
    stem.y = float(axis[1])
    stem.z = float(axis[2])
    return stem


def berries_from_pose_info(
    transforms: Iterable[TransformStamped],
    stamp,
    target_frame: str = 'world',
    branch_id: Optional[int] = None,
) -> List[PoseStamped]:
    """Parse berry link poses for one branch (fruit centers only)."""
    bid = branch_id if branch_id is not None else _pick_branch_id(transforms)
    prefix = f'berry_{bid}_'
    berries: List[PoseStamped] = []
    for transform in transforms:
        link = _link_name(transform)
        if not link.startswith(prefix):
            continue
        ps = pose_from_transform(transform)
        ps.header.stamp = stamp
        ps.header.frame_id = target_frame
        berries.append(ps)
    berries.sort(key=lambda b: b.pose.position.x, reverse=True)
    return berries


def world_pose_to_base_link(pose: PoseStamped, stamp=None) -> PoseStamped:
    out = PoseStamped()
    out.header.frame_id = 'base_link'
    out.header.stamp = stamp if stamp is not None else pose.header.stamp
    out.pose.position.x = pose.pose.position.x
    out.pose.position.y = pose.pose.position.y
    out.pose.position.z = pose.pose.position.z - WORLD_TO_BASE_Z
    out.pose.orientation = pose.pose.orientation
    return out


def transform_pose_to_frame(
    pose: PoseStamped,
    target_frame: str,
    tf_buffer,
    logger=None,
) -> PoseStamped | None:
    if target_frame == 'base_link' and pose.header.frame_id in ('', 'world', 'map'):
        return world_pose_to_base_link(pose)
    if tf_buffer is None:
        if logger is not None:
            logger.warning(
                f'No TF buffer for {pose.header.frame_id} -> {target_frame}')
        return None
    try:
        import rclpy

        tf = tf_buffer.lookup_transform(
            target_frame,
            pose.header.frame_id,
            rclpy.time.Time(),
        )
        out = PoseStamped()
        out.header.frame_id = target_frame
        out.header.stamp = pose.header.stamp
        out.pose = do_transform_pose(pose.pose, tf)
        return out
    except Exception as exc:
        if logger is not None:
            logger.warning(
                f'TF {pose.header.frame_id} -> {target_frame} failed: {exc}')
        return None


def pose_with_noise(pose: Pose, noise_m: float) -> Pose:
    import random

    out = Pose()
    out.orientation = pose.orientation
    out.position.x = pose.position.x + random.gauss(0, noise_m)
    out.position.y = pose.position.y + random.gauss(0, noise_m)
    out.position.z = pose.position.z + random.gauss(0, noise_m)
    return out


# --- Static / BT-only reference (branch_1 tip cluster, base_link) ---
BRANCH1_STEM = (-0.945, 0.167, 0.278)
# Slot clamp: fruit–branch junction + 3 cm toward trunk on branch_1 (static BT reference).
BRANCH1_CONTACT_BASE = (0.259, 0.033, 0.550)
BRANCH1_CLUSTER_BASE = (
    (0.305, 0.020, 0.545),
    (0.298, 0.032, 0.540),
    (0.292, 0.015, 0.550),
    (0.287, 0.028, 0.542),
)


def branch1_static_cluster(
    stamp,
    target_frame: str = 'base_link',
) -> Tuple[List[PoseStamped], PoseStamped, Vector3]:
    """Tip cluster geometry (not collinear along branch) for BT-only tests."""
    berries: List[PoseStamped] = []
    header_frame = target_frame
    for x, y, z in BRANCH1_CLUSTER_BASE:
        ps = PoseStamped()
        ps.header.frame_id = header_frame
        ps.header.stamp = stamp
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z if target_frame == 'base_link' else z + WORLD_TO_BASE_Z
        ps.pose.orientation.w = 1.0
        berries.append(ps)

    cx, cy, cz = BRANCH1_CONTACT_BASE
    contact = PoseStamped()
    contact.header.frame_id = header_frame
    contact.header.stamp = stamp
    contact.pose.position.x = cx
    contact.pose.position.y = cy
    contact.pose.position.z = cz if target_frame == 'base_link' else cz + WORLD_TO_BASE_Z
    contact.pose.orientation.w = 1.0

    ax, ay, az = BRANCH1_STEM
    al = math.sqrt(ax * ax + ay * ay + az * az)
    stem = Vector3(x=ax / al, y=ay / al, z=az / al)
    return berries, contact, stem
