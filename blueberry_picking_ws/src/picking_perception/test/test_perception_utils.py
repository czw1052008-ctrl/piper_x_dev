"""Unit tests for perception_utils (branch GT geometry)."""

import math

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped

from picking_perception.perception_utils import (
    berries_from_pose_info,
    branch_cluster_contact_world,
    compose_pose_static_to_world,
    highest_branch_with_berries,
    stem_direction_from_branch,
)


def _transform(child: str, x: float, y: float, z: float, q: Quaternion | None = None) -> TransformStamped:
    t = TransformStamped()
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z
    if q is None:
        t.transform.rotation.w = 1.0
    else:
        t.transform.rotation = q
    return t


def test_berries_from_branch1_links():
    transforms = [
        _transform('branch_1', 0.36, 0.02, 0.55),
        _transform('berry_1_1', 0.305, 0.020, 0.645),
        _transform('berry_1_2', 0.298, 0.032, 0.640),
        _transform('berry_1_3', 0.292, 0.015, 0.650),
        _transform('berry_1_4', 0.287, 0.028, 0.642),
    ]
    berries = berries_from_pose_info(transforms, None, 'world', branch_id=1)
    assert len(berries) == 4


def test_berries_from_gz_slash_link_names():
    transforms = [
        _transform('blueberry_plant/branch_1', 0.36, 0.02, 0.55),
        _transform('blueberry_plant/berry_1_1', 0.305, 0.020, 0.645),
        _transform('blueberry_plant/berry_1_2', 0.298, 0.032, 0.640),
    ]
    berries = berries_from_pose_info(transforms, None, 'world', branch_id=1)
    assert len(berries) == 2
    contact = branch_cluster_contact_world(transforms, 1)
    assert contact is not None


def test_stem_direction_from_branch_geometry():
    q = Quaternion()
    q.w = 1.0
    transforms = [
        _transform('branch_1', 0.30, 0.02, 0.55, q),
        _transform('berry_1_1', 0.305, 0.020, 0.645),
    ]
    stem = stem_direction_from_branch(transforms, 1, (0.305, 0.020, 0.645))
    assert stem is not None
    length = math.sqrt(stem.x ** 2 + stem.y ** 2 + stem.z ** 2)
    assert abs(length - 1.0) < 1e-6


def test_branch_slot_contact_3cm_from_pedicel():
    q = Quaternion()
    q.w = 1.0
    branch_center = (0.30, 0.02, 0.55)
    berries_world = [
        (0.320, 0.015, 0.640),
        (0.305, -0.010, 0.638),
        (0.315, 0.018, 0.642),
        (0.308, 0.008, 0.637),
    ]
    transforms = [_transform('branch_1', *branch_center, q)]
    for i, pos in enumerate(berries_world, start=1):
        transforms.append(_transform(f'berry_1_{i}', *pos))

    contact = branch_cluster_contact_world(transforms, 1, clamp_back_from_pedicel_m=0.03)
    assert contact is not None
    t_min = min(p[2] - branch_center[2] for p in berries_world)
    assert abs(contact[2] - (branch_center[2] + t_min - 0.03)) < 0.005


def test_highest_branch_with_berries_picks_top_cluster():
    q = Quaternion()
    q.w = 1.0
    q3 = Quaternion()
    q3.w = 0.7071
    q3.y = 0.7071
    transforms = [
        _transform('branch_3', 0.32, 0.00, 0.58, q3),
        _transform('berry_3_1', 0.250, 0.010, 0.580),
        _transform('berry_3_2', 0.245, -0.005, 0.582),
        _transform('branch_2', 0.30, 0.02, 0.72, q),
        _transform('berry_2_1', 0.305, 0.020, 0.780),
        _transform('berry_2_2', 0.298, 0.032, 0.775),
    ]
    assert highest_branch_with_berries(transforms) == 2


def test_vibration_pick_branch_prefers_horizontal_arm_facing():
    q = Quaternion()
    q.w = 1.0
    transforms = [
        _transform('branch_2', 0.30, 0.02, 0.62, q),
        _transform('berry_2_1', 0.305, 0.020, 0.700),
    ]
    # branch_3: horizontal (+X), toward arm (lower x)
    q3 = Quaternion()
    q3.w = 0.7071
    q3.y = 0.7071
    transforms.extend([
        _transform('branch_3', 0.32, 0.00, 0.58, q3),
        _transform('berry_3_1', 0.250, 0.010, 0.580),
        _transform('berry_3_2', 0.245, -0.005, 0.582),
    ])
    from picking_perception.perception_utils import vibration_pick_branch_id
    assert vibration_pick_branch_id(transforms) == 3


def test_tip_cluster_not_collinear_along_branch():
    """Berries at branch tip form a cluster, not a tanghulu string along the axis."""
    import numpy as np

    q = Quaternion()
    q.w = 1.0
    branch_center = (0.30, 0.02, 0.55)
    # Tip cluster: spread in XY near branch +Z tip (world z ~ 0.64)
    berries_world = [
        (0.320, 0.015, 0.640),
        (0.305, -0.010, 0.638),
        (0.315, 0.018, 0.642),
        (0.308, 0.008, 0.637),
    ]
    transforms = [_transform('branch_1', *branch_center, q)]
    for i, pos in enumerate(berries_world, start=1):
        transforms.append(_transform(f'berry_1_{i}', *pos))

    contact = branch_cluster_contact_world(transforms, 1)
    assert contact is not None

    axis = np.array([0.0, 0.0, 1.0])
    branch_base = np.array([branch_center[0], branch_center[1], branch_center[2] - 0.09])
    projections = [
        float(np.dot(np.array(p) - branch_base, axis)) for p in berries_world
    ]
    spread_along_axis = max(projections) - min(projections)
    assert spread_along_axis < 0.04

    # Contact sits on branch axis between center and berry tip.
    tip = np.array([branch_center[0], branch_center[1], branch_center[2] + 0.09])
    contact_pt = np.array(contact)
    assert float(np.linalg.norm(contact_pt - tip)) > 0.025
    assert contact_pt[2] < min(p[2] for p in berries_world)


def test_compose_pose_static_to_world():
    model_tf = TransformStamped()
    model_tf.header.frame_id = 'blueberry_picking'
    model_tf.child_frame_id = 'blueberry_plant'
    model_tf.transform.translation.x = 0.45
    model_tf.transform.translation.z = 0.41
    model_tf.transform.rotation.w = 1.0

    berry_tf = TransformStamped()
    berry_tf.header.frame_id = 'blueberry_plant'
    berry_tf.child_frame_id = 'blueberry_plant/berry_1_1'
    berry_tf.transform.translation.x = 0.02
    berry_tf.transform.translation.y = 0.015
    berry_tf.transform.translation.z = 0.09
    berry_tf.transform.rotation.w = 1.0

    world = compose_pose_static_to_world([model_tf, berry_tf])
    assert len(world) == 1
    assert world[0].child_frame_id == 'berry_1_1'
    assert abs(world[0].transform.translation.x - 0.47) < 1e-6
    assert abs(world[0].transform.translation.z - 0.50) < 1e-6


def test_matrix_from_tf_rotates_and_translates():
    """Slanted optical frame: camera +Z must map via full SE3, not translation-only."""
    from geometry_msgs.msg import TransformStamped
    from picking_perception.perception_utils import matrix_from_tf, transform_xyz

    tf = TransformStamped()
    tf.transform.translation.x = 0.60
    tf.transform.translation.y = 0.33
    tf.transform.translation.z = 0.48
    # -90 deg about Y: cam +Z -> world -X → (0,0,0.5) + t ≈ (0.10, 0.33, 0.48)
    tf.transform.rotation.x = 0.0
    tf.transform.rotation.y = -0.70710678
    tf.transform.rotation.z = 0.0
    tf.transform.rotation.w = 0.70710678
    T = matrix_from_tf(tf)
    bx, by, bz = transform_xyz(T, 0.0, 0.0, 0.5)
    assert abs(bx - 0.10) < 1e-5
    assert abs(by - 0.33) < 1e-5
    assert abs(bz - 0.48) < 1e-5
    # translation-only must NOT match
    assert abs(bx - 0.60) > 0.1
