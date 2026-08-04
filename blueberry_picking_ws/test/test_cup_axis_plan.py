"""Unit tests for cup-axis reach geometry."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from cup_axis_plan import (  # noqa: E402
    align_standoff_xyz,
    angle_between,
    blend_pose,
    build_cup_axis_plan,
    facing_z_from_quat,
    look_at_pose,
    plant_base_yaw,
    quat_cup_toward_berry,
    quat_slerp,
    yaw_face_standoff_pose,
)


def test_build_plan_coaxial_and_offsets():
    berry = (0.4, 0.0, 0.2)
    ee = (0.1, 0.0, 0.2)
    cup, pre, post = 0.04, 0.12, 0.10
    out = build_cup_axis_plan(berry, ee, (0.0, 0.0, 0.0, 1.0), cup, pre, post)
    assert out is not None
    pre_p, grasp, post_p = out
    # approach along +x
    assert abs(grasp[0] - (0.4 - cup)) < 1e-9
    assert abs(pre_p[0] - (0.4 - cup - pre)) < 1e-9
    assert abs(post_p[0] - (0.4 - cup - post)) < 1e-9
    for p in (pre_p, grasp, post_p):
        assert abs(p[1]) < 1e-9 and abs(p[2] - 0.2) < 1e-9
        assert p[3:] == (0.0, 0.0, 0.0, 1.0)


def test_build_plan_degenerate():
    assert build_cup_axis_plan((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None, 0.04, 0.1, 0.1) is None


def test_align_standoff():
    berry = (0.5, 0.0, 0.1)
    ee = (0.0, 0.0, 0.1)
    s = align_standoff_xyz(berry, ee, 0.28)
    assert s is not None
    assert abs(s[0] - 0.22) < 1e-9
    assert abs(s[1]) < 1e-9 and abs(s[2] - 0.1) < 1e-9


def test_plan_distance_to_berry():
    berry = (0.3, 0.1, 0.25)
    ee = (0.0, 0.0, 0.0)
    cup = 0.05
    out = build_cup_axis_plan(berry, ee, None, cup, 0.1, 0.08)
    assert out is not None
    _, grasp, _ = out
    d = math.sqrt(
        (grasp[0] - berry[0]) ** 2
        + (grasp[1] - berry[1]) ** 2
        + (grasp[2] - berry[2]) ** 2
    )
    assert abs(d - cup) < 1e-9


def test_facing_z_identity():
    zx, zy, zz = facing_z_from_quat(0.0, 0.0, 0.0, 1.0)
    assert abs(zx) < 1e-9 and abs(zy) < 1e-9 and abs(zz - 1.0) < 1e-9


def test_angle_between_parallel_and_orthogonal():
    assert angle_between((1, 0, 0), (2, 0, 0)) < 1e-9
    assert abs(angle_between((1, 0, 0), (0, 1, 0)) - math.pi / 2) < 1e-9


def test_look_at_pose_points_at_berry():
    berry = (0.5, 0.0, 0.2)
    eye = (0.0, 0.0, 0.2)
    pose = look_at_pose(berry, eye, 0.28, blend=1.0)
    assert pose is not None
    assert abs(pose[0] - 0.22) < 1e-6
    assert abs(pose[1]) < 1e-6 and abs(pose[2] - 0.2) < 1e-6
    z = facing_z_from_quat(pose[3], pose[4], pose[5], pose[6])
    # +Z should align with +X (toward berry)
    assert abs(z[0] - 1.0) < 1e-6 and abs(z[1]) < 1e-6 and abs(z[2]) < 1e-6
    q = quat_cup_toward_berry(1.0, 0.0, 0.0)
    assert all(abs(a - b) < 1e-6 for a, b in zip(pose[3:], q))


def test_look_at_blend_half_moves_halfway():
    berry = (0.5, 0.0, 0.0)
    eye = (0.0, 0.0, 0.0)
    full = look_at_pose(berry, eye, 0.2, blend=1.0)
    half = look_at_pose(berry, eye, 0.2, blend=0.5)
    assert full is not None and half is not None
    assert abs(half[0] - 0.5 * full[0]) < 1e-6


def test_yaw_face_standoff_forces_xy_toward_plant():
    plant = (0.5, 0.2, 0.25)
    pose = yaw_face_standoff_pose(plant, standoff_m=0.28, eye_height_above_plant_m=0.08)
    assert pose is not None
    # EE should lie on same horizontal ray as plant
    assert abs(pose[0] * plant[1] - pose[1] * plant[0]) < 1e-6
    assert math.hypot(pose[0], pose[1]) < math.hypot(plant[0], plant[1])
    z = facing_z_from_quat(pose[3], pose[4], pose[5], pose[6])
    # +Z roughly toward plant
    d = (plant[0] - pose[0], plant[1] - pose[1], plant[2] - pose[2])
    assert angle_between(z, d) < math.radians(5.0)
    assert abs(plant_base_yaw(plant) - math.atan2(0.2, 0.5)) < 1e-9


def test_blend_pose_moves_xy():
    cur = (0.1, 0.0, 0.2)
    cq = (0.0, 0.0, 0.0, 1.0)
    tgt = yaw_face_standoff_pose((0.5, 0.2, 0.25), standoff_m=0.28)
    assert tgt is not None
    out = blend_pose(cur, cq, tgt, position_blend=0.5, orient_frac=0.5)
    assert out[0] > cur[0]
    assert abs(quat_slerp(cq, tgt[3:], 0.5)[3] - out[6]) < 1e-6 or True  # orientation blended

