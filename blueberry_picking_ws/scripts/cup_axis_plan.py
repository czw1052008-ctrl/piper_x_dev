"""Cup-axis suction reach waypoints (pure geometry, no ROS)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

# (x, y, z, qx, qy, qz, qw)
PoseXYZQ = Tuple[float, float, float, float, float, float, float]


def quat_cup_toward_berry(ax: float, ay: float, az: float) -> Tuple[float, float, float, float]:
    """Orientation with tool +Z along approach (ax,ay,az)."""
    z = [ax, ay, az]
    n = math.sqrt(z[0] ** 2 + z[1] ** 2 + z[2] ** 2) or 1.0
    z = [z[0] / n, z[1] / n, z[2] / n]
    ref = [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0]
    x = [
        ref[1] * z[2] - ref[2] * z[1],
        ref[2] * z[0] - ref[0] * z[2],
        ref[0] * z[1] - ref[1] * z[0],
    ]
    xn = math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) or 1.0
    x = [x[0] / xn, x[1] / xn, x[2] / xn]
    y = [
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0],
    ]
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def build_cup_axis_plan(
    berry_xyz: Tuple[float, float, float],
    ee_xyz: Optional[Tuple[float, float, float]],
    ee_quat: Optional[Tuple[float, float, float, float]],
    cup: float,
    pre: float,
    post: float,
) -> Optional[Tuple[PoseXYZQ, PoseXYZQ, PoseXYZQ]]:
    """Return (pre_grasp, grasp, post_grasp) as xyz+quat, or None if degenerate."""
    bx, by, bz = berry_xyz
    if ee_xyz is not None:
        ax = bx - ee_xyz[0]
        ay = by - ee_xyz[1]
        az = bz - ee_xyz[2]
    else:
        ax, ay, az = bx, by, bz
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-6:
        return None
    ax, ay, az = ax / n, ay / n, az / n
    if ee_quat is None:
        qx, qy, qz, qw = quat_cup_toward_berry(ax, ay, az)
    else:
        qx, qy, qz, qw = ee_quat

    def _at(extra: float) -> PoseXYZQ:
        reach = cup + extra
        return (
            bx - ax * reach,
            by - ay * reach,
            bz - az * reach,
            qx, qy, qz, qw,
        )

    return _at(pre), _at(0.0), _at(post)


def align_standoff_xyz(
    berry_xyz: Tuple[float, float, float],
    ee_xyz: Optional[Tuple[float, float, float]],
    standoff_m: float,
) -> Optional[Tuple[float, float, float]]:
    """Point on berry←ee ray at standoff_m before the berry (for ALIGNING)."""
    bx, by, bz = berry_xyz
    if ee_xyz is not None:
        ax = bx - ee_xyz[0]
        ay = by - ee_xyz[1]
        az = bz - ee_xyz[2]
    else:
        ax, ay, az = bx, by, bz
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-6:
        return None
    ax, ay, az = ax / n, ay / n, az / n
    return bx - ax * standoff_m, by - ay * standoff_m, bz - az * standoff_m


def facing_z_from_quat(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Tool/camera +Z axis in the same frame as the quaternion (base_link)."""
    # R @ [0,0,1] for quaternion (x,y,z,w)
    x, y, z, w = qx, qy, qz, qw
    return (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def angle_between(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    """Angle in radians between two 3-vectors."""
    ax, ay, az = a
    bx, by, bz = b
    na = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    nb = math.sqrt(bx * bx + by * by + bz * bz) or 1.0
    dot = (ax * bx + ay * by + az * bz) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def quat_slerp(
    q0: Tuple[float, float, float, float],
    q1: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    """Spherical linear interpolation; t in [0,1]."""
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1
        dot = -dot
    if dot > 0.9995:
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)
        w = w0 + t * (w1 - w0)
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        s = math.sin(theta)
        w_a = math.sin((1.0 - t) * theta) / s
        w_b = math.sin(t * theta) / s
        x = w_a * x0 + w_b * x1
        y = w_a * y0 + w_b * y1
        z = w_a * z0 + w_b * z1
        w = w_a * w0 + w_b * w1
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return x / n, y / n, z / n, w / n


def look_at_pose(
    berry_xyz: Tuple[float, float, float],
    eye_xyz: Tuple[float, float, float],
    standoff_m: float,
    *,
    blend: float = 1.0,
) -> Optional[PoseXYZQ]:
    """Pose with +Z toward berry; position on ray at standoff (blend in [0,1] for small step).

    blend=1 → full look-at at standoff; blend<1 → interpolate current eye toward that pose
    (position along eye→target, orientation is full look-at — IK small step uses position blend).
    """
    bx, by, bz = berry_xyz
    ex, ey, ez = eye_xyz
    ax, ay, az = bx - ex, by - ey, bz - ez
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-6:
        return None
    ax, ay, az = ax / n, ay / n, az / n
    qx, qy, qz, qw = quat_cup_toward_berry(ax, ay, az)
    full = (
        bx - ax * standoff_m,
        by - ay * standoff_m,
        bz - az * standoff_m,
    )
    b = max(0.0, min(1.0, float(blend)))
    px = ex + (full[0] - ex) * b
    py = ey + (full[1] - ey) * b
    pz = ez + (full[2] - ez) * b
    return (px, py, pz, qx, qy, qz, qw)


def plant_base_yaw(plant_xyz: Tuple[float, float, float]) -> float:
    """Horizontal yaw from base origin to plant (rad, base_link)."""
    return math.atan2(plant_xyz[1], plant_xyz[0])


def yaw_face_standoff_pose(
    plant_xyz: Tuple[float, float, float],
    *,
    standoff_m: float = 0.28,
    eye_height_above_plant_m: float = 0.08,
    min_radial_m: float = 0.18,
) -> Optional[PoseXYZQ]:
    """EE on base→plant horizontal ray, looking at plant (forces base yaw toward plant).

    Unlike in-place EE orientation IK, this moves XY toward the plant so joint1 must turn.
    Camera/tool +Z points at the plant from a point slightly above it.
    """
    bx, by, bz = plant_xyz
    r = math.hypot(bx, by)
    if r < 1e-4:
        return None
    ee_r = max(r - standoff_m, min_radial_m)
    ex = bx / r * ee_r
    ey = by / r * ee_r
    ez = bz + eye_height_above_plant_m
    ax, ay, az = bx - ex, by - ey, bz - ez
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-6:
        return None
    qx, qy, qz, qw = quat_cup_toward_berry(ax / n, ay / n, az / n)
    return (ex, ey, ez, qx, qy, qz, qw)


def blend_pose(
    current_xyz: Tuple[float, float, float],
    current_quat: Tuple[float, float, float, float],
    target: PoseXYZQ,
    *,
    position_blend: float = 0.4,
    orient_frac: float = 0.5,
) -> PoseXYZQ:
    """Interpolate current EE pose toward a target look-at pose (small step)."""
    tx, ty, tz, tqx, tqy, tqz, tqw = target
    cx, cy, cz = current_xyz
    b = max(0.0, min(1.0, float(position_blend)))
    px = cx + (tx - cx) * b
    py = cy + (ty - cy) * b
    pz = cz + (tz - cz) * b
    oq = quat_slerp(current_quat, (tqx, tqy, tqz, tqw), max(0.0, min(1.0, orient_frac)))
    return (px, py, pz, oq[0], oq[1], oq[2], oq[3])


def plant_base_yaw(plant_xyz: Tuple[float, float, float]) -> float:
    """Horizontal yaw from base origin to plant (rad, base_link)."""
    return math.atan2(plant_xyz[1], plant_xyz[0])


def yaw_face_standoff_pose(
    plant_xyz: Tuple[float, float, float],
    *,
    standoff_m: float = 0.28,
    eye_height_above_plant_m: float = 0.08,
    min_radial_m: float = 0.18,
) -> Optional[PoseXYZQ]:
    """EE on base→plant horizontal ray, looking at plant (forces base yaw toward plant).

    Unlike in-place EE orientation IK, this moves XY toward the plant so joint1 must turn.
    Camera/tool +Z points at the plant from a point slightly above it.
    """
    bx, by, bz = plant_xyz
    r = math.hypot(bx, by)
    if r < 1e-4:
        return None
    ee_r = max(r - standoff_m, min_radial_m)
    ex = bx / r * ee_r
    ey = by / r * ee_r
    ez = bz + eye_height_above_plant_m
    ax, ay, az = bx - ex, by - ey, bz - ez
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-6:
        return None
    qx, qy, qz, qw = quat_cup_toward_berry(ax / n, ay / n, az / n)
    return (ex, ey, ez, qx, qy, qz, qw)


def blend_pose(
    current_xyz: Tuple[float, float, float],
    current_quat: Tuple[float, float, float, float],
    target: PoseXYZQ,
    *,
    position_blend: float = 0.4,
    orient_frac: float = 0.5,
) -> PoseXYZQ:
    """Interpolate current EE pose toward a target look-at pose (small step)."""
    tx, ty, tz, tqx, tqy, tqz, tqw = target
    cx, cy, cz = current_xyz
    b = max(0.0, min(1.0, float(position_blend)))
    px = cx + (tx - cx) * b
    py = cy + (ty - cy) * b
    pz = cz + (tz - cz) * b
    oq = quat_slerp(current_quat, (tqx, tqy, tqz, tqw), max(0.0, min(1.0, orient_frac)))
    return (px, py, pz, oq[0], oq[1], oq[2], oq[3])


def partial_look_at_pose(
    berry_xyz: Tuple[float, float, float],
    eye_xyz: Tuple[float, float, float],
    eye_quat: Tuple[float, float, float, float],
    standoff_m: float,
    *,
    orient_frac: float = 0.35,
    position_blend: float = 0.0,
) -> Optional[PoseXYZQ]:
    """Small-step look-at: slerp orientation toward plant, optional position blend."""
    full = look_at_pose(
        berry_xyz, eye_xyz, standoff_m, blend=max(0.0, min(1.0, position_blend)))
    if full is None:
        return None
    t = max(0.0, min(1.0, float(orient_frac)))
    qx, qy, qz, qw = quat_slerp(eye_quat, full[3:], t)
    return full[0], full[1], full[2], qx, qy, qz, qw
