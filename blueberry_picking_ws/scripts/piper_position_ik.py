"""Position-only IK for Piper X with joint6 locked at 0.

MoveIt `/compute_ik` (KDL/TRAC) solves full SE(3). With j6 fixed in the URDF
and MoveIt joint_limits, the arm is 5-DOF and almost any absolute 6-D pose
(including 1 cm translation at fixed orientation) returns NO_IK_SOLUTION (-31).

Identity and FK→IK round-trips still succeed because those poses lie on the
reachable 5-DOF manifold. Visual servo therefore maps Cartesian *position*
waypoints with a damped least-squares Jacobian on joints 1–5 (j6 held at 0).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Fixed joint origins from piper_x_description.urdf (parent→child before joint motion).
# Each entry: (xyz, rpy, axis) with revolute axis about local Z after the fixed RPY.
_JOINT_FIXED = (
    # joint1
    ((0.0, 0.0, 0.123), (0.0, 0.0, 3.1415926), (0.0, 0.0, 1.0)),
    # joint2
    ((0.0, 0.0, 0.0), (-1.5707963, -3.005806, -3.1415926), (0.0, 0.0, 1.0)),
    # joint3
    ((0.28503, 0.0, 0.0), (0.0, 0.0, 2.8380798), (0.0, 0.0, 1.0)),
    # joint4
    ((0.27364, 0.0, 0.0), (0.0, 0.0, 0.0806342), (0.0, 0.0, 1.0)),
    # joint5
    ((0.07466, 0.0, 0.0), (-1.5707963, 1.5707963, 0.0), (0.0, 0.0, 1.0)),
    # joint6 (locked at 0)
    ((0.0, -0.035, 0.0), (1.5707963, 0.0, 0.0), (0.0, 0.0, 1.0)),
)

# Soft URDF position limits (j6 forced to 0 regardless).
_JOINT_LIMITS = (
    (-2.6179938, 2.6179938),
    (0.0, 3.1415926),
    (-2.9670597, 0.0),
    (-1.553343, 1.553343),
    (-1.553343, 1.553343),
    (0.0, 0.0),
)


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def _axis_angle(axis: Sequence[float], q: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(ax))
    if n < 1e-12:
        return np.eye(3)
    ax = ax / n
    x, y, z = ax
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def _joint_transform(xyz: Sequence[float], rpy: Sequence[float],
                     axis: Sequence[float], q: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_matrix(rpy[0], rpy[1], rpy[2]) @ _axis_angle(axis, q)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def fk_link6(q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Forward kinematics base_link → link6. Returns (R_3x3, t_3)."""
    if len(q) != 6:
        raise ValueError('expected 6 joint values')
    T = np.eye(4)
    for i, (xyz, rpy, axis) in enumerate(_JOINT_FIXED):
        qi = 0.0 if i == 5 else float(q[i])
        T = T @ _joint_transform(xyz, rpy, axis, qi)
    return T[:3, :3].copy(), T[:3, 3].copy()


def fk_xyz(q: Sequence[float]) -> np.ndarray:
    return fk_link6(q)[1]


def clamp_joints(q: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i, v in enumerate(q):
        lo, hi = _JOINT_LIMITS[i]
        if i == 5:
            out.append(0.0)
        else:
            out.append(float(max(lo, min(hi, float(v)))))
    return out


def _rot_error_vec(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Small-angle orientation error ω such that R_cur ≈ exp([ω]×) R_des."""
    R_err = R_cur @ R_des.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ], dtype=float)


def fk_link6_T(q: Sequence[float]) -> np.ndarray:
    """base_link → link6 as 4×4."""
    R, t = fk_link6(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def position_ik(
    target_xyz: Sequence[float],
    seed_q: Sequence[float],
    *,
    max_iters: int = 12,
    tol_m: float = 8e-4,
    damp: float = 1e-3,
    max_step: float = 0.20,
    eps: float = 1e-4,
) -> Optional[List[float]]:
    """DEPRECATED for vision paths: position-only, orientation free (can flip wrist).

    Prefer aim_uv_ik (mid-range) or approach_axis_ik / keep_orient (contact).
    Kept for tiny non-vision nudges only.

    Returns joint list on success, or None if the residual stays above tol_m.
    """
    goal = np.asarray(target_xyz, dtype=float).reshape(3)
    q = clamp_joints(seed_q)
    best_q = list(q)
    best_err = float('inf')

    for _ in range(max_iters):
        p = fk_xyz(q)
        err = goal - p
        nerr = float(np.linalg.norm(err))
        if nerr < best_err:
            best_err = nerr
            best_q = list(q)
        if nerr <= tol_m:
            return clamp_joints(q)

        J = np.zeros((3, 5), dtype=float)
        for i in range(5):
            dq = list(q)
            dq[i] = float(dq[i] + eps)
            J[:, i] = (fk_xyz(dq) - p) / eps

        A = J @ J.T + damp * np.eye(3)
        try:
            dq5 = J.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            return None
        step = float(np.linalg.norm(dq5))
        if step > max_step:
            dq5 *= max_step / step
        for i in range(5):
            q[i] = float(q[i] + dq5[i])
        q[5] = 0.0
        q = clamp_joints(q)

    if best_err <= tol_m * 2.5:
        return clamp_joints(best_q)
    return None


def position_ik_keep_orient(
    target_xyz: Sequence[float],
    seed_q: Sequence[float],
    *,
    max_iters: int = 40,
    tol_m: float = 1.5e-3,
    tol_rad: float = 0.04,
    w_pos: float = 1.0,
    w_ori: float = 0.35,
    damp: float = 2e-3,
    max_step: float = 0.12,
    eps: float = 1e-4,
) -> Optional[List[float]]:
    """Position waypoint with seed orientation held as soft constraint (j6=0).

    5-DOF cannot always hit exact SE(3); weighted LS prefers keeping look
    direction over large base yaw while still closing Cartesian position.
    """
    goal = np.asarray(target_xyz, dtype=float).reshape(3)
    q0 = clamp_joints(seed_q)
    R_des, _ = fk_link6(q0)
    q = list(q0)
    best_q = list(q)
    best_cost = float('inf')

    for _ in range(max_iters):
        R, p = fk_link6(q)
        e_pos = goal - p
        e_ori = _rot_error_vec(R, R_des)
        n_pos = float(np.linalg.norm(e_pos))
        n_ori = float(np.linalg.norm(e_ori))
        cost = w_pos * n_pos + w_ori * n_ori
        if cost < best_cost:
            best_cost = cost
            best_q = list(q)
        if n_pos <= tol_m and n_ori <= tol_rad:
            return clamp_joints(q)

        err = np.concatenate([w_pos * e_pos, w_ori * e_ori])
        J = np.zeros((6, 5), dtype=float)
        for i in range(5):
            dq = list(q)
            dq[i] = float(dq[i] + eps)
            Ri, pi = fk_link6(dq)
            J[:3, i] = w_pos * (pi - p) / eps
            J[3:, i] = w_ori * (_rot_error_vec(Ri, R_des) - e_ori) / eps

        A = J @ J.T + damp * np.eye(6)
        try:
            dq5 = J.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            break
        step = float(np.linalg.norm(dq5))
        if step > max_step:
            dq5 *= max_step / step
        if step < 1e-9:
            break
        for i in range(5):
            q[i] = float(q[i] + dq5[i])
        q[5] = 0.0
        q = clamp_joints(q)

    R_b, p_b = fk_link6(best_q)
    if (
        float(np.linalg.norm(goal - p_b)) <= tol_m * 2.5
        and float(np.linalg.norm(_rot_error_vec(R_b, R_des))) <= tol_rad * 2.5
    ):
        return clamp_joints(best_q)
    # Position still usable if orientation stayed closer than free position_ik.
    if float(np.linalg.norm(goal - p_b)) <= tol_m * 2.5:
        return clamp_joints(best_q)
    return None


def position_ik_keep_orient_chunked(
    target_xyz: Sequence[float],
    seed_q: Sequence[float],
    *,
    chunk_m: float = 0.04,
    **kwargs,
) -> Optional[List[float]]:
    """keep_orient via short Cartesian waypoints (large single-shot DLS often fails).

    5-DOF *can* reach the XYZ (free ``position_ik`` succeeds); a single
    attitude-weighted solve from ~20–30 cm away often stalls ~5–10 mm short.
    Chunking (~4 cm) lets orientation drift gradually per segment and recovers
    the same class of solutions as free IK without dumping attitude in one jump.
    """
    goal = np.asarray(target_xyz, dtype=float).reshape(3)
    q = clamp_joints(seed_q)
    p0 = fk_xyz(q)
    delta = goal - p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return list(q)
    # Prefer single-shot when already short.
    if dist <= float(chunk_m) * 1.25:
        return position_ik_keep_orient(tuple(goal.tolist()), q, **kwargs)

    n = max(2, int(math.ceil(dist / max(1e-3, float(chunk_m)))))
    for i in range(1, n + 1):
        ti = tuple((p0 + delta * (i / n)).tolist())
        qi = position_ik_keep_orient(ti, q, **kwargs)
        if qi is None:
            return None
        q = qi
    # Final polish onto exact goal from last waypoint seed.
    qf = position_ik_keep_orient(tuple(goal.tolist()), q, **kwargs)
    return qf if qf is not None else list(q)


def _point_in_cam(
    q: Sequence[float],
    point_base: np.ndarray,
    T_link6_cam: np.ndarray,
) -> Optional[np.ndarray]:
    T_bc = fk_link6_T(q) @ T_link6_cam
    p = _invert_T(T_bc) @ np.array(
        [point_base[0], point_base[1], point_base[2], 1.0], dtype=float)
    if float(p[2]) <= 1e-4:
        return None
    return p[:3].copy()


def aim_uv_ik(
    point_base: Sequence[float],
    aim_uv: Sequence[float],
    seed_q: Sequence[float],
    T_link6_cam: np.ndarray,
    *,
    focal_px: float,
    image_wh: Tuple[float, float] = (640.0, 480.0),
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    z_ref: Optional[float] = None,
    w_uv: float = 1.0,
    w_z: float = 0.35,
    max_iters: int = 60,
    tol_pix: float = 10.0,
    tol_z_m: float = 0.025,
    damp: float = 3e-3,
    max_step: float = 0.15,
    eps: float = 1e-4,
) -> Optional[List[float]]:
    """5-DOF IK: put a fixed base point on aim UV; soft-keep cam depth.

    Constraints (fits j1–j5, j6=0):
      • image (u,v) → aim_uv
      • optional z_cam ≈ z_ref (do not dive into the plant)
    Orientation is free except as required by those constraints — no SE(3)
    keep_orient / free position_ik split.
    """
    pt = np.asarray(point_base, dtype=float).reshape(3)
    u_aim, v_aim = float(aim_uv[0]), float(aim_uv[1])
    fx_ = float(focal_px if fx is None else fx)
    fy_ = float(focal_px if fy is None else fy)
    cx_ = float(0.5 * float(image_wh[0]) if cx is None else cx)
    cy_ = float(0.5 * float(image_wh[1]) if cy is None else cy)
    q = clamp_joints(seed_q)
    if z_ref is None:
        p0 = _point_in_cam(q, pt, T_link6_cam)
        z_ref = float(p0[2]) if p0 is not None else 0.35

    best_q = list(q)
    best_pix = float('inf')

    for _ in range(max_iters):
        p_c = _point_in_cam(q, pt, T_link6_cam)
        if p_c is None:
            break
        u = cx_ + fx_ * float(p_c[0]) / float(p_c[2])
        v = cy_ + fy_ * float(p_c[1]) / float(p_c[2])
        du = u_aim - u
        dv = v_aim - v
        dz = float(z_ref) - float(p_c[2])
        pix = float(np.hypot(du, dv))
        if pix < best_pix:
            best_pix = pix
            best_q = list(q)
        if pix <= tol_pix and abs(dz) <= tol_z_m:
            return clamp_joints(q)

        err = np.array([w_uv * du, w_uv * dv, w_z * dz], dtype=float)
        J = np.zeros((3, 5), dtype=float)
        for i in range(5):
            dq = list(q)
            dq[i] = float(dq[i] + eps)
            p_i = _point_in_cam(dq, pt, T_link6_cam)
            if p_i is None:
                continue
            ui = cx_ + fx_ * float(p_i[0]) / float(p_i[2])
            vi = cy_ + fy_ * float(p_i[1]) / float(p_i[2])
            # J maps Δq → Δ(u,v,z); solve J dq = (aim-u, aim-v, z_ref-z).
            J[0, i] = w_uv * (ui - u) / eps
            J[1, i] = w_uv * (vi - v) / eps
            J[2, i] = w_z * (float(p_i[2]) - float(p_c[2])) / eps

        A = J @ J.T + damp * np.eye(3)
        try:
            dq5 = J.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            break
        step = float(np.linalg.norm(dq5))
        if step > max_step:
            dq5 *= max_step / step
        if step < 1e-9:
            break
        for i in range(5):
            q[i] = float(q[i] + dq5[i])
        q[5] = 0.0
        q = clamp_joints(q)

    p_b = _point_in_cam(best_q, pt, T_link6_cam)
    if p_b is None:
        return None
    ub = cx_ + fx_ * float(p_b[0]) / float(p_b[2])
    vb = cy_ + fy_ * float(p_b[1]) / float(p_b[2])
    if float(np.hypot(u_aim - ub, v_aim - vb)) <= tol_pix * 2.0:
        return clamp_joints(best_q)
    return None


def approach_axis_ik(
    target_xyz: Sequence[float],
    berry_base: Sequence[float],
    seed_q: Sequence[float],
    T_link6_cam: np.ndarray,
    *,
    w_pos: float = 1.0,
    w_dir: float = 0.4,
    max_iters: int = 60,
    tol_m: float = 2.5e-3,
    tol_dir_rad: float = 0.12,
    damp: float = 2e-3,
    max_step: float = 0.12,
    eps: float = 1e-4,
) -> Optional[List[float]]:
    """5-DOF IK: EE position + camera +Z aimed at berry (roll free).

    Replaces free position_ik on contact climbs: attitude may change, but only
    to look at the berry — not an arbitrary wrist flip.
    """
    goal = np.asarray(target_xyz, dtype=float).reshape(3)
    berry = np.asarray(berry_base, dtype=float).reshape(3)
    q = clamp_joints(seed_q)
    best_q = list(q)
    best_cost = float('inf')
    z_cam_link = T_link6_cam[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=float)

    def _err(qv: Sequence[float]) -> Tuple[np.ndarray, float, float]:
        R, p = fk_link6(qv)
        e_pos = goal - p
        # Camera optical axis in base.
        z_base = R @ z_cam_link
        cam_o = (fk_link6_T(qv) @ T_link6_cam)[:3, 3]
        to_b = berry - cam_o
        n = float(np.linalg.norm(to_b))
        if n < 1e-6:
            e_dir = np.zeros(3)
            ang = 0.0
        else:
            des = to_b / n
            # Small rotation taking z_base → des: z × des
            e_dir = np.cross(z_base, des)
            cos_a = float(np.clip(np.dot(z_base, des), -1.0, 1.0))
            ang = float(np.arccos(cos_a))
        n_pos = float(np.linalg.norm(e_pos))
        return (
            np.concatenate([w_pos * e_pos, w_dir * e_dir]),
            n_pos,
            ang,
        )

    for _ in range(max_iters):
        err, n_pos, ang = _err(q)
        cost = n_pos + 0.25 * ang
        if cost < best_cost:
            best_cost = cost
            best_q = list(q)
        if n_pos <= tol_m and ang <= tol_dir_rad:
            return clamp_joints(q)

        J = np.zeros((6, 5), dtype=float)
        for i in range(5):
            dq = list(q)
            dq[i] = float(dq[i] + eps)
            e_i, _, _ = _err(dq)
            J[:, i] = (e_i - err) / eps

        A = J @ J.T + damp * np.eye(6)
        try:
            # J = ∂err/∂q; descend ||err|| with J dq = -err.
            dq5 = J.T @ np.linalg.solve(A, -err)
        except np.linalg.LinAlgError:
            break
        step = float(np.linalg.norm(dq5))
        if step > max_step:
            dq5 *= max_step / step
        if step < 1e-9:
            break
        for i in range(5):
            q[i] = float(q[i] + dq5[i])
        q[5] = 0.0
        q = clamp_joints(q)

    _, n_pos, ang = _err(best_q)
    if n_pos <= tol_m * 2.5 and ang <= tol_dir_rad * 2.0:
        return clamp_joints(best_q)
    if n_pos <= tol_m * 2.5:
        return clamp_joints(best_q)
    return None


def approach_axis_ik_chunked(
    target_xyz: Sequence[float],
    berry_base: Sequence[float],
    seed_q: Sequence[float],
    T_link6_cam: np.ndarray,
    *,
    chunk_m: float = 0.04,
    **kwargs,
) -> Optional[List[float]]:
    """approach_axis via short EE waypoints (long single-shot often stalls).

    Each chunk re-aims camera +Z at the berry while advancing position. Prefer
    this over keep_orient_chunked on near contact — pure translate drifts the
    wrist until the berry leaves the frame (161209: look ~38°).
    """
    goal = np.asarray(target_xyz, dtype=float).reshape(3)
    q = clamp_joints(seed_q)
    p0 = fk_xyz(q)
    delta = goal - p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return list(q)
    if dist <= float(chunk_m) * 1.25:
        return approach_axis_ik(
            tuple(goal.tolist()), berry_base, q, T_link6_cam, **kwargs)

    n = max(2, int(math.ceil(dist / max(1e-3, float(chunk_m)))))
    last_ok = list(q)
    for i in range(1, n + 1):
        ti = tuple((p0 + delta * (i / n)).tolist())
        qi = approach_axis_ik(ti, berry_base, q, T_link6_cam, **kwargs)
        if qi is None:
            # Accept partial progress if we advanced at least one chunk.
            if i == 1:
                return None
            return last_ok
        q = qi
        last_ok = list(q)
    qf = approach_axis_ik(
        tuple(goal.tolist()), berry_base, q, T_link6_cam, **kwargs)
    return qf if qf is not None else last_ok


def cup_axis_ik(
    target_xyz: Sequence[float],
    berry_base: Sequence[float],
    seed_q: Sequence[float],
    *,
    tip_offset_link6: Optional[Sequence[float]] = None,
    w_pos: float = 1.0,
    w_dir: float = 0.45,
    max_iters: int = 80,
    tol_m: float = 2.5e-3,
    tol_dir_rad: float = 0.12,
    damp: float = 2e-3,
    max_step: float = 0.15,
    eps: float = 1e-4,
) -> Optional[List[float]]:
    """5-DOF IK: EE position + cup axis (link6 +Z) aimed tip→berry.

    Near contact must keep the soft cup facing the fruit. Camera optical aim
    (approach_axis) is wrong here: cam is ~8 cm off the tip, so cam-look and
    cup-look diverge near contact. keep_orient_chunked also fails — each
    waypoint rewrites R_des and the cup drifts (161209: 4.8°→12.6°).
    """
    goal = np.asarray(target_xyz, dtype=float).reshape(3)
    berry = np.asarray(berry_base, dtype=float).reshape(3)
    off = np.asarray(
        tip_offset_link6 if tip_offset_link6 is not None else (0.0, 0.0, 0.05),
        dtype=float,
    ).reshape(3)
    q = clamp_joints(seed_q)
    best_q = list(q)
    best_cost = float('inf')

    def _err(qv: Sequence[float]) -> Tuple[np.ndarray, float, float]:
        R, p = fk_link6(qv)
        e_pos = goal - p
        z_cup = R @ np.array([0.0, 0.0, 1.0], dtype=float)
        tip = p + R @ off
        to_b = berry - tip
        n = float(np.linalg.norm(to_b))
        if n < 1e-6:
            e_dir = np.zeros(3)
            ang = 0.0
        else:
            des = to_b / n
            e_dir = np.cross(z_cup, des)
            cos_a = float(np.clip(np.dot(z_cup, des), -1.0, 1.0))
            ang = float(np.arccos(cos_a))
        n_pos = float(np.linalg.norm(e_pos))
        return (
            np.concatenate([w_pos * e_pos, w_dir * e_dir]),
            n_pos,
            ang,
        )

    for _ in range(max_iters):
        err, n_pos, ang = _err(q)
        cost = n_pos + 0.25 * ang
        if cost < best_cost:
            best_cost = cost
            best_q = list(q)
        if n_pos <= tol_m and ang <= tol_dir_rad:
            return clamp_joints(q)

        J = np.zeros((6, 5), dtype=float)
        for i in range(5):
            dq = list(q)
            dq[i] = float(dq[i] + eps)
            e_i, _, _ = _err(dq)
            J[:, i] = (e_i - err) / eps

        A = J @ J.T + damp * np.eye(6)
        try:
            dq5 = J.T @ np.linalg.solve(A, -err)
        except np.linalg.LinAlgError:
            break
        step = float(np.linalg.norm(dq5))
        if step > max_step:
            dq5 *= max_step / step
        if step < 1e-9:
            break
        for i in range(5):
            q[i] = float(q[i] + dq5[i])
        q[5] = 0.0
        q = clamp_joints(q)

    _, n_pos, ang = _err(best_q)
    if n_pos <= tol_m * 2.5 and ang <= tol_dir_rad * 2.0:
        return clamp_joints(best_q)
    if n_pos <= tol_m * 2.5:
        return clamp_joints(best_q)
    return None


def cup_axis_ik_chunked(
    target_xyz: Sequence[float],
    berry_base: Sequence[float],
    seed_q: Sequence[float],
    *,
    chunk_m: float = 0.04,
    tip_offset_link6: Optional[Sequence[float]] = None,
    **kwargs,
) -> Optional[List[float]]:
    """cup_axis_ik along short EE waypoints (near contact primary)."""
    goal = np.asarray(target_xyz, dtype=float).reshape(3)
    q = clamp_joints(seed_q)
    p0 = fk_xyz(q)
    delta = goal - p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return list(q)
    kw = dict(kwargs)
    if tip_offset_link6 is not None:
        kw['tip_offset_link6'] = tip_offset_link6
    if dist <= float(chunk_m) * 1.25:
        return cup_axis_ik(tuple(goal.tolist()), berry_base, q, **kw)

    n = max(2, int(math.ceil(dist / max(1e-3, float(chunk_m)))))
    last_ok = list(q)
    for i in range(1, n + 1):
        ti = tuple((p0 + delta * (i / n)).tolist())
        qi = cup_axis_ik(ti, berry_base, q, **kw)
        if qi is None:
            if i == 1:
                return None
            return last_ok
        q = qi
        last_ok = list(q)
    qf = cup_axis_ik(tuple(goal.tolist()), berry_base, q, **kw)
    return qf if qf is not None else last_ok
