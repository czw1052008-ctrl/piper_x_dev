"""Wrist camera extrinsic helpers (link6 -> optical), shared by calib scripts."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / 'config' / 'real_robot.env'


def parse_rpy_rad(rpy_text: str) -> Tuple[float, float, float]:
    """Parse CAMERA_MOUNT_RPY string → (roll, pitch, yaw) rad."""
    parts = [float(x.strip()) for x in rpy_text.split(',')]
    while len(parts) < 3:
        parts.append(0.0)
    return parts[0], parts[1], parts[2]


def load_mount_from_env(path: Path = DEFAULT_ENV) -> Tuple[float, float, float, float]:
    """Return (tx, ty, tz, rx_rad) — rx only; use load_mount_rpy for full RPY."""
    tx, ty, tz, rx, _ry, _rz = load_mount_rpy(path)
    return tx, ty, tz, rx


def load_mount_rpy(path: Path = DEFAULT_ENV) -> Tuple[float, float, float, float, float, float]:
    """Return (tx, ty, tz, roll, pitch, yaw) rad from real_robot.env."""
    text = path.read_text(encoding='utf-8')
    tx = _env_float(text, 'CAMERA_MOUNT_TX', 0.0)
    ty = _env_float(text, 'CAMERA_MOUNT_TY', -0.07)
    tz = _env_float(text, 'CAMERA_MOUNT_TZ', 0.04)
    rx, ry, rz = parse_rpy_rad(_env_str(text, 'CAMERA_MOUNT_RPY', '-0.389557,0.0,0.0'))
    return tx, ty, tz, rx, ry, rz


def _env_float(text: str, key: str, default: float) -> float:
    m = re.search(rf'(?m)^{re.escape(key)}=([^\n#]+)', text)
    return float(m.group(1).strip()) if m else default


def _env_str(text: str, key: str, default: str) -> str:
    m = re.search(rf'(?m)^{re.escape(key)}=([^\n#]+)', text)
    return m.group(1).strip() if m else default


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS tf2 roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def rx_matrix(rx: float) -> np.ndarray:
    return rpy_matrix(rx, 0.0, 0.0)


def T_link6_cam(
    tx: float = 0.0, ty: float = -0.07, tz: float = 0.04,
    rx: float = -0.389557, ry: float = 0.0, rz: float = 0.0,
) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_matrix(rx, ry, rz)
    T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
    return T


def T_base_cam(
    joints: Sequence[float], tx: float, ty: float, tz: float,
    rx: float, ry: float = 0.0, rz: float = 0.0,
) -> np.ndarray:
    from piper_position_ik import fk_link6_T
    return fk_link6_T(list(joints)) @ T_link6_cam(tx, ty, tz, rx, ry, rz)


def uv_to_ray_cam(uv: Tuple[float, float], K: Tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = K
    u, v = uv
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    return ray / (np.linalg.norm(ray) + 1e-12)


def ray_base(T_b_c: np.ndarray, uv: Tuple[float, float],
             K: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    ray_cam = uv_to_ray_cam(uv, K)
    R = T_b_c[:3, :3]
    o = T_b_c[:3, 3].copy()
    d = R @ ray_cam
    d /= np.linalg.norm(d) + 1e-12
    return o, d


def point_to_ray_dist_m(gt: Sequence[float], o: np.ndarray, d: np.ndarray) -> float:
    w = np.asarray(gt, dtype=np.float64) - o
    along = float(np.dot(w, d))
    perp = w - along * d
    return float(np.linalg.norm(perp))


def depth_to_base(
    uv: Tuple[float, float], z_m: float, joints: Sequence[float],
    K: Tuple[float, float, float, float],
    tx: float, ty: float, tz: float,
    rx: float, ry: float = 0.0, rz: float = 0.0,
    *, sphere_r_m: float = 0.0,
) -> np.ndarray:
    fx, fy, cx, cy = K
    u, v = uv
    z_use = float(z_m) + float(sphere_r_m)
    cam = np.array([(u - cx) / fx * z_use, (v - cy) / fy * z_use, z_use], dtype=np.float64)
    T = T_base_cam(joints, tx, ty, tz, rx, ry, rz)
    return (T @ np.array([cam[0], cam[1], cam[2], 1.0], dtype=np.float64))[:3]


def project_gt_uv(
    gt: Sequence[float], joints: Sequence[float], K: Tuple[float, float, float, float],
    tx: float, ty: float, tz: float,
    rx: float, ry: float = 0.0, rz: float = 0.0,
) -> Tuple[float, float]:
    T = T_base_cam(joints, tx, ty, tz, rx, ry, rz)
    p = np.linalg.inv(T) @ np.array([gt[0], gt[1], gt[2], 1.0], dtype=np.float64)
    if p[2] <= 1e-4:
        raise ValueError('GT behind camera')
    fx, fy, cx, cy = K
    return float(fx * p[0] / p[2] + cx), float(fy * p[1] / p[2] + cy)
