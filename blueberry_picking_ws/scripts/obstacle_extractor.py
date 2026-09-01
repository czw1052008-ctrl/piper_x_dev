"""Extract obstacle occupancy spheres from RGB-D (DEPRECATED).

Prefer occupancy_map.OccupancyVolume + occupancy_map_node (Occupancy+ESDF).
Kept for offline imports / old experiments only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_OBSTACLE_RADIUS_M = 0.04


@dataclass
class BBox2D:
    u0: float
    v0: float
    u1: float
    v1: float

    def contains(self, u: float, v: float, *, dilate_px: float = 0.0) -> bool:
        return (
            self.u0 - dilate_px <= u <= self.u1 + dilate_px
            and self.v0 - dilate_px <= v <= self.v1 + dilate_px
        )


@dataclass
class DepthFrameSpec:
    depth_m: np.ndarray
    K: np.ndarray
    T_base_cam: np.ndarray
    berry_bboxes: List[BBox2D] = field(default_factory=list)


@dataclass
class ObstacleExtractConfig:
    depth_min_m: float = 0.15
    depth_max_m: float = 2.0
    voxel_m: float = 0.03
    max_obstacles: int = 12
    berry_exclude_radius_m: float = 0.028
    self_exclude_radius_m: float = 0.06
    arm_link_exclude_radius_m: float = 0.055
    bbox_dilate_px: float = 8.0
    default_radius_m: float = DEFAULT_OBSTACLE_RADIUS_M
    subsample_stride: int = 4
    min_voxel_points: int = 12
    # Keep only points inside workspace AABB (xmin,xmax,ymin,ymax,zmin,zmax).
    workspace_aabb: Tuple[float, float, float, float, float, float] = (
        -0.15, 0.70, 0.10, 0.95, 0.02, 0.65,
    )


@dataclass
class ObstacleSphere:
    position: Tuple[float, float, float]
    radius_m: float = DEFAULT_OBSTACLE_RADIUS_M
    source: str = 'depth_voxel'
    bbox_uv: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
    score: float = 0.0


def decode_depth_m(msg_encoding: str, raw: np.ndarray, height: int, width: int) -> np.ndarray:
    enc = msg_encoding
    if enc in ('32FC1', '32FC'):
        return raw.reshape(height, width).astype(np.float32)
    if enc in ('16UC1', 'mono16'):
        mm = raw.reshape(height, width).astype(np.float32)
        return mm / 1000.0
    return raw.reshape(height, width, -1).astype(np.float32)


def backproject_depth_to_base(
    depth: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    *,
    depth_min_m: float,
    depth_max_m: float,
    stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pts_base Nx3, u N, v N) for valid depth pixels."""
    H, W = depth.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    st = max(1, int(stride))
    vs = np.arange(0, H, st, dtype=np.int32)
    us = np.arange(0, W, st, dtype=np.int32)
    u_grid, v_grid = np.meshgrid(us, vs)
    z = depth[v_grid, u_grid].astype(np.float64)
    valid = (z >= depth_min_m) & (z <= depth_max_m) & np.isfinite(z)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    z_v = z[valid]
    u_v = u_grid[valid].astype(np.float64)
    v_v = v_grid[valid].astype(np.float64)
    x_c = (u_v - cx) * z_v / fx
    y_c = (v_v - cy) * z_v / fy
    ones = np.ones_like(z_v)
    pts_cam = np.stack([x_c, y_c, z_v, ones], axis=0)
    pts_base = (T_base_cam @ pts_cam)[:3].T
    return pts_base, u_v.astype(np.int32), v_v.astype(np.int32)


def _mask_berry_pixels(
    u: np.ndarray,
    v: np.ndarray,
    bboxes: Sequence[BBox2D],
    *,
    dilate_px: float,
) -> np.ndarray:
    if not bboxes or u.size == 0:
        return np.zeros(u.shape[0], dtype=bool)
    keep = np.zeros(u.shape[0], dtype=bool)
    for bb in bboxes:
        keep |= (
            (u >= bb.u0 - dilate_px) & (u <= bb.u1 + dilate_px)
            & (v >= bb.v0 - dilate_px) & (v <= bb.v1 + dilate_px)
        )
    return keep


def _exclude_spheres(
    pts: np.ndarray,
    centers: Sequence[Sequence[float]],
    radius_m: float,
) -> np.ndarray:
    if pts.size == 0 or not centers or radius_m <= 0:
        return np.ones(pts.shape[0], dtype=bool)
    r2 = float(radius_m) ** 2
    ok = np.ones(pts.shape[0], dtype=bool)
    for c in centers:
        cc = np.asarray(c, dtype=np.float64).reshape(3)
        d2 = np.sum((pts - cc) ** 2, axis=1)
        ok &= d2 > r2
    return ok


def voxel_top_k(
    pts: np.ndarray,
    *,
    voxel_m: float,
    k: int,
    min_points: int,
) -> List[Tuple[np.ndarray, int]]:
    if pts.size == 0 or k <= 0:
        return []
    v = max(1e-3, float(voxel_m))
    keys = np.floor(pts / v).astype(np.int64)
    buckets: dict = {}
    for i, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(pts[i])
    ranked = []
    for _, arr in buckets.items():
        if len(arr) < min_points:
            continue
        block = np.asarray(arr, dtype=np.float64)
        ranked.append((block.mean(axis=0), len(arr)))
    ranked.sort(key=lambda x: -x[1])
    return ranked[:k]


def _in_aabb(
    pts: np.ndarray,
    aabb: Sequence[float],
) -> np.ndarray:
    if pts.size == 0 or len(aabb) < 6:
        return np.ones(pts.shape[0], dtype=bool)
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in aabb[:6]]
    return (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        & (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    )


def extract_obstacle_spheres(
    frames: Sequence[DepthFrameSpec],
    *,
    berry_positions_base: Sequence[Sequence[float]],
    tip_xyz_base: Optional[Sequence[float]] = None,
    arm_link_xyz_base: Optional[Sequence[Sequence[float]]] = None,
    cfg: Optional[ObstacleExtractConfig] = None,
) -> List[ObstacleSphere]:
    """Fuse depth frames → top-K obstacle spheres in base_link.

    Occupied = depth points in workspace, excluding berries and arm/EE (ego).
    Free space for sim is the complement of these spheres ∪ berry ∪ ego inside AABB.
    """
    cfg = cfg or ObstacleExtractConfig()
    all_pts: List[np.ndarray] = []
    for fr in frames:
        pts, u, v = backproject_depth_to_base(
            fr.depth_m, fr.K, fr.T_base_cam,
            depth_min_m=cfg.depth_min_m,
            depth_max_m=cfg.depth_max_m,
            stride=cfg.subsample_stride,
        )
        if pts.size == 0:
            continue
        berry_2d = _mask_berry_pixels(u, v, fr.berry_bboxes, dilate_px=cfg.bbox_dilate_px)
        ok = ~berry_2d
        ok &= _in_aabb(pts, cfg.workspace_aabb)
        ok &= _exclude_spheres(pts, berry_positions_base, cfg.berry_exclude_radius_m)
        if tip_xyz_base is not None:
            ok &= _exclude_spheres(pts, [tip_xyz_base], cfg.self_exclude_radius_m)
        if arm_link_xyz_base:
            ok &= _exclude_spheres(
                pts, arm_link_xyz_base, cfg.arm_link_exclude_radius_m)
        sel = pts[ok]
        if sel.size > 0:
            all_pts.append(sel)
    if not all_pts:
        return []
    merged = np.vstack(all_pts)
    top = voxel_top_k(
        merged,
        voxel_m=cfg.voxel_m,
        k=cfg.max_obstacles,
        min_points=cfg.min_voxel_points,
    )
    out: List[ObstacleSphere] = []
    for i, (centroid, count) in enumerate(top):
        out.append(ObstacleSphere(
            position=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            radius_m=cfg.default_radius_m,
            source='depth_voxel',
            score=float(count),
        ))
    return out


def project_base_to_uv(
    xyz_base: Sequence[float],
    K: np.ndarray,
    T_base_cam: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Project base_link point to camera UV (for debug bbox_uv)."""
    T_cam_base = np.linalg.inv(T_base_cam)
    p = np.array([*xyz_base, 1.0], dtype=np.float64)
    pc = T_cam_base @ p
    if pc[2] <= 1e-4:
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * pc[0] / pc[2] + cx
    v = fy * pc[1] / pc[2] + cy
    return float(u), float(v)
