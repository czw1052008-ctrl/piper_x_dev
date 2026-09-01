"""Hemisphere workspace occupancy + log-odds fusion + CPU ESDF.

Workspace: upper hemisphere in base_link (|p-c|<=R, z>=z_min).
Depth from fixed (+ wrist) cameras integrates into shared log-odds (late fusion).
Ego = FK capsules (not visual). ACTIVE berry = BERRY; other berries = OCC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2
EGO = 3
BERRY = 4

DEFAULT_CENTER = (0.0, 0.0, 0.0)
DEFAULT_RADIUS_M = 0.85
DEFAULT_Z_MIN = 0.02
DEFAULT_VOXEL_M = 0.02
DEFAULT_CROP_SIZE = 32

L_HIT = 0.85
L_MISS = -0.40
L_MAX = 3.5
L_OCC_THR = 0.85
L_FREE_THR = -0.55


def hemisphere_aabb(
    center: Sequence[float] = DEFAULT_CENTER,
    radius_m: float = DEFAULT_RADIUS_M,
    z_min: float = DEFAULT_Z_MIN,
) -> Tuple[float, float, float, float, float, float]:
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r = float(radius_m)
    return (cx - r, cx + r, cy - r, cy + r, float(z_min), cz + r)


@dataclass
class OccupancyConfig:
    center: Tuple[float, float, float] = DEFAULT_CENTER
    radius_m: float = DEFAULT_RADIUS_M
    z_min: float = DEFAULT_Z_MIN
    voxel_m: float = DEFAULT_VOXEL_M
    depth_min_m: float = 0.15
    depth_max_m: float = 2.5
    ray_stride_px: int = 2
    hit_inflate_voxels: int = 1
    berry_radius_m: float = 0.025
    ego_link_radius_m: float = 0.055
    ego_mid_radius_m: float = 0.048
    ego_tip_radius_m: float = 0.042
    crop_size: int = DEFAULT_CROP_SIZE
    logodds_decay: float = 0.92
    w_fixed: float = 1.0
    w_wrist: float = 1.2
    w_wrist_near: float = 1.6
    wrist_near_z_m: float = 0.35
    n_ray_samples: int = 10

    @property
    def aabb(self) -> Tuple[float, float, float, float, float, float]:
        return hemisphere_aabb(self.center, self.radius_m, self.z_min)


@dataclass
class OccupancyCrop:
    origin_xyz: Tuple[float, float, float]
    voxel_m: float
    size_xyz: Tuple[int, int, int]
    labels: np.ndarray
    esdf: np.ndarray


class OccupancyVolume:
    def __init__(self, cfg: Optional[OccupancyConfig] = None) -> None:
        self.cfg = cfg or OccupancyConfig()
        xmin, xmax, ymin, ymax, zmin, zmax = self.cfg.aabb
        v = float(self.cfg.voxel_m)
        self.origin = np.array([xmin, ymin, zmin], dtype=np.float64)
        self.voxel_m = v
        self.dims = np.array([
            max(1, int(np.ceil((xmax - xmin) / v))),
            max(1, int(np.ceil((ymax - ymin) / v))),
            max(1, int(np.ceil((zmax - zmin) / v))),
        ], dtype=np.int32)
        self.logodds = np.zeros(self.dims, dtype=np.float32)
        self.labels = np.full(self.dims, UNKNOWN, dtype=np.uint8)
        self.esdf = np.full(self.dims, np.nan, dtype=np.float32)
        self._ego_mask = np.zeros(self.dims, dtype=bool)
        self._esdf_valid = False
        self._workspace_mask = self._build_workspace_mask()
        self._active_berries: List[Tuple[float, float, float]] = []
        self._obstacle_berries: List[Tuple[float, float, float]] = []

    def _build_workspace_mask(self) -> np.ndarray:
        nx, ny, nz = [int(x) for x in self.dims]
        ii, jj, kk = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        xyz = self.origin + (np.stack([ii, jj, kk], axis=-1).astype(np.float64) + 0.5) * self.voxel_m
        c = np.asarray(self.cfg.center, dtype=np.float64)
        d2 = np.sum((xyz - c) ** 2, axis=-1)
        return (d2 <= float(self.cfg.radius_m) ** 2) & (xyz[..., 2] >= float(self.cfg.z_min))

    @property
    def aabb(self) -> Tuple[float, float, float, float, float, float]:
        return self.cfg.aabb

    def in_workspace(self, xyz: Sequence[float]) -> bool:
        p = np.asarray(xyz, dtype=np.float64)
        c = np.asarray(self.cfg.center, dtype=np.float64)
        if float(p[2]) < float(self.cfg.z_min):
            return False
        return float(np.sum((p - c) ** 2)) <= float(self.cfg.radius_m) ** 2

    def reset(self) -> None:
        self.logodds.fill(0.0)
        self.labels.fill(UNKNOWN)
        self.esdf.fill(np.nan)
        self._ego_mask.fill(False)
        self._esdf_valid = False

    def decay(self) -> None:
        d = float(self.cfg.logodds_decay)
        if 0.0 < d < 1.0:
            self.logodds *= d
        self._esdf_valid = False

    def world_to_idx(self, xyz: Sequence[float]) -> Optional[Tuple[int, int, int]]:
        p = (np.asarray(xyz, dtype=np.float64) - self.origin) / self.voxel_m
        i, j, k = int(np.floor(p[0])), int(np.floor(p[1])), int(np.floor(p[2]))
        if (0 <= i < self.dims[0] and 0 <= j < self.dims[1]
                and 0 <= k < self.dims[2]):
            return i, j, k
        return None

    def idx_to_world(self, i: int, j: int, k: int) -> np.ndarray:
        return self.origin + (np.array([i, j, k], dtype=np.float64) + 0.5) * self.voxel_m

    def _mark_sphere_mask(
        self,
        center: Sequence[float],
        radius_m: float,
        mask: np.ndarray,
    ) -> None:
        c = np.asarray(center, dtype=np.float64)
        r = float(radius_m)
        if r <= 0:
            return
        lo = np.floor((c - r - self.origin) / self.voxel_m).astype(int)
        hi = np.ceil((c + r - self.origin) / self.voxel_m).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, self.dims)
        if np.any(lo >= hi):
            return
        rs = np.arange(lo[0], hi[0])
        cs = np.arange(lo[1], hi[1])
        zs = np.arange(lo[2], hi[2])
        ii, jj, kk = np.meshgrid(rs, cs, zs, indexing='ij')
        centers = (
            self.origin
            + (np.stack([ii, jj, kk], axis=-1).astype(np.float64) + 0.5) * self.voxel_m
        )
        inside = np.sum((centers - c) ** 2, axis=-1) <= r * r
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside

    def _mark_capsule_mask(
        self,
        a: Sequence[float],
        b: Sequence[float],
        radius_m: float,
        mask: np.ndarray,
        *,
        n_samp: int = 8,
    ) -> None:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        for t in np.linspace(0.0, 1.0, max(2, int(n_samp))):
            self._mark_sphere_mask(a + (b - a) * float(t), radius_m, mask)

    def clear_ego_mask(self) -> None:
        self._ego_mask.fill(False)

    def apply_ego(
        self,
        tip_xyz: Sequence[float],
        link_xyz: Sequence[Sequence[float]],
        *,
        base_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.clear_ego_mask()
        pts: List[np.ndarray] = [np.asarray(base_xyz, dtype=np.float64)]
        for p in link_xyz:
            pts.append(np.asarray(p, dtype=np.float64))
        pts.append(np.asarray(tip_xyz, dtype=np.float64))
        for i in range(len(pts) - 1):
            if i <= 1:
                r = float(self.cfg.ego_link_radius_m)
            elif i >= len(pts) - 2:
                r = float(self.cfg.ego_tip_radius_m)
            else:
                r = float(self.cfg.ego_mid_radius_m)
            self._mark_capsule_mask(pts[i], pts[i + 1], r, self._ego_mask, n_samp=10)
        self._ego_mask &= self._workspace_mask
        self._esdf_valid = False

    def apply_berries(
        self,
        active_positions: Sequence[Sequence[float]],
        obstacle_berry_positions: Sequence[Sequence[float]],
    ) -> None:
        self._active_berries = [tuple(float(x) for x in p) for p in active_positions]
        self._obstacle_berries = [tuple(float(x) for x in p) for p in obstacle_berry_positions]

    def insert_depth_frame(
        self,
        depth_m: np.ndarray,
        K: np.ndarray,
        T_base_cam: np.ndarray,
        *,
        weight: float = 1.0,
    ) -> int:
        H, W = depth_m.shape[:2]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        st = max(1, int(self.cfg.ray_stride_px))
        cam_o = T_base_cam[:3, 3].astype(np.float64)
        R = T_base_cam[:3, :3].astype(np.float64)
        dmin, dmax = float(self.cfg.depth_min_m), float(self.cfg.depth_max_m)
        w = float(weight)

        vs = np.arange(0, H, st, dtype=np.int32)
        us = np.arange(0, W, st, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)
        z = depth_m[vv, uu].astype(np.float64)
        valid = np.isfinite(z) & (z >= dmin) & (z <= dmax)
        if not np.any(valid):
            return 0
        z_v = z[valid]
        u_v = uu[valid].astype(np.float64)
        v_v = vv[valid].astype(np.float64)
        x_c = (u_v - cx) * z_v / fx
        y_c = (v_v - cy) * z_v / fy
        pts_cam = np.stack([x_c, y_c, z_v], axis=1)
        hits_w = (R @ pts_cam.T).T + cam_o

        rel = (hits_w - self.origin[None, :]) / self.voxel_m
        ii = np.floor(rel[:, 0]).astype(np.int32)
        jj = np.floor(rel[:, 1]).astype(np.int32)
        kk = np.floor(rel[:, 2]).astype(np.int32)
        inside = (
            (ii >= 0) & (ii < self.dims[0])
            & (jj >= 0) & (jj < self.dims[1])
            & (kk >= 0) & (kk < self.dims[2])
        )
        hits = 0
        if np.any(inside):
            ii_i, jj_i, kk_i = ii[inside], jj[inside], kk[inside]
            ws = self._workspace_mask[ii_i, jj_i, kk_i]
            ego = self._ego_mask[ii_i, jj_i, kk_i]
            ok = ws & (~ego)
            hits = int(np.sum(ok))
            if hits > 0:
                self.logodds[ii_i[ok], jj_i[ok], kk_i[ok]] += float(L_HIT) * w
                inf = int(self.cfg.hit_inflate_voxels)
                if inf > 0:
                    base_i, base_j, base_k = ii_i[ok], jj_i[ok], kk_i[ok]
                    for di in range(-inf, inf + 1):
                        for dj in range(-inf, inf + 1):
                            for dk in range(-inf, inf + 1):
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = base_i + di, base_j + dj, base_k + dk
                                good = (
                                    (ni >= 0) & (ni < self.dims[0])
                                    & (nj >= 0) & (nj < self.dims[1])
                                    & (nk >= 0) & (nk < self.dims[2])
                                )
                                if not np.any(good):
                                    continue
                                ni, nj, nk = ni[good], nj[good], nk[good]
                                keep = (
                                    self._workspace_mask[ni, nj, nk]
                                    & (~self._ego_mask[ni, nj, nk])
                                )
                                self.logodds[ni[keep], nj[keep], nk[keep]] += (
                                    float(L_HIT) * 0.35 * w)

        n_samp = max(4, int(self.cfg.n_ray_samples))
        fracs = np.linspace(0.08, 0.88, n_samp)
        for f in fracs:
            pts = cam_o[None, :] + (hits_w - cam_o[None, :]) * float(f)
            rel = (pts - self.origin[None, :]) / self.voxel_m
            ii = np.floor(rel[:, 0]).astype(np.int32)
            jj = np.floor(rel[:, 1]).astype(np.int32)
            kk = np.floor(rel[:, 2]).astype(np.int32)
            inside = (
                (ii >= 0) & (ii < self.dims[0])
                & (jj >= 0) & (jj < self.dims[1])
                & (kk >= 0) & (kk < self.dims[2])
            )
            if not np.any(inside):
                continue
            ii, jj, kk = ii[inside], jj[inside], kk[inside]
            keep = self._workspace_mask[ii, jj, kk] & (~self._ego_mask[ii, jj, kk])
            self.logodds[ii[keep], jj[keep], kk[keep]] += float(L_MISS) * w

        np.clip(self.logodds, -float(L_MAX), float(L_MAX), out=self.logodds)
        self._esdf_valid = False
        return hits

    def materialize_labels(self) -> None:
        lab = np.full(self.dims, UNKNOWN, dtype=np.uint8)
        ws = self._workspace_mask
        occ = ws & (self.logodds >= float(L_OCC_THR))
        free = ws & (self.logodds <= float(L_FREE_THR))
        lab[free] = FREE
        lab[occ] = OCCUPIED
        for p in self._obstacle_berries:
            self._paint_sphere_label(lab, p, float(self.cfg.berry_radius_m), OCCUPIED)
        lab[self._ego_mask] = EGO
        for p in self._active_berries:
            self._paint_sphere_label(lab, p, float(self.cfg.berry_radius_m), BERRY)
        lab[~ws] = UNKNOWN
        self.labels = lab
        self._esdf_valid = False

    def _paint_sphere_label(
        self,
        lab: np.ndarray,
        center: Sequence[float],
        radius_m: float,
        label: int,
    ) -> None:
        c = np.asarray(center, dtype=np.float64)
        r = float(radius_m)
        lo = np.floor((c - r - self.origin) / self.voxel_m).astype(int)
        hi = np.ceil((c + r - self.origin) / self.voxel_m).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, self.dims)
        if np.any(lo >= hi):
            return
        rs = np.arange(lo[0], hi[0])
        cs = np.arange(lo[1], hi[1])
        zs = np.arange(lo[2], hi[2])
        ii, jj, kk = np.meshgrid(rs, cs, zs, indexing='ij')
        centers = (
            self.origin
            + (np.stack([ii, jj, kk], axis=-1).astype(np.float64) + 0.5) * self.voxel_m
        )
        mask = np.sum((centers - c) ** 2, axis=-1) <= r * r
        mask &= self._workspace_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][mask] = np.uint8(label)

    def compute_esdf(self) -> None:
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError as exc:
            raise RuntimeError('scipy required for ESDF') from exc
        occ = (self.labels == OCCUPIED)
        dist_vox = distance_transform_edt(~occ).astype(np.float32)
        self.esdf = dist_vox * float(self.voxel_m)
        self.esdf[occ] = 0.0
        self.esdf[~self._workspace_mask] = np.nan
        self._esdf_valid = True

    def ensure_esdf(self) -> None:
        if not self._esdf_valid:
            self.compute_esdf()

    def query(self, xyz: Sequence[float]) -> Tuple[int, float]:
        if not self.in_workspace(xyz):
            return UNKNOWN, float('nan')
        idx = self.world_to_idx(xyz)
        if idx is None:
            return UNKNOWN, float('nan')
        self.ensure_esdf()
        i, j, k = idx
        return int(self.labels[i, j, k]), float(self.esdf[i, j, k])

    def is_free(
        self,
        xyz: Sequence[float],
        *,
        margin_m: float = 0.0,
        allow_berry: bool = True,
    ) -> bool:
        lab, d = self.query(xyz)
        if lab in (OCCUPIED, UNKNOWN, EGO):
            return False
        if lab == BERRY and not allow_berry:
            return False
        if not np.isfinite(d):
            return False
        return d >= float(margin_m)

    def crop_around(self, center: Sequence[float], size: Optional[int] = None) -> OccupancyCrop:
        self.ensure_esdf()
        n = int(size or self.cfg.crop_size)
        half = n // 2
        c_idx = self.world_to_idx(center)
        if c_idx is None:
            p = np.clip(
                (np.asarray(center, dtype=np.float64) - self.origin) / self.voxel_m,
                0, self.dims - 1)
            c_idx = (int(p[0]), int(p[1]), int(p[2]))
        ci, cj, ck = c_idx
        i0 = int(np.clip(ci - half, 0, max(0, self.dims[0] - n)))
        j0 = int(np.clip(cj - half, 0, max(0, self.dims[1] - n)))
        k0 = int(np.clip(ck - half, 0, max(0, self.dims[2] - n)))
        lab = np.full((n, n, n), UNKNOWN, dtype=np.uint8)
        es = np.full((n, n, n), np.nan, dtype=np.float32)
        si1 = min(i0 + n, self.dims[0])
        sj1 = min(j0 + n, self.dims[1])
        sk1 = min(k0 + n, self.dims[2])
        di, dj, dk = si1 - i0, sj1 - j0, sk1 - k0
        lab[:di, :dj, :dk] = self.labels[i0:si1, j0:sj1, k0:sk1]
        es[:di, :dj, :dk] = self.esdf[i0:si1, j0:sj1, k0:sk1]
        origin = tuple(float(x) for x in (self.origin + np.array([i0, j0, k0]) * self.voxel_m))
        return OccupancyCrop(
            origin_xyz=origin,  # type: ignore[arg-type]
            voxel_m=float(self.voxel_m),
            size_xyz=(n, n, n),
            labels=lab,
            esdf=es,
        )

    def occupied_count(self) -> int:
        return int(np.sum(self.labels == OCCUPIED))

    def free_count(self) -> int:
        return int(np.sum(self.labels == FREE))

    def z_slice_labels(self, z_m: float) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        idx = self.world_to_idx([
            self.origin[0] + 0.5 * self.voxel_m,
            self.origin[1] + 0.5 * self.voxel_m,
            z_m,
        ])
        k = idx[2] if idx is not None else int(self.dims[2] // 2)
        k = int(np.clip(k, 0, self.dims[2] - 1))
        xmin, xmax, ymin, ymax, _, _ = self.cfg.aabb
        return self.labels[:, :, k].copy(), (xmin, xmax, ymin, ymax)

    def flatten_labels(self) -> np.ndarray:
        return self.labels.reshape(-1)

    def flatten_esdf(self) -> np.ndarray:
        self.ensure_esdf()
        return self.esdf.reshape(-1)
