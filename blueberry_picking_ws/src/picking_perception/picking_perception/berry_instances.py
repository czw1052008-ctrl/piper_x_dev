"""Split semantic berry blobs into per-fruit instances (P1).

2D: distance-transform peaks + watershed (no detector).
3D: Euclidean clustering of RGB-D points at berry-diameter scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

SEM_BERRY = 1
SEM_BRANCH = 2
SEM_RIGID = 3
SEM_EGO = 4

# Typical ripe blueberry diameter ~8–18 mm; cluster gap ~ that scale.
BERRY_EPS_M = 0.016
BERRY_MIN_POINTS = 4
BERRY_MIN_PX = 8
BERRY_MAX_R_M = 0.020


@dataclass
class InstanceContour:
    id: int
    class_id: int
    polygon_uv: np.ndarray  # (N, 2) float32
    bbox_uv: Tuple[float, float, float, float]
    area_px: int


def _connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    from scipy.ndimage import label as nd_label
    lab, n = nd_label(mask)
    return lab.astype(np.uint16), int(n)


def split_berry_mask_2d(
    berry: np.ndarray,
    *,
    min_peak_dist_px: int = 6,
    min_peak_dt: float = 2.0,
) -> np.ndarray:
    """Watershed split of a berry class mask. Labels 1..K, 0 = background."""
    import cv2

    mask = berry.astype(bool)
    inst = np.zeros(berry.shape[:2], dtype=np.uint16)
    if not np.any(mask):
        return inst
    u8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(u8, cv2.DIST_L2, 5)
    k = int(max(3, int(min_peak_dist_px) * 2 + 1))
    if k % 2 == 0:
        k += 1
    dilated = cv2.dilate(dist, np.ones((k, k), np.uint8))
    peaks = (dist >= dilated - 1e-6) & (dist >= float(min_peak_dt)) & mask
    n_pk, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    if n_pk <= 1:
        inst[mask] = 1
        return inst
    markers = markers.astype(np.int32)
    markers[~mask] = 0
    dummy = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    ws = cv2.watershed(dummy, markers)
    inst[mask & (ws > 0)] = ws[mask & (ws > 0)].astype(np.uint16)
    leftover = mask & (inst == 0)
    if np.any(leftover):
        ys, xs = np.where(inst > 0)
        if ys.size:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.stack([ys, xs], axis=1))
            ly, lx = np.where(leftover)
            _, nn = tree.query(np.stack([ly, lx], axis=1), k=1)
            inst[ly, lx] = inst[ys[nn], xs[nn]]
        else:
            inst[leftover] = 1
    return inst


def instances_from_semantic(sem: np.ndarray) -> np.ndarray:
    """Fallback instance map when no heatmap: berry CC (not watershed), branch/rigid/ego CC."""
    from picking_perception.z_slice_geometry import instances_for_lift
    return instances_for_lift(sem, hm=None)


def cluster_xyz(
    pts: np.ndarray,
    *,
    eps_m: float = BERRY_EPS_M,
    min_pts: int = BERRY_MIN_POINTS,
) -> np.ndarray:
    """Euclidean connected components. Returns per-point labels, -1 = noise."""
    from scipy.spatial import cKDTree

    n = int(pts.shape[0])
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    tree = cKDTree(pts)
    visited = np.zeros(n, dtype=bool)
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        members: List[int] = []
        while stack:
            j = stack.pop()
            members.append(j)
            for k in tree.query_ball_point(pts[j], float(eps_m)):
                if not visited[k]:
                    visited[k] = True
                    stack.append(int(k))
        if len(members) >= int(min_pts):
            labels[members] = cid
            cid += 1
    return labels


def _backproject_uv(
    depth_m: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    *,
    depth_min: float,
    depth_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (keep_index, Nx3 xyz). us/vs are pixel coords."""
    H, W = depth_m.shape[:2]
    us = us.astype(np.int32)
    vs = vs.astype(np.int32)
    ok = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    us, vs = us[ok], vs[ok]
    if us.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.float64)
    z = depth_m[vs, us].astype(np.float64)
    valid = np.isfinite(z) & (z >= depth_min) & (z <= depth_max)
    us, vs, z = us[valid], vs[valid], z[valid]
    if us.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x_c = (us.astype(np.float64) - cx) * z / fx
    y_c = (vs.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x_c, y_c, z], axis=1)
    R = np.asarray(T_base_cam[:3, :3], dtype=np.float64)
    o = np.asarray(T_base_cam[:3, 3], dtype=np.float64)
    xyz = (R @ pts_cam.T).T + o
    lin = vs * W + us
    return lin, xyz


def refine_berry_instances_3d(
    sem: np.ndarray,
    inst: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_base_cam: Optional[np.ndarray],
    *,
    eps_m: float = BERRY_EPS_M,
    stride: int = 2,
    depth_min: float = 0.12,
    depth_max: float = 2.5,
    extent_split_m: float = 0.035,
) -> np.ndarray:
    """Split a berry id if 3D points form multiple clusters (no connecting chain)
    or the cloud is longer than one fruit (k-means)."""
    out = inst.astype(np.uint16, copy=True)
    if T_base_cam is None or K is None or depth_m is None:
        return out
    if depth_m.shape[:2] != sem.shape[:2]:
        import cv2
        depth_m = cv2.resize(
            depth_m.astype(np.float32),
            (sem.shape[1], sem.shape[0]),
            interpolation=cv2.INTER_NEAREST)
    _h, W = sem.shape[:2]
    next_id = int(out.max()) + 1 if out.size else 1
    for iid in list({int(i) for i in np.unique(out) if int(i) > 0}):
        sel = out == iid
        if not np.any(sem[sel] == SEM_BERRY):
            continue
        ys, xs = np.where(sel)
        if ys.size < BERRY_MIN_PX:
            continue
        ys_s = ys[:: max(1, int(stride))]
        xs_s = xs[:: max(1, int(stride))]
        _lin, xyz = _backproject_uv(
            depth_m, K, T_base_cam, xs_s, ys_s,
            depth_min=depth_min, depth_max=depth_max)
        if len(xyz) < BERRY_MIN_POINTS * 2:
            continue
        cl = cluster_xyz(xyz, eps_m=eps_m)
        uniq = [int(c) for c in np.unique(cl) if int(c) >= 0]
        if len(uniq) <= 1:
            extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
            if extent < float(extent_split_m):
                continue
            k = int(max(2, min(6, round(extent / 0.018))))
            cl = _kmeans_xyz(xyz, k)
            uniq = [int(c) for c in np.unique(cl) if int(c) >= 0]
            if len(uniq) <= 1:
                continue
        means = np.stack([xyz[cl == c].mean(axis=0) for c in uniq], axis=0)
        _lin_all, xyz_all = _backproject_uv(
            depth_m, K, T_base_cam, xs, ys,
            depth_min=depth_min, depth_max=depth_max)
        if len(xyz_all) < BERRY_MIN_POINTS:
            continue
        from scipy.spatial import cKDTree
        tree = cKDTree(means)
        _, nn = tree.query(xyz_all, k=1)
        new_ids = []
        for k, _c in enumerate(uniq):
            if k == 0:
                new_ids.append(iid)
            else:
                new_ids.append(next_id)
                next_id += 1
        assigned = np.array([new_ids[int(j)] for j in nn], dtype=np.uint16)
        vs_keep = (_lin_all // W).astype(np.int32)
        us_keep = (_lin_all % W).astype(np.int32)
        out[sel] = np.uint16(iid)
        out[vs_keep, us_keep] = assigned
    return out


def _kmeans_xyz(pts: np.ndarray, k: int, iters: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=int(k), replace=False)
    centers = pts[idx].copy()
    labels = np.zeros(len(pts), dtype=np.int32)
    for _ in range(iters):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d, axis=1).astype(np.int32)
        for j in range(int(k)):
            sel = labels == j
            if np.any(sel):
                centers[j] = pts[sel].mean(axis=0)
    return labels


def contour_for_id(inst: np.ndarray, iid: int, *, max_verts: int = 48) -> Optional[InstanceContour]:
    import cv2

    m = (inst == int(iid)).astype(np.uint8)
    area = int(np.sum(m))
    if area < BERRY_MIN_PX:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, max(0.8, 0.012 * peri), True)
    poly = approx.reshape(-1, 2).astype(np.float32)
    if len(poly) > max_verts:
        idx = np.linspace(0, len(poly) - 1, max_verts).astype(np.int32)
        poly = poly[idx]
    if len(poly) < 3:
        x, y, w, h = cv2.boundingRect(c)
        poly = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    xs, ys = poly[:, 0], poly[:, 1]
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    return InstanceContour(
        id=int(iid), class_id=SEM_BERRY, polygon_uv=poly, bbox_uv=bbox, area_px=area)


def contours_from_instances(
    sem: np.ndarray,
    inst: np.ndarray,
    *,
    classes: Sequence[int] = (SEM_BERRY, SEM_BRANCH, SEM_RIGID, SEM_EGO),
) -> List[InstanceContour]:
    import cv2

    out: List[InstanceContour] = []
    for iid in np.unique(inst):
        iid = int(iid)
        if iid <= 0:
            continue
        sel = inst == iid
        hits = sem[sel]
        if hits.size == 0:
            continue
        cls = int(np.bincount(hits.astype(np.int64)).argmax())
        if cls not in classes:
            continue
        m = sel.astype(np.uint8)
        area = int(np.sum(m))
        if area < BERRY_MIN_PX:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        peri = float(cv2.arcLength(c, True))
        approx = cv2.approxPolyDP(c, max(0.8, 0.012 * peri), True)
        poly = approx.reshape(-1, 2).astype(np.float32)
        if len(poly) < 3:
            x, y, w, h = cv2.boundingRect(c)
            poly = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        bbox = (
            float(poly[:, 0].min()), float(poly[:, 1].min()),
            float(poly[:, 0].max()), float(poly[:, 1].max()))
        out.append(InstanceContour(
            id=iid, class_id=cls, polygon_uv=poly, bbox_uv=bbox, area_px=area))
    return out


def overlay_instance_contours(
    bgr_or_rgb: np.ndarray,
    inst: np.ndarray,
    contours: Sequence[InstanceContour],
    *,
    rgb: bool = True,
) -> np.ndarray:
    import cv2

    vis = bgr_or_rgb.copy()
    for c in contours:
        pts = np.round(c.polygon_uv).astype(np.int32)
        if c.class_id == SEM_BERRY:
            color = (255, 0, 220) if rgb else (220, 0, 255)
        elif c.class_id == SEM_BRANCH:
            color = (20, 200, 40) if rgb else (40, 200, 20)
        elif c.class_id == SEM_EGO:
            color = (40, 200, 220) if rgb else (220, 200, 40)
        else:
            color = (220, 30, 30) if rgb else (30, 30, 220)
        cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)
        x0, y0 = int(pts[0, 0]), int(pts[0, 1])
        cv2.putText(
            vis, str(c.id), (x0, max(12, y0 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis
