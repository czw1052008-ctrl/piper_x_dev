"""P3 lift: semantic instances → base_z slices with 2D polygons.

Appearance comes from mask ∩ depth, not spheres/capsules/OBBs.
Each object is a stack of XY contours (optionally raster-filled for RViz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

SEM_BG = 0
SEM_BERRY = 1
SEM_BRANCH = 2
SEM_RIGID = 3
SEM_EGO = 4
SEM_RESIDUAL = 5  # leftover valid depth, not a DINOv3 class

CLASS_NAMES = {
    SEM_BERRY: 'berry',
    SEM_BRANCH: 'branch',
    SEM_RIGID: 'rigid',
    SEM_EGO: 'ego',
    SEM_RESIDUAL: 'residual',
}

# Layer thickness (m). Adaptive cap via max_slices.
DZ_M = {
    SEM_BERRY: 0.004,
    SEM_BRANCH: 0.006,
    SEM_RIGID: 0.010,
    SEM_EGO: 0.008,
    SEM_RESIDUAL: 0.012,
}
CELL_M = {
    SEM_BERRY: 0.003,
    SEM_BRANCH: 0.004,
    SEM_RIGID: 0.006,
    SEM_EGO: 0.005,
    SEM_RESIDUAL: 0.008,
}
MIN_PX = {
    SEM_BERRY: 8,
    SEM_BRANCH: 12,
    SEM_RIGID: 20,
    SEM_EGO: 20,
    SEM_RESIDUAL: 40,
}
BACKPROJECT_STRIDE = {
    SEM_BERRY: 1,
    SEM_BRANCH: 2,
    SEM_RIGID: 2,
    SEM_EGO: 2,
    SEM_RESIDUAL: 3,
}

MAX_SLICES = 48
MAX_POLY_VERTS = 64
MIN_PTS_SLICE = 4
DEPTH_MIN_M = 0.08   # sensor near clip; replace with camera_info when available
DEPTH_MAX_M = 2.5    # sensor far clip; replace with camera_info when available


@dataclass
class ZSlice:
    z_min: float
    z_max: float
    xy: np.ndarray  # (N, 2) closed-or-open outer contour, metres, base_link XY


@dataclass
class SlicedObject:
    id: int
    class_id: int
    pts_xyz: np.ndarray
    slices: List[ZSlice] = field(default_factory=list)
    centroid_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_pts: int = 0
    xyz_std: float = 0.0
    source: str = 'fixed'  # fixed | wrist | fused
    visible_wrist: bool = False
    centroid_fixed_xyz: Optional[Tuple[float, float, float]] = None
    fuse_delta_m: float = -1.0
    pick_role: int = 0  # 0 pending 1 active 2 done


@dataclass
class FuseReport:
    active_id: int = -1
    associated: bool = False
    dist_m: float = -1.0
    delta_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_wrist_berries: int = 0
    n_replaced: int = 0
    n_merged_struct: int = 0
    n_wrist_extra: int = 0
    reject_reason: str = ''
    min_pair_m: float = -1.0


def _depth_locally_consistent(
    depth_m: np.ndarray, depth_min: float, depth_max: float, *, win: int = 5,
) -> np.ndarray:
    """Image-space speckle mask. Isolated depths that disagree with neighbors are dropped.

    Uses local median / MAD so it does not depend on scene height in metres.
    """
    from scipy.ndimage import median_filter
    z = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(z) & (z >= depth_min) & (z <= depth_max)
    filled = np.where(valid, z, 0.0)
    med = median_filter(filled, size=int(win))
    dlt = np.abs(z - med)
    mad = median_filter(dlt, size=int(win)) + 1e-4
    return valid & (dlt <= (4.0 * 1.4826 * mad))


def backproject_mask(
    depth_m: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    mask: np.ndarray,
    *,
    stride: int = 1,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
) -> np.ndarray:
    """Nx3 points in base_link."""
    if mask.shape[:2] != depth_m.shape[:2]:
        import cv2
        mask = cv2.resize(
            mask.astype(np.uint8),
            (depth_m.shape[1], depth_m.shape[0]),
            interpolation=cv2.INTER_NEAREST).astype(bool)
    H, W = depth_m.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    step = max(1, int(stride))
    vs = np.arange(0, H, step)
    us = np.arange(0, W, step)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu]
    keep = mask[vv, uu].astype(bool) & np.isfinite(z) & (z >= depth_min) & (z <= depth_max)
    speckle_ok = _depth_locally_consistent(depth_m, depth_min, depth_max)
    keep = keep & speckle_ok[vv, uu]
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64)
    z_v = z[keep].astype(np.float64)
    u_v = uu[keep].astype(np.float64)
    v_v = vv[keep].astype(np.float64)
    pts_cam = np.stack([(u_v - cx) * z_v / fx, (v_v - cy) * z_v / fy, z_v], axis=1)
    R = np.asarray(T_base_cam[:3, :3], dtype=np.float64)
    o = np.asarray(T_base_cam[:3, 3], dtype=np.float64)
    return (R @ pts_cam.T).T + o


def _cc_ids(mask: np.ndarray, min_px: int) -> List[np.ndarray]:
    from scipy.ndimage import label as nd_label
    lab, n = nd_label(mask.astype(bool))
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        if int(np.sum(sel)) >= int(min_px):
            out.append(sel)
    return out


def instances_for_lift(
    sem: np.ndarray,
    hm: Optional[np.ndarray] = None,
    *,
    min_score: float = 0.12,
    min_dist_px: int = 6,
) -> np.ndarray:
    """Instance map for P3: berry = nearest heatmap peak (full mask, not a circle).

    Branch / rigid / ego = connected components. Residual = leftover valid-depth
    is NOT encoded here (needs depth); call residual_mask() separately.
    """
    h, w = sem.shape[:2]
    inst = np.zeros((h, w), dtype=np.uint16)
    next_id = 1
    berry = sem == SEM_BERRY
    peaks = []
    if hm is not None and np.any(berry):
        from picking_perception.instance_gt import peaks_from_heatmap
        snap = max(8, int(0.03 * max(h, w)))
        snapped = []
        for x, y, s in peaks_from_heatmap(hm, min_score=min_score, min_dist_px=min_dist_px):
            yy = int(np.clip(y, 0, h - 1))
            xx = int(np.clip(x, 0, w - 1))
            if berry[yy, xx]:
                snapped.append((xx, yy, s))
                continue
            ys, xs = np.where(berry)
            if ys.size == 0:
                continue
            d2 = (xs - xx) ** 2 + (ys - yy) ** 2
            j = int(np.argmin(d2))
            if d2[j] <= snap ** 2:
                snapped.append((int(xs[j]), int(ys[j]), s))
        peaks = snapped
    if berry.any() and peaks:
        ys, xs = np.where(berry)
        pk = np.array([(p[0], p[1]) for p in peaks], dtype=np.float64)
        d2 = (xs[:, None] - pk[:, 0]) ** 2 + (ys[:, None] - pk[:, 1]) ** 2
        nearest = np.argmin(d2, axis=1)
        owned = np.zeros(len(ys), dtype=bool)
        for k in range(len(peaks)):
            sel = nearest == k
            if not np.any(sel):
                continue
            d = np.sqrt(d2[sel, k])
            r = float(np.median(d))
            own = d <= max(2.5 * r, float(min_dist_px))
            idx = np.where(sel)[0][own]
            inst[ys[idx], xs[idx]] = np.uint16(next_id + k)
            owned[idx] = True
        next_id = next_id + len(peaks)
        leftover = ~owned
        if np.any(leftover):
            leftover_mask = np.zeros_like(berry)
            leftover_mask[ys[leftover], xs[leftover]] = True
            for sel in _cc_ids(leftover_mask, MIN_PX[SEM_BERRY]):
                inst[sel] = np.uint16(next_id)
                next_id += 1
    elif berry.any():
        for sel in _cc_ids(berry, MIN_PX[SEM_BERRY]):
            inst[sel] = np.uint16(next_id)
            next_id += 1
    for cls in (SEM_BRANCH, SEM_RIGID, SEM_EGO):
        for sel in _cc_ids(sem == cls, MIN_PX[cls]):
            inst[sel] = np.uint16(next_id)
            next_id += 1
    return inst


def residual_mask(sem: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    if depth_m.shape[:2] != sem.shape[:2]:
        import cv2
        depth_m = cv2.resize(
            depth_m.astype(np.float32),
            (sem.shape[1], sem.shape[0]),
            interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(depth_m) & (depth_m >= DEPTH_MIN_M) & (depth_m <= DEPTH_MAX_M)
    resid = valid & (sem == SEM_BG)
    return resid & ~_residual_near_clip_mode(depth_m, resid)


def _residual_near_clip_mode(depth_m: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """True on residual pixels that are the near-range fill pile of this image.

    RGB-D often reports the sensor near clip on empty sky / missing returns.
    Detected when residual camera-Z is bimodal: a near mode well below the
    median of the same residual mask (relative split, not a workspace height).
    """
    z = np.asarray(depth_m, dtype=np.float32)[resid]
    if z.size < 64:
        return np.zeros(resid.shape, dtype=bool)
    med = float(np.median(z))
    p05 = float(np.percentile(z, 5))
    if not np.isfinite(med) or med <= 0.0 or p05 >= 0.5 * med:
        return np.zeros(resid.shape, dtype=bool)
    return resid & (depth_m < 0.5 * med)


def polygon_from_xy(
    xy: np.ndarray,
    *,
    cell_m: float,
    max_verts: int = MAX_POLY_VERTS,
) -> Optional[np.ndarray]:
    """Outer contour of occupied XY cells. Concave-safe (not convex hull)."""
    import cv2

    if xy.shape[0] < MIN_PTS_SLICE:
        return None
    cell = float(max(cell_m, 1e-4))
    pad = cell * 2.0
    xmin = float(xy[:, 0].min()) - pad
    ymin = float(xy[:, 1].min()) - pad
    xmax = float(xy[:, 0].max()) + pad
    ymax = float(xy[:, 1].max()) + pad
    w = int(np.ceil((xmax - xmin) / cell)) + 1
    h = int(np.ceil((ymax - ymin) / cell)) + 1
    if w < 2 or h < 2 or w * h > 400_000:
        cell = max(cell, np.sqrt(((xmax - xmin) * (ymax - ymin)) / 80_000.0))
        w = int(np.ceil((xmax - xmin) / cell)) + 1
        h = int(np.ceil((ymax - ymin) / cell)) + 1
    grid = np.zeros((h, w), dtype=np.uint8)
    us = np.clip(((xy[:, 0] - xmin) / cell).astype(np.int32), 0, w - 1)
    vs = np.clip(((xy[:, 1] - ymin) / cell).astype(np.int32), 0, h - 1)
    grid[vs, us] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def _bbox_poly() -> Optional[np.ndarray]:
        ys, xs = np.where(grid > 0)
        if xs.size == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        poly = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        poly[:, 0] = poly[:, 0] * cell + xmin
        poly[:, 1] = poly[:, 1] * cell + ymin
        return poly

    n_occ = int(np.count_nonzero(grid))
    if not cnts:
        return _bbox_poly() if n_occ >= 12 else None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 1.5:
        # Thin wall → bbox; 4-cell speckle must not become a solid rectangle.
        if n_occ >= 12:
            return _bbox_poly()
        return None
    peri = float(cv2.arcLength(c, True))
    eps = max(0.8, 0.01 * peri)
    approx = cv2.approxPolyDP(c, eps, True)
    poly = approx.reshape(-1, 2).astype(np.float64)
    if len(poly) > max_verts:
        idx = np.linspace(0, len(poly) - 1, max_verts, endpoint=False).astype(np.int32)
        poly = poly[idx]
    if len(poly) < 3:
        return _bbox_poly()
    poly[:, 0] = poly[:, 0] * cell + xmin
    poly[:, 1] = poly[:, 1] * cell + ymin
    return poly


def slices_from_points(
    pts: np.ndarray,
    *,
    class_id: int,
    dz: Optional[float] = None,
    cell_m: Optional[float] = None,
    max_slices: int = MAX_SLICES,
) -> List[ZSlice]:
    if pts.shape[0] < MIN_PTS_SLICE:
        return []
    z = pts[:, 2]
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    span = max(zmax - zmin, 1e-4)
    dz_use = float(dz if dz is not None else DZ_M.get(int(class_id), 0.008))
    n = int(np.ceil(span / dz_use))
    if n > max_slices:
        dz_use = span / float(max_slices)
        n = max_slices
    n = max(1, n)
    cell = float(cell_m if cell_m is not None else CELL_M.get(int(class_id), 0.005))
    layers: List[Tuple[float, float, np.ndarray]] = []
    for i in range(n):
        lo = zmin + i * dz_use
        hi = zmin + (i + 1) * dz_use
        if i == n - 1:
            hi = zmax + 1e-6
        sel = (z >= lo) & (z < hi)
        if i == n - 1:
            sel = (z >= lo) & (z <= hi + 1e-9)
        xy = pts[sel, :2]
        if xy.shape[0] < MIN_PTS_SLICE:
            continue
        layers.append((lo, hi, xy))
    if not layers:
        return []
    densest = max(x.shape[0] for _, _, x in layers)
    min_keep = max(MIN_PTS_SLICE, 0.02 * float(densest))
    out: List[ZSlice] = []
    for lo, hi, xy in layers:
        if xy.shape[0] < min_keep:
            continue
        poly = polygon_from_xy(xy, cell_m=cell)
        if poly is None:
            continue
        out.append(ZSlice(z_min=float(lo), z_max=float(hi), xy=poly))
    return out


def object_from_points(iid: int, class_id: int, pts: np.ndarray) -> Optional[SlicedObject]:
    if pts.shape[0] < MIN_PTS_SLICE:
        return None
    c = np.median(pts, axis=0)
    std = float(np.mean(np.std(pts, axis=0)))
    slices = slices_from_points(pts, class_id=class_id)
    if not slices:
        return None
    return SlicedObject(
        id=int(iid),
        class_id=int(class_id),
        pts_xyz=pts,
        slices=slices,
        centroid_xyz=(float(c[0]), float(c[1]), float(c[2])),
        n_pts=int(pts.shape[0]),
        xyz_std=std,
    )


def _nn_cell_m(pts: np.ndarray) -> float:
    """Voxel size from this cloud's own neighbor spacing (not a scene height)."""
    from scipy.spatial import cKDTree

    n = int(pts.shape[0])
    if n < 8:
        return 0.01
    step = max(1, n // 4000)
    sample = pts[::step]
    tree = cKDTree(sample)
    d, _ = tree.query(sample, k=2)
    med = float(np.median(d[:, 1]))
    return float(max(med * 2.5, 1e-4))


def split_spatial_clusters(pts: np.ndarray) -> List[np.ndarray]:
    """Split a backprojected cloud into 3D-connected components.

    2D mask CCs ignore depth discontinuities, so one instance id can contain
    two objects 0.5 m apart. Voxel CC uses this cloud's neighbor spacing.
    Sibling CCs smaller than 2% of the largest are dropped as flyers.
    """
    from scipy.ndimage import binary_dilation, label as nd_label

    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < MIN_PTS_SLICE:
        return []
    cell = _nn_cell_m(pts)
    origin = pts.min(axis=0)
    ijk = np.floor((pts - origin) / cell).astype(np.int32)
    ijk0 = ijk - ijk.min(axis=0)
    shape = tuple(int(x) + 1 for x in ijk0.max(axis=0))
    vol = int(np.prod(shape))
    if vol > 1_500_000:
        cell *= float(np.cbrt(vol / 400_000.0))
        ijk = np.floor((pts - origin) / cell).astype(np.int32)
        ijk0 = ijk - ijk.min(axis=0)
        shape = tuple(int(x) + 1 for x in ijk0.max(axis=0))
    grid = np.zeros(shape, dtype=np.uint8)
    grid[tuple(ijk0.T)] = 1
    grid = binary_dilation(grid, iterations=1).astype(np.uint8)
    lab, ncc = nd_label(grid)
    if ncc <= 1:
        return [pts]
    labels_of_pts = lab[tuple(ijk0.T)]
    counts = [(int(np.sum(labels_of_pts == i)), i) for i in range(1, ncc + 1)]
    counts.sort(reverse=True)
    min_keep = max(MIN_PTS_SLICE, int(0.02 * counts[0][0]))
    out: List[np.ndarray] = []
    for cnt, i in counts:
        if cnt < min_keep:
            continue
        out.append(pts[labels_of_pts == i])
    return out if out else [pts]


def objects_from_points(iid: int, class_id: int, pts: np.ndarray) -> List[SlicedObject]:
    out: List[SlicedObject] = []
    for k, cluster in enumerate(split_spatial_clusters(pts)):
        obj = object_from_points(int(iid) + k, class_id, cluster)
        if obj is not None:
            out.append(obj)
    return out


def lift_scene(
    sem: np.ndarray,
    inst: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    *,
    include_residual: bool = True,
    max_objects: int = 80,
    source: str = 'fixed',
) -> List[SlicedObject]:
    """Lift each instance (+ optional residual CCs) to z-sliced objects."""
    if depth_m.shape[:2] != sem.shape[:2]:
        import cv2
        depth_m = cv2.resize(
            depth_m.astype(np.float32),
            (sem.shape[1], sem.shape[0]),
            interpolation=cv2.INTER_NEAREST)
    out: List[SlicedObject] = []
    emit_id = 1
    for iid in np.unique(inst):
        iid = int(iid)
        if iid <= 0:
            continue
        sel = inst == iid
        hits = sem[sel]
        if hits.size == 0:
            continue
        cls = int(np.bincount(hits.astype(np.int64)).argmax())
        if cls not in (SEM_BERRY, SEM_BRANCH, SEM_RIGID, SEM_EGO):
            continue
        if int(np.sum(sel)) < MIN_PX.get(cls, 8):
            continue
        pts = backproject_mask(
            depth_m, K, T_base_cam, sel,
            stride=BACKPROJECT_STRIDE.get(cls, 2))
        for obj in objects_from_points(emit_id, cls, pts):
            obj.source = str(source)
            obj.id = emit_id
            emit_id += 1
            out.append(obj)
            if len(out) >= max_objects:
                return _keep_relative_mass(out)
    if include_residual:
        rmask = residual_mask(sem, depth_m)
        for sel in _cc_ids(rmask, MIN_PX[SEM_RESIDUAL]):
            pts = backproject_mask(
                depth_m, K, T_base_cam, sel,
                stride=BACKPROJECT_STRIDE[SEM_RESIDUAL])
            for obj in objects_from_points(emit_id, SEM_RESIDUAL, pts):
                obj.source = str(source)
                obj.id = emit_id
                emit_id += 1
                out.append(obj)
                if len(out) >= max_objects:
                    return _keep_relative_mass(out)
    return _keep_relative_mass(out)


def _keep_relative_mass(objs: List[SlicedObject]) -> List[SlicedObject]:
    """Drop speckle objects that are <2% of the largest same-class cloud this frame."""
    if not objs:
        return objs
    by: dict = {}
    for o in objs:
        by.setdefault(int(o.class_id), []).append(o)
    kept: List[SlicedObject] = []
    for group in by.values():
        biggest = max(int(o.n_pts) for o in group)
        thr = max(MIN_PTS_SLICE, int(0.02 * biggest))
        for o in group:
            if int(o.n_pts) >= thr:
                kept.append(o)
    return kept


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Even-odd test. poly (N,2)."""
    n = int(poly.shape[0])
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def point_hits_object(xyz: Sequence[float], obj: SlicedObject, *, margin_xy: float = 0.0) -> bool:
    p = np.asarray(xyz, dtype=np.float64)
    for sl in obj.slices:
        if p[2] < sl.z_min or p[2] > sl.z_max:
            continue
        if margin_xy <= 0.0:
            if point_in_polygon(float(p[0]), float(p[1]), sl.xy):
                return True
            continue
        # shrink/expand via simple vertex offset is skip; test raw + nearby verts
        if point_in_polygon(float(p[0]), float(p[1]), sl.xy):
            return True
    return False


def fill_cells(poly: np.ndarray, cell_m: float) -> Tuple[np.ndarray, float, float, float]:
    """Rasterize polygon → occupied cell centres (M,2) plus origin/cell."""
    import cv2

    cell = float(max(cell_m, 1e-4))
    xmin = float(poly[:, 0].min())
    ymin = float(poly[:, 1].min())
    xmax = float(poly[:, 0].max())
    ymax = float(poly[:, 1].max())
    w = max(2, int(np.ceil((xmax - xmin) / cell)) + 1)
    h = int(np.ceil((ymax - ymin) / cell)) + 1
    if w * h > 120_000:
        cell = max(cell, np.sqrt(((xmax - xmin) * (ymax - ymin) + 1e-9) / 80_000.0))
        w = max(2, int(np.ceil((xmax - xmin) / cell)) + 1)
        h = int(np.ceil((ymax - ymin) / cell)) + 1
    grid = np.zeros((h, w), dtype=np.uint8)
    pts = np.round((poly - np.array([xmin, ymin])) / cell).astype(np.int32)
    cv2.fillPoly(grid, [pts], 1)
    vs, us = np.where(grid > 0)
    if vs.size == 0:
        return np.zeros((0, 2), dtype=np.float64), xmin, ymin, cell
    centres = np.stack([xmin + (us + 0.5) * cell, ymin + (vs + 0.5) * cell], axis=1)
    return centres, xmin, ymin, cell


def _append_tri(tris: List[np.ndarray], a, b, c) -> None:
    tris.append(np.asarray(a, dtype=np.float64))
    tris.append(np.asarray(b, dtype=np.float64))
    tris.append(np.asarray(c, dtype=np.float64))


def prism_triangles(obj: SlicedObject, *, max_cells: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    """Triangle vertices (3k, 3) and outline segments (2m, 3) for one object."""
    cell = CELL_M.get(int(obj.class_id), 0.005)
    tris: List[np.ndarray] = []
    edges: List[np.ndarray] = []
    for sl in obj.slices:
        z0, z1 = float(sl.z_min), float(sl.z_max)
        poly = sl.xy
        n = int(poly.shape[0])
        if n < 3:
            continue
        for i in range(n):
            j = (i + 1) % n
            p0 = np.array([poly[i, 0], poly[i, 1], z0])
            p1 = np.array([poly[j, 0], poly[j, 1], z0])
            q0 = np.array([poly[i, 0], poly[i, 1], z1])
            q1 = np.array([poly[j, 0], poly[j, 1], z1])
            _append_tri(tris, p0, p1, q1)
            _append_tri(tris, p0, q1, q0)
            edges.append(q0)
            edges.append(q1)
        centres, _, _, cell_use = fill_cells(poly, cell)
        if centres.shape[0] > max_cells:
            step = int(np.ceil(centres.shape[0] / max_cells))
            centres = centres[::step]
        hx = 0.5 * cell_use
        for cx, cy in centres:
            a = np.array([cx - hx, cy - hx, z1])
            b = np.array([cx + hx, cy - hx, z1])
            c = np.array([cx + hx, cy + hx, z1])
            d = np.array([cx - hx, cy + hx, z1])
            _append_tri(tris, a, b, c)
            _append_tri(tris, a, c, d)
            a0 = np.array([cx - hx, cy - hx, z0])
            b0 = np.array([cx + hx, cy - hx, z0])
            c0 = np.array([cx + hx, cy + hx, z0])
            d0 = np.array([cx - hx, cy + hx, z0])
            _append_tri(tris, a0, c0, b0)
            _append_tri(tris, a0, d0, c0)
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.stack(tris, axis=0), np.stack(edges, axis=0) if edges else np.zeros((0, 3))


def _xyz(obj: SlicedObject) -> np.ndarray:
    return np.asarray(obj.centroid_xyz, dtype=np.float64)


def berries_of(objs: Sequence[SlicedObject]) -> List[SlicedObject]:
    return [o for o in objs if int(o.class_id) == SEM_BERRY]


def pick_active_berry(
    berries: Sequence[SlicedObject],
    *,
    fruit_id: int = -1,
    look_xyz: Optional[Sequence[float]] = None,
) -> Optional[SlicedObject]:
    if not berries:
        return None
    if int(fruit_id) >= 0:
        for b in berries:
            if int(b.id) == int(fruit_id):
                return b
    if look_xyz is None:
        return berries[0]
    look = np.asarray(look_xyz, dtype=np.float64)
    return min(berries, key=lambda b: float(np.linalg.norm(_xyz(b) - look)))


def associate_wrist_berries(
    fixed: Sequence[SlicedObject],
    wrist: Sequence[SlicedObject],
) -> List[Tuple[SlicedObject, SlicedObject, float]]:
    """P4/T1 only: greedy nearest pairs if the two clouds overlap at their own scale.

    Not a mapping filter. Do not pass a scene-scale centimetre gate.
    """
    fb = berries_of(fixed)
    wb = berries_of(wrist)
    used = set()
    pairs: List[Tuple[SlicedObject, SlicedObject, float]] = []
    order = sorted(wb, key=lambda b: int(b.n_pts), reverse=True)
    for w in order:
        best = None
        best_d = 1e9
        for f in fb:
            if int(f.id) in used:
                continue
            d = float(np.linalg.norm(_xyz(w) - _xyz(f)))
            if d < best_d:
                best_d = d
                best = f
        if best is None or not detections_overlap(best, w):
            continue
        used.add(int(best.id))
        pairs.append((best, w, best_d))
    return pairs


def _clone_obj(o: SlicedObject) -> SlicedObject:
    return SlicedObject(
        id=o.id, class_id=o.class_id, pts_xyz=o.pts_xyz, slices=o.slices,
        centroid_xyz=o.centroid_xyz, n_pts=o.n_pts, xyz_std=o.xyz_std,
        source=o.source or 'fixed', visible_wrist=False,
        centroid_fixed_xyz=None, fuse_delta_m=-1.0, pick_role=0,
    )


def _aabb(obj: SlicedObject) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(obj.pts_xyz)
    if pts.size == 0:
        c = _xyz(obj)
        return c, c
    return pts.min(axis=0), pts.max(axis=0)


def aabb_overlap(a: SlicedObject, b: SlicedObject, pad_m: float = 0.0) -> bool:
    amin, amax = _aabb(a)
    bmin, bmax = _aabb(b)
    return bool(np.all(amin - pad_m <= bmax) and np.all(bmin - pad_m <= amax))


def _r_xy(obj: SlicedObject) -> float:
    """Robust in-plane radius from the object's own points (not a scene-scale constant)."""
    p = np.asarray(obj.pts_xyz)
    if p.size == 0:
        return float(CELL_M.get(int(obj.class_id), 0.005))
    xy = p[:, :2]
    c = np.median(xy, axis=0)
    r = float(np.median(np.linalg.norm(xy - c, axis=1)))
    return max(r, float(CELL_M.get(int(obj.class_id), 0.005)))


def detections_overlap(a: SlicedObject, b: SlicedObject) -> bool:
    """Same class and spatially the same object: compact → radius test; extended → AABB hit."""
    if int(a.class_id) != int(b.class_id):
        return False
    if int(a.class_id) == SEM_BERRY:
        return float(np.linalg.norm(_xyz(a) - _xyz(b))) <= (_r_xy(a) + _r_xy(b))
    return aabb_overlap(a, b, pad_m=0.0)


def _union_into(dst: SlicedObject, w: SlicedObject) -> None:
    if dst.centroid_fixed_xyz is None:
        dst.centroid_fixed_xyz = tuple(float(v) for v in dst.centroid_xyz)
    pts = np.concatenate(
        [np.asarray(dst.pts_xyz, dtype=np.float64),
         np.asarray(w.pts_xyz, dtype=np.float64)], axis=0)
    if pts.shape[0] > 30000:
        pts = pts[:: int(np.ceil(pts.shape[0] / 30000.0))]
    rebuilt = object_from_points(dst.id, dst.class_id, pts)
    if rebuilt is None:
        dst.pts_xyz = pts
        dst.n_pts = int(pts.shape[0])
        c = np.median(pts, axis=0)
        dst.centroid_xyz = (float(c[0]), float(c[1]), float(c[2]))
        if w.slices and not dst.slices:
            dst.slices = w.slices
    else:
        dst.pts_xyz = rebuilt.pts_xyz
        dst.slices = rebuilt.slices
        dst.n_pts = rebuilt.n_pts
        dst.xyz_std = rebuilt.xyz_std
        dst.centroid_xyz = rebuilt.centroid_xyz
    dst.source = 'fused'
    dst.visible_wrist = True
    dst.fuse_delta_m = float(np.linalg.norm(_xyz(dst) - np.asarray(dst.centroid_fixed_xyz)))


def ingest_wrist_detections(
    fused: List[SlicedObject],
    wrist: Sequence[SlicedObject],
) -> Tuple[int, int]:
    """Keep every wrist detection. Union into a same-class overlap; else append."""
    n_merged = 0
    n_extra = 0
    next_id = max((o.id for o in fused), default=0) + 1
    for w in sorted(wrist, key=lambda o: int(o.n_pts), reverse=True):
        if int(w.class_id) == SEM_EGO:
            # ego remains FK for planning; still show overlapping arm pixels if already lifted
            hits = [d for d in fused if detections_overlap(d, w)]
            if hits:
                _union_into(max(hits, key=lambda d: int(d.n_pts)), w)
                n_merged += 1
            continue
        hits = [d for d in fused if detections_overlap(d, w)]
        if hits:
            _union_into(max(hits, key=lambda d: int(d.n_pts)), w)
            n_merged += 1
            continue
        extra = _clone_obj(w)
        extra.id = next_id
        extra.source = 'wrist'
        extra.visible_wrist = True
        fused.append(extra)
        next_id += 1
        n_extra += 1
    return n_merged, n_extra


def fuse_world(
    fixed: Sequence[SlicedObject],
    wrist: Sequence[SlicedObject],
    *,
    fruit_id: int = -1,
    look_xyz: Optional[Sequence[float]] = None,
) -> Tuple[List[SlicedObject], FuseReport]:
    """Both cameras in; overlap → union; non-overlap → keep both. fruit_id only labels ACTIVE."""
    fused = [_clone_obj(o) for o in fixed]
    report = FuseReport(n_wrist_berries=len(berries_of(wrist)))
    fb0, wb = berries_of(fused), berries_of(wrist)
    if fb0 and wb:
        report.min_pair_m = min(
            float(np.linalg.norm(_xyz(f) - _xyz(w))) for f in fb0 for w in wb)
    report.n_merged_struct, report.n_wrist_extra = ingest_wrist_detections(fused, wrist)
    report.n_replaced = report.n_merged_struct
    active = pick_active_berry(
        berries_of(fused), fruit_id=fruit_id, look_xyz=look_xyz)
    if active is None:
        report.reject_reason = 'no_berry'
        return fused, report
    report.active_id = int(active.id)
    active.pick_role = 1
    if active.visible_wrist:
        report.associated = True
        report.dist_m = float(active.fuse_delta_m)
        if active.centroid_fixed_xyz is not None:
            report.delta_xyz = tuple(
                float(v) for v in (_xyz(active) - np.asarray(active.centroid_fixed_xyz)))
    elif wb:
        nearest = min(wb, key=lambda b: float(np.linalg.norm(_xyz(b) - _xyz(active))))
        dist = float(np.linalg.norm(_xyz(nearest) - _xyz(active)))
        report.dist_m = dist
        report.delta_xyz = tuple(float(v) for v in (_xyz(nearest) - _xyz(active)))
        report.reject_reason = 'active_not_in_wrist'
    else:
        report.reject_reason = 'no_wrist_berry'
    return fused, report
