"""Berry instance GT + decode: semantic gate (cluster) + center heatmap → circles.

Circles live only inside berry-class pixels. Do not watershed the whole cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

SEM_BERRY = 1


def center_radius_from_mask(mask: np.ndarray):
    """Centroid + equivalent-disk radius from a tight instance mask (not a detector)."""
    ys, xs = np.where(mask)
    if ys.size < 4:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    r = float(np.sqrt(float(ys.size) / np.pi))
    return cx, cy, r

# Image-space berry radius when depth is unavailable (wrist ~15–40px, far fixed smaller).
BERRY_R_MIN_PX = 5
BERRY_R_MAX_FRAC = 0.045  # of max(H, W)


@dataclass
class BerryCenter:
    x: float
    y: float
    sigma: float
    area_px: int


@dataclass
class BerryCircle:
    id: int
    x: float
    y: float
    r: float
    score: float


def parse_yolo_seg_line(line: str, h: int, w: int):
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    cid = int(float(parts[0]))
    xy = np.array([float(v) for v in parts[1:]], dtype=np.float64)
    if xy.size < 6 or xy.size % 2:
        return None
    pts = xy.reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return cid, pts.astype(np.int32)


def maps_from_yolo_seg(txt: Path, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, List[BerryCenter]]:
    """Semantic uint8 + berry instance uint16 + centers. One txt line = one instance."""
    import cv2

    sem = np.zeros((h, w), dtype=np.uint8)
    inst = np.zeros((h, w), dtype=np.uint16)
    centers: List[BerryCenter] = []
    berry_id = 1
    if not txt.is_file():
        return sem, inst, centers
    for line in txt.read_text(encoding='utf-8').splitlines():
        parsed = parse_yolo_seg_line(line, h, w)
        if parsed is None:
            continue
        cid, pts = parsed
        pix = int(cid) + 1
        if pix < 1 or pix > 4:
            continue
        cv2.fillPoly(sem, [pts], pix)
        if pix != SEM_BERRY:
            continue
        layer = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(layer, [pts], 1)
        area = int(np.sum(layer))
        if area < 4:
            continue
        inst[layer > 0] = np.uint16(berry_id)
        ys, xs = np.where(layer > 0)
        cx = float(xs.mean())
        cy = float(ys.mean())
        sigma = float(np.clip(0.45 * np.sqrt(area / np.pi), 2.0, 20.0))
        centers.append(BerryCenter(x=cx, y=cy, sigma=sigma, area_px=area))
        berry_id += 1
    return sem, inst, centers


def centers_from_inst(inst: np.ndarray) -> List[BerryCenter]:
    """One id in *_inst.png → one fruit center."""
    centers: List[BerryCenter] = []
    if inst is None:
        return centers
    for iid in np.unique(inst):
        iid = int(iid)
        if iid <= 0:
            continue
        ys, xs = np.where(inst == iid)
        area = int(ys.size)
        if area < 4:
            continue
        sigma = float(np.clip(0.45 * np.sqrt(area / np.pi), 2.0, 20.0))
        centers.append(BerryCenter(
            x=float(xs.mean()), y=float(ys.mean()), sigma=sigma, area_px=area))
    return centers


def centers_from_yolo_det(
    txt: Path, h: int, w: int, *, berry_cls: Sequence[int] = (0,),
    max_area_frac: float = 0.03,
) -> List[BerryCenter]:
    """YOLO detect boxes → fruit centers only. Skip cluster-sized boxes. No semantic fill."""
    centers: List[BerryCenter] = []
    if not txt.is_file():
        return centers
    allow = {int(c) for c in berry_cls}
    max_area = float(max_area_frac) * float(h * w)
    for line in Path(txt).read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = int(float(parts[0]))
        if cid not in allow:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        px = cx * w
        py = cy * h
        pw = bw * w
        ph = bh * h
        if pw < 2 or ph < 2 or pw * ph > max_area:
            continue
        area = max(1, int(pw * ph))
        sigma = float(np.clip(0.45 * 0.25 * (pw + ph), 2.0, 20.0))
        centers.append(BerryCenter(x=px, y=py, sigma=sigma, area_px=area))
    return centers


def heatmap_from_centers(
    h: int, w: int, centers: Sequence[BerryCenter],
) -> np.ndarray:
    hm = np.zeros((h, w), dtype=np.float32)
    if not centers:
        return hm
    yy, xx = np.ogrid[:h, :w]
    for c in centers:
        s2 = 2.0 * float(c.sigma) ** 2
        g = np.exp(-((xx - c.x) ** 2 + (yy - c.y) ** 2) / max(s2, 1e-6))
        np.maximum(hm, g, out=hm)
    return hm


def resize_centers(centers: Sequence[BerryCenter], *, src_hw, dst_hw) -> List[BerryCenter]:
    sh, sw = src_hw
    dh, dw = dst_hw
    sx = dw / max(sw, 1)
    sy = dh / max(sh, 1)
    ss = 0.5 * (sx + sy)
    return [
        BerryCenter(
            x=c.x * sx, y=c.y * sy,
            sigma=float(np.clip(c.sigma * ss, 2.0, 24.0)),
            area_px=max(1, int(c.area_px * sx * sy)),
        )
        for c in centers
    ]


def peaks_from_heatmap(
    hm: np.ndarray,
    *,
    min_score: float = 0.12,
    min_dist_px: int = 6,
) -> List[Tuple[int, int, float]]:
    from scipy.ndimage import maximum_filter

    if hm.size == 0:
        return []
    k = int(max(3, min_dist_px))
    if k % 2 == 0:
        k += 1
    mx = maximum_filter(hm, size=k)
    thr = max(float(min_score), 0.40 * float(np.max(hm)))
    peaks = (hm >= mx - 1e-6) & (hm >= thr)
    ys, xs = np.where(peaks)
    scored = [(int(x), int(y), float(hm[y, x])) for y, x in zip(ys, xs)]
    scored.sort(key=lambda t: -t[2])
    kept: List[Tuple[int, int, float]] = []
    for x, y, s in scored:
        if any((x - px) ** 2 + (y - py) ** 2 < min_dist_px ** 2 for px, py, _ in kept):
            continue
        kept.append((x, y, s))
    return kept


def _snap_to_berry(
    x: int, y: int, berry: np.ndarray, *, max_dist_px: int,
) -> Optional[Tuple[int, int]]:
    h, w = berry.shape[:2]
    if 0 <= y < h and 0 <= x < w and berry[y, x]:
        return int(x), int(y)
    ys, xs = np.where(berry)
    if ys.size == 0:
        return None
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    j = int(np.argmin(d2))
    if d2[j] > int(max_dist_px) ** 2:
        return None
    return int(xs[j]), int(ys[j])


def radius_from_heatmap(
    hm: np.ndarray,
    x: int,
    y: int,
    berry: np.ndarray,
    *,
    min_r: float = BERRY_R_MIN_PX,
    max_r: Optional[float] = None,
) -> float:
    """Half-height of the peak along 16 rays, stopped at cluster edge."""
    h, w = hm.shape[:2]
    if max_r is None:
        max_r = max(min_r, BERRY_R_MAX_FRAC * max(h, w))
    peak = float(hm[y, x]) if (0 <= y < h and 0 <= x < w) else 0.0
    if peak <= 1e-6:
        return float(min_r)
    thr = 0.45 * peak
    rs: List[float] = []
    for ang in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        hit = float(max_r)
        for t in range(1, int(max_r) + 1):
            xx = int(round(x + t * ca))
            yy = int(round(y + t * sa))
            if not (0 <= xx < w and 0 <= yy < h) or (not berry[yy, xx]) or hm[yy, xx] < thr:
                hit = float(t)
                break
        rs.append(hit)
    return float(np.clip(float(np.median(rs)), min_r, max_r))


def circles_from_heatmap(
    sem: np.ndarray,
    hm: np.ndarray,
    *,
    min_score: float = 0.12,
    min_dist_px: int = 6,
) -> List[BerryCircle]:
    """Peaks inside the berry semantic gate → one circle per fruit. No YOLO."""
    berry = sem == SEM_BERRY
    if not np.any(berry):
        return []
    h, w = sem.shape[:2]
    max_r = max(BERRY_R_MIN_PX, BERRY_R_MAX_FRAC * max(h, w))
    snap = max(8, int(0.02 * max(h, w)))
    peaks = peaks_from_heatmap(hm, min_score=min_score, min_dist_px=min_dist_px)
    out: List[BerryCircle] = []
    iid = 1
    for x, y, score in peaks:
        snapped = _snap_to_berry(x, y, berry, max_dist_px=snap)
        if snapped is None:
            continue
        x, y = snapped
        r = radius_from_heatmap(hm, x, y, berry, min_r=BERRY_R_MIN_PX, max_r=max_r)
        out.append(BerryCircle(id=iid, x=float(x), y=float(y), r=r, score=float(score)))
        iid += 1
    return out


def instances_from_circles(
    sem: np.ndarray,
    circles: Sequence[BerryCircle],
) -> np.ndarray:
    """Berry id only inside (circle ∩ berry). Leftover cluster pixels stay 0 (not a fruit)."""
    from picking_perception.berry_instances import BERRY_MIN_PX, SEM_BRANCH, SEM_RIGID
    from scipy.ndimage import label as nd_label

    h, w = sem.shape[:2]
    inst = np.zeros((h, w), dtype=np.uint16)
    berry = sem == SEM_BERRY
    if circles:
        yy, xx = np.ogrid[:h, :w]
        d2 = np.full((h, w), np.inf, dtype=np.float64)
        nearest = np.zeros((h, w), dtype=np.int32)
        inside = np.zeros((h, w), dtype=bool)
        for i, c in enumerate(circles, start=1):
            dc = (xx - c.x) ** 2 + (yy - c.y) ** 2
            hit = berry & (dc <= (c.r * c.r))
            closer = hit & (dc < d2)
            d2[closer] = dc[closer]
            nearest[closer] = i
            inside |= hit
        inst[inside] = nearest[inside].astype(np.uint16)
    next_id = int(inst.max()) + 1 if inst.size else 1
    for cls in (SEM_BRANCH, SEM_RIGID):
        cc, ncc = nd_label(sem == cls)
        for i in range(1, ncc + 1):
            if int(np.sum(cc == i)) < BERRY_MIN_PX:
                continue
            inst[cc == i] = np.uint16(next_id)
            next_id += 1
    return inst


def instances_from_heatmap(
    sem: np.ndarray,
    hm: np.ndarray,
    *,
    min_score: float = 0.12,
    min_dist_px: int = 6,
) -> np.ndarray:
    """Berry instances = circles gated by semantic cluster. No watershed on the blob."""
    circles = circles_from_heatmap(
        sem, hm, min_score=min_score, min_dist_px=min_dist_px)
    return instances_from_circles(sem, circles)


def overlay_berry_circles(
    bgr_or_rgb: np.ndarray,
    circles: Sequence[BerryCircle],
    *,
    rgb: bool = True,
) -> np.ndarray:
    import cv2

    vis = bgr_or_rgb.copy()
    color = (255, 0, 220) if rgb else (220, 0, 255)
    center_col = (0, 255, 255) if not rgb else (255, 255, 0)
    for c in circles:
        cx, cy, r = int(round(c.x)), int(round(c.y)), int(max(2, round(c.r)))
        cv2.circle(vis, (cx, cy), r, color, 2, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 3, center_col, -1, cv2.LINE_AA)
        cv2.putText(
            vis, str(c.id), (cx + 4, max(12, cy - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def contours_from_circles(circles: Sequence[BerryCircle], *, n_verts: int = 16):
    from picking_perception.berry_instances import InstanceContour, SEM_BERRY

    out = []
    for c in circles:
        angs = np.linspace(0.0, 2.0 * np.pi, int(n_verts), endpoint=False)
        poly = np.stack(
            [c.x + c.r * np.cos(angs), c.y + c.r * np.sin(angs)], axis=1).astype(np.float32)
        xs, ys = poly[:, 0], poly[:, 1]
        bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        out.append(InstanceContour(
            id=int(c.id), class_id=SEM_BERRY, polygon_uv=poly, bbox_uv=bbox,
            area_px=int(max(1, round(np.pi * c.r * c.r)))))
    return out


def write_instance_png(path: Path, inst: np.ndarray) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), inst.astype(np.uint16))
