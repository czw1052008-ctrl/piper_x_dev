"""2D berry lock + depth fusion for close-range approach (FP optional)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from picking_perception.yolo_berry_detector import YoloDetection


@dataclass
class TrackedBerry:
    pose_cam: np.ndarray
    confidence: float
    mode: str  # fp | mono | coast
    uv: Tuple[int, int]
    z_m: float
    bbox: Tuple[int, int, int, int]


@dataclass
class _LockState:
  track_id: int
  last_uv: Tuple[float, float]
  last_z: float
  last_bbox: Tuple[int, int, int, int]
  lost_frames: int = 0
  last_mode: str = 'mono'


def _mask_center(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
      h, w = mask.shape[:2]
      return w // 2, h // 2
    return int(xs.mean()), int(ys.mean())


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax1 - ax0, 0) * max(ay1 - ay0, 0)
    area_b = max(bx1 - bx0, 0) * max(by1 - by0, 0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _pose_from_xyz(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


class BerryTracker:
    """Maintain a locked berry across frames when depth / FP fail up close."""

    def __init__(
        self,
        *,
        berry_diameter_m: float = 0.015,
        depth_min_m: float = 0.03,
        depth_max_m: float = 1.8,
        min_valid_depth_px: int = 15,
        track_lost_max_frames: int = 25,
        lock_match_iou: float = 0.08,
        lock_match_px: float = 80.0,
        fp_retry_interval: int = 8,
    ) -> None:
        self.berry_diameter_m = berry_diameter_m
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.min_valid_depth_px = min_valid_depth_px
        self.track_lost_max_frames = track_lost_max_frames
        self.lock_match_iou = lock_match_iou
        self.lock_match_px = lock_match_px
        self.fp_retry_interval = fp_retry_interval
        self._lock: Optional[_LockState] = None
        self._next_track_id = 1
        self._frames_since_fp = 0

    def reset_lock(self) -> None:
        self._lock = None
        self._frames_since_fp = 0

    @property
    def locked_index(self) -> int:
        return -1

    def depth_in_mask(self, depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
        vals = depth[mask > 0]
        valid = vals[(vals >= self.depth_min_m) & (vals <= self.depth_max_m)]
        if valid.size < self.min_valid_depth_px:
            return None
        return float(np.median(valid))

    def mono_depth_from_mask(self, mask: np.ndarray, k: np.ndarray) -> float:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return 0.35
        area = float(xs.size)
        r_px = max(math.sqrt(area / math.pi), 3.0)
        fx = float(k[0, 0])
        z = fx * (self.berry_diameter_m * 0.5) / r_px
        return float(np.clip(z, self.depth_min_m, self.depth_max_m))

    def mono_depth_near_uv(
        self,
        mask: np.ndarray,
        u: float,
        v: float,
        k: np.ndarray,
        *,
        radius_px: Optional[float] = None,
    ) -> float:
        """Mono depth from mask pixels near a reference UV (surface-ward subset)."""
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return self.mono_depth_from_mask(mask, k)
        dists = np.hypot(xs.astype(np.float64) - float(u), ys.astype(np.float64) - float(v))
        if radius_px is None:
            radius_px = max(25.0, math.sqrt(float(xs.size) / math.pi) * 0.6)
        near = dists <= float(radius_px)
        if not near.any():
            return self.mono_depth_from_mask(mask, k)
        sub = np.zeros_like(mask)
        sub[ys[near], xs[near]] = mask[ys[near], xs[near]]
        return self.mono_depth_from_mask(sub, k)

    def uv_to_cam_xyz(
        self, u: int, v: int, z: float, k: np.ndarray,
    ) -> Tuple[float, float, float]:
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return x, y, z

    def _match_lock_index(self, dets: List[YoloDetection]) -> int:
        if self._lock is None or not dets:
            return -1
        lu, lv = self._lock.last_uv
        px_thresh = self.lock_match_px * (1.0 + 0.35 * max(self._lock.lost_frames, 0))
        iou_thresh = max(0.02, self.lock_match_iou * 0.5)
        best_i, best_score = -1, -1.0
        for i, det in enumerate(dets):
            iou = _bbox_iou(det.bbox_xyxy, self._lock.last_bbox)
            u, v = _mask_center(det.mask)
            dist = math.hypot(u - lu, v - lv)
            dist_score = max(0.0, 1.0 - dist / px_thresh)
            score = max(iou, dist_score)
            if score > best_score:
                best_score = score
                best_i = i
        if best_score < iou_thresh and best_i >= 0:
            u, v = _mask_center(dets[best_i].mask)
            if math.hypot(u - lu, v - lv) > px_thresh:
                best_i = -1
        if best_i < 0:
            return -1
        if best_score < self.lock_match_iou and best_score < 0.15:
            return -1
        return best_i

    def _coast_tracked(self, k: np.ndarray) -> TrackedBerry:
        assert self._lock is not None
        u = int(round(self._lock.last_uv[0]))
        v = int(round(self._lock.last_uv[1]))
        z = self._lock.last_z
        x, y, zz = self.uv_to_cam_xyz(u, v, z, k)
        return TrackedBerry(
            pose_cam=_pose_from_xyz(x, y, zz),
            confidence=0.35,
            mode='coast',
            uv=(u, v),
            z_m=float(zz),
            bbox=self._lock.last_bbox,
        )

    def _track_one(
        self,
        det: YoloDetection,
        depth: np.ndarray,
        k: np.ndarray,
        fp_wrapper,
        rgb: np.ndarray,
        try_fp: bool,
    ) -> TrackedBerry:
        u, v = _mask_center(det.mask)
        z_depth = self.depth_in_mask(depth, det.mask)
        z_mono = self.mono_depth_from_mask(det.mask, k)
        z_prior = self._lock.last_z if self._lock is not None else z_mono

        mode = 'mono'
        pose = None
        conf = float(det.confidence)

        if try_fp and z_depth is not None:
            try:
                fp_list = fp_wrapper.detect_all(rgb, depth, k, det.mask)
            except Exception:
                fp_list = []
            if fp_list:
                pose_fp, score = fp_list[0]
                z_fp = float(pose_fp[2, 3])
                if z_fp >= self.depth_min_m:
                    pose = pose_fp
                    mode = 'fp'
                    conf = float(score) * det.confidence
                    z_mono = z_fp

        if pose is None:
            if z_depth is not None:
                z = 0.65 * z_depth + 0.35 * z_mono
            elif self._lock is not None:
                # Close range: grow berry in image → z shrinks; trust mono over stale prior
                z = 0.8 * z_mono + 0.2 * z_prior
                mode = 'coast' if self._lock.lost_frames > 0 else 'mono'
            else:
                z = z_mono
            x, y, zz = self.uv_to_cam_xyz(u, v, z, k)
            pose = _pose_from_xyz(x, y, zz)

        return TrackedBerry(
            pose_cam=pose,
            confidence=conf,
            mode=mode,
            uv=(u, v),
            z_m=float(pose[2, 3]),
            bbox=det.bbox_xyxy,
        )

    def process(
        self,
        yolo_dets: List[YoloDetection],
        rgb: np.ndarray,
        depth: np.ndarray,
        k: np.ndarray,
        fp_wrapper,
        *,
        reset_lock: bool = False,
        force_lock_idx: Optional[int] = None,
    ) -> Tuple[List[TrackedBerry], int, str]:
        if reset_lock:
            self.reset_lock()

        if not yolo_dets:
            if self._lock is not None:
                self._lock.lost_frames += 1
                if self._lock.lost_frames > self.track_lost_max_frames:
                    self.reset_lock()
                    return [], -1, 'TRACK lost (no YOLO detections)'
                # Keep publishing coast pose so FSM can continue tracking one fruit.
                coast = self._coast_tracked(k)
                return [coast], 0, (
                    f'TRACK coast no-yolo lost={self._lock.lost_frames}/'
                    f'{self.track_lost_max_frames} id={self._lock.track_id}')
            return [], -1, 'YOLO found no blueberries'

        self._frames_since_fp += 1
        try_fp = self._frames_since_fp >= self.fp_retry_interval

        berries: List[TrackedBerry] = []
        for det in yolo_dets:
            berries.append(self._track_one(det, depth, k, fp_wrapper, rgb, try_fp=try_fp))

        lock_idx = self._match_lock_index(yolo_dets)
        if self._lock is None:
            if (
                force_lock_idx is not None
                and 0 <= int(force_lock_idx) < len(berries)
            ):
                lock_idx = int(force_lock_idx)
            else:
                # Prefer lowest in image (largest v): user picks bottom berry in view.
                viable = [i for i, b in enumerate(berries) if b.confidence >= 0.15]
                pool = viable if viable else list(range(len(berries)))
                lock_idx = int(max(pool, key=lambda i: berries[i].uv[1]))
            det = yolo_dets[lock_idx]
            self._lock = _LockState(
                track_id=self._next_track_id,
                last_uv=(float(berries[lock_idx].uv[0]), float(berries[lock_idx].uv[1])),
                last_z=berries[lock_idx].z_m,
                last_bbox=det.bbox_xyxy,
                lost_frames=0,
                last_mode=berries[lock_idx].mode,
            )
            self._next_track_id += 1
            self._frames_since_fp = 0
            mode = berries[lock_idx].mode
            return berries, lock_idx, f'TRACK acquire id={self._lock.track_id} mode={mode}'

        if lock_idx < 0:
            self._lock.lost_frames += 1
            if self._lock.lost_frames > self.track_lost_max_frames:
                self.reset_lock()
                return berries, -1, 'TRACK lock lost — re-acquire next call'
            coast = self._coast_tracked(k)
            return [coast], 0, (
                f'TRACK coast lost={self._lock.lost_frames}/{self.track_lost_max_frames} '
                f'id={self._lock.track_id} uv={coast.uv}')

        self._lock.lost_frames = 0
        self._lock.last_uv = (float(berries[lock_idx].uv[0]), float(berries[lock_idx].uv[1]))
        self._lock.last_z = berries[lock_idx].z_m
        self._lock.last_bbox = berries[lock_idx].bbox
        self._lock.last_mode = berries[lock_idx].mode
        if berries[lock_idx].mode == 'fp':
            self._frames_since_fp = 0

        b = berries[lock_idx]
        return (
            berries,
            lock_idx,
            f'TRACK lock id={self._lock.track_id} idx={lock_idx} z={b.z_m:.2f}m mode={b.mode}',
        )
