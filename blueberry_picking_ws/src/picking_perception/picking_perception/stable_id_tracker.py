"""Frame-to-frame stable ID tracker in image + optional base_link 3D.

Designed for moving-camera / static-target robotics:
  - Associate primarily by 3D distance when xyz is available
  - On detector miss: *coast* — keep publishing last xyz for max_misses frames
  - On re-detect near last xyz: reuse the same track_id

Detection dropout ≠ target gone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class TrackObservation:
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float = 0.0
    xyz: Optional[Tuple[float, float, float]] = None
    payload: object = None


@dataclass
class TrackUpdate:
    """One output slot for the current frame (live detect or world-frame coast)."""

    track_id: int
    coast: bool
    bbox_xyxy: Tuple[float, float, float, float]
    xyz: Optional[Tuple[float, float, float]]
    confidence: float
    hits: int
    misses: int
    payload: object = None  # live observation payload, else None when coasting


@dataclass
class _Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    xyz: Optional[Tuple[float, float, float]] = None
    hits: int = 1
    misses: int = 0
    confidence: float = 0.0
    payload: object = None


@dataclass
class StableIdTracker:
    """Greedy matcher: prefer 3D gate, fallback IoU; emit coast on misses."""

    iou_thresh: float = 0.3
    max_misses: int = 20
    max_dist_m: float = 0.20
    use_xyz: bool = True
    # Prefer 3D association when both sides have xyz (moving camera).
    prefer_xyz: bool = True
    coast_conf_scale: float = 0.55
    min_coast_conf: float = 0.15
    _next_id: int = 1
    _tracks: Dict[int, _Track] = field(default_factory=dict)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _pair_score(self, tr: _Track, o: TrackObservation) -> Optional[float]:
        if self.prefer_xyz and self.use_xyz and tr.xyz is not None and o.xyz is not None:
            d = float(np.linalg.norm(np.asarray(tr.xyz) - np.asarray(o.xyz)))
            if d > self.max_dist_m:
                return None
            # Higher is better; 3D dominates.
            score = 2.0 * max(0.0, 1.0 - d / self.max_dist_m)
            iou = _iou(tr.bbox, o.bbox_xyxy)
            if iou > 0:
                score += 0.25 * iou
            return score

        score = _iou(tr.bbox, o.bbox_xyxy)
        if score < self.iou_thresh:
            return None
        if self.use_xyz and tr.xyz is not None and o.xyz is not None:
            d = float(np.linalg.norm(np.asarray(tr.xyz) - np.asarray(o.xyz)))
            if d > self.max_dist_m:
                return None
            score = score + 0.5 * max(0.0, 1.0 - d / self.max_dist_m)
        return score

    def update(self, observations: Sequence[TrackObservation]) -> List[Tuple[int, TrackObservation]]:
        """Backward-compatible: live matches only (no coast rows)."""
        return [
            (u.track_id, TrackObservation(
                bbox_xyxy=u.bbox_xyxy,
                confidence=u.confidence,
                xyz=u.xyz,
                payload=u.payload,
            ))
            for u in self.update_full(observations)
            if not u.coast and u.payload is not None
        ]

    def update_full(self, observations: Sequence[TrackObservation]) -> List[TrackUpdate]:
        """Associate live dets; unmatched tracks are emitted as coast until max_misses."""
        obs = list(observations)
        track_ids = list(self._tracks.keys())
        assigned_t: set = set()
        assigned_o: set = set()
        pairs: List[Tuple[float, int, int]] = []

        for ti, tid in enumerate(track_ids):
            tr = self._tracks[tid]
            for oi, o in enumerate(obs):
                score = self._pair_score(tr, o)
                if score is None:
                    continue
                pairs.append((score, ti, oi))

        pairs.sort(key=lambda t: -t[0])
        live: List[TrackUpdate] = []
        for _, ti, oi in pairs:
            if ti in assigned_t or oi in assigned_o:
                continue
            tid = track_ids[ti]
            o = obs[oi]
            tr = self._tracks[tid]
            tr.bbox = o.bbox_xyxy
            if o.xyz is not None:
                tr.xyz = o.xyz
            tr.confidence = o.confidence
            tr.payload = o.payload
            tr.hits += 1
            tr.misses = 0
            assigned_t.add(ti)
            assigned_o.add(oi)
            live.append(TrackUpdate(
                track_id=tid,
                coast=False,
                bbox_xyxy=tr.bbox,
                xyz=tr.xyz,
                confidence=tr.confidence,
                hits=tr.hits,
                misses=0,
                payload=o.payload,
            ))

        for oi, o in enumerate(obs):
            if oi in assigned_o:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(
                track_id=tid,
                bbox=o.bbox_xyxy,
                xyz=o.xyz,
                confidence=o.confidence,
                payload=o.payload,
            )
            live.append(TrackUpdate(
                track_id=tid,
                coast=False,
                bbox_xyxy=o.bbox_xyxy,
                xyz=o.xyz,
                confidence=o.confidence,
                hits=1,
                misses=0,
                payload=o.payload,
            ))

        coast: List[TrackUpdate] = []
        doomed: List[int] = []
        for ti, tid in enumerate(track_ids):
            if ti in assigned_t:
                continue
            tr = self._tracks[tid]
            tr.misses += 1
            if tr.misses > self.max_misses:
                doomed.append(tid)
                continue
            conf = max(self.min_coast_conf, float(tr.confidence) * self.coast_conf_scale)
            tr.confidence = conf
            coast.append(TrackUpdate(
                track_id=tid,
                coast=True,
                bbox_xyxy=tr.bbox,
                xyz=tr.xyz,
                confidence=conf,
                hits=tr.hits,
                misses=tr.misses,
                payload=tr.payload,
            ))

        for tid in doomed:
            del self._tracks[tid]

        out = live + coast
        out.sort(key=lambda u: u.track_id)
        return out
