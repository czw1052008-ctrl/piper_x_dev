#!/usr/bin/env python3
"""Build a near-to-far fruit queue from global camera detections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class FruitTarget:
    index: int
    track_id: int
    confidence: float
    base_xyz: Tuple[float, float, float]
    dist_m: float


def _berry_xyz(berry) -> Optional[Tuple[float, float, float]]:
    try:
        p = berry.pose.pose.position
        return float(p.x), float(p.y), float(p.z)
    except Exception:
        return None


def sort_berries_near_to_far(berries: Sequence, origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[FruitTarget]:
    """Sort global detections by distance from origin (default base_link)."""
    ox, oy, oz = origin
    out: List[FruitTarget] = []
    for i, b in enumerate(berries):
        xyz = _berry_xyz(b)
        if xyz is None:
            continue
        x, y, z = xyz
        if not all(math.isfinite(v) for v in (x, y, z)):
            continue
        if abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6:
            continue
        dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2)
        out.append(FruitTarget(
            index=i,
            track_id=int(getattr(b, 'track_id', -1)),
            confidence=float(getattr(b, 'confidence', 0.0)),
            base_xyz=(x, y, z),
            dist_m=dist,
        ))
    out.sort(key=lambda t: t.dist_m)
    return out


def filter_berries_in_cluster(
    berries: Sequence,
    cluster_center: Tuple[float, float, float],
    *,
    radius_m: float = 0.25,
) -> List:
    cx, cy, cz = cluster_center
    kept = []
    for b in berries:
        xyz = _berry_xyz(b)
        if xyz is None:
            continue
        x, y, z = xyz
        d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        if d <= radius_m:
            kept.append(b)
    return kept
