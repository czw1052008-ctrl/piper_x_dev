#!/usr/bin/env python3
"""Global-camera cluster selection and align-pose planning.

Current: load refine_entry_pose / align_decision JSON (agent/VLM prep).
Future: auto plan from global YOLO + depth so wrist sees full cluster.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from fruit_queue import FruitTarget, _berry_xyz, filter_berries_in_cluster, sort_berries_near_to_far

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_ENTRY_POSE = SCRIPTS_DIR.parent / 'log' / 'real_robot' / 'refine_entry_pose.json'


@dataclass
class ClusterPlan:
    cluster_index: int
    cluster_center: Tuple[float, float, float]
    cluster_score: float
    align_joints_rad: List[float]
    fruit_targets: List[FruitTarget]
    source: str


def _cluster_score(berries: Sequence) -> Tuple[float, Tuple[float, float, float]]:
    """Heuristic: prefer larger, nearer clusters (sum conf / mean dist)."""
    if not berries:
        return 0.0, (0.0, 0.0, 0.0)
    xs, ys, zs, confs = [], [], [], []
    for b in berries:
        xyz = _berry_xyz(b)
        if xyz is None:
            continue
        x, y, z = xyz
        xs.append(x)
        ys.append(y)
        zs.append(z)
        confs.append(float(getattr(b, 'confidence', 0.0)))
    if not xs:
        return 0.0, (0.0, 0.0, 0.0)
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)
    mean_dist = math.sqrt(cx * cx + cy * cy + cz * cz)
    score = sum(confs) / max(mean_dist, 0.15)
    return score, (cx, cy, cz)


def select_best_cluster(berries: Sequence, *, cluster_radius_m: float = 0.25) -> Tuple[int, Tuple[float, float, float], float]:
    """Group berries by proximity; return index of best cluster center."""
    if not berries:
        return -1, (0.0, 0.0, 0.0), 0.0

    # Simple greedy: each berry seeds a cluster; pick highest score.
    seeds = []
    for i, b in enumerate(berries):
        xyz = _berry_xyz(b)
        if xyz is None:
            continue
        members = filter_berries_in_cluster(berries, xyz, radius_m=cluster_radius_m)
        score, center = _cluster_score(members)
        seeds.append((score, i, center, len(members)))
    if not seeds:
        return -1, (0.0, 0.0, 0.0), 0.0
    seeds.sort(key=lambda t: (t[0], t[3]), reverse=True)
    score, idx, center, _ = seeds[0]
    return idx, center, score


def load_align_joints_from_json(
    entry_pose_path: Path = DEFAULT_ENTRY_POSE,
    align_decision_path: Optional[Path] = None,
) -> Optional[List[float]]:
    """Load target joints from refine_entry_pose.json or align_decision coarse_ok."""
    if entry_pose_path.is_file():
        with open(entry_pose_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        joints = data.get('joints_rad')
        if joints and len(joints) >= 6:
            return [float(v) for v in joints[:6]]

    if align_decision_path and align_decision_path.is_file():
        with open(align_decision_path, 'r', encoding='utf-8') as f:
            dec = json.load(f)
        if dec.get('action') in ('coarse_ok', 'done', 'set_joints'):
            jd = dec.get('joints_deg') or {}
            if jd:
                keys = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
                return [math.radians(float(jd.get(k, 0.0))) for k in keys]
    return None


def plan_cluster_align(
    berries: Sequence,
    *,
    entry_pose_path: Path = DEFAULT_ENTRY_POSE,
    align_decision_path: Optional[Path] = None,
    cluster_radius_m: float = 0.25,
) -> Optional[ClusterPlan]:
    """Select cluster + fruit queue; align joints from JSON until VLM planner lands."""
    cluster_idx, center, score = select_best_cluster(berries, cluster_radius_m=cluster_radius_m)
    if cluster_idx < 0:
        return None

    members = filter_berries_in_cluster(berries, center, radius_m=cluster_radius_m)
    fruit_targets = sort_berries_near_to_far(members, origin=(0.0, 0.0, 0.0))
    joints = load_align_joints_from_json(entry_pose_path, align_decision_path)
    if joints is None:
        return None

    source = 'json'
    if align_decision_path and align_decision_path.is_file():
        source = f'align_decision:{align_decision_path.name}'
    elif entry_pose_path.is_file():
        source = f'entry_pose:{entry_pose_path.name}'

    return ClusterPlan(
        cluster_index=cluster_idx,
        cluster_center=center,
        cluster_score=score,
        align_joints_rad=joints,
        fruit_targets=fruit_targets,
        source=source,
    )


def load_test_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
