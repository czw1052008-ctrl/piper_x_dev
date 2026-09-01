#!/usr/bin/env python3
"""Offline static sim for pick trajectory planner (planner_input as GT world).

Rollout: tool 6D polyline → occupancy/ESDF collision + IK feasibility.
Freespace = OccupancyLocal ESDF (OctoMap-style), not obstacle spheres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from occupancy_map import BERRY, EGO, FREE, OCCUPIED, UNKNOWN
from piper_position_ik import tip_approach_axis, tip_xyz, tool_pose_ik
from picking_msgs.msg import PlannerInput, PlannerTaskContext
from tool_trajectory_utils import build_tool_trajectory_4s

DEFAULT_TIP_OFFSET = (0.0, 0.01883, 0.06152)


@dataclass
class SimRewardWeights:
    dist: float = 1.0
    path: float = 0.15
    collision: float = 5.0
    ik_fail: float = 3.0
    success: float = 10.0
    collision_margin_m: float = 0.01


@dataclass
class SimRolloutResult:
    ok: bool
    reward: float
    tip_path_m: float
    min_obstacle_clearance_m: float
    ik_failures: int
    final_tip_err_m: float
    collision: bool
    detail: str = ''


@dataclass
class PlannerSimEnv:
    """One static snapshot = one PlannerInput (clusters/berries/occupancy/ego frozen)."""

    planner_input: PlannerInput
    tip_offset_link6: Tuple[float, float, float] = DEFAULT_TIP_OFFSET
    weights: SimRewardWeights = field(default_factory=SimRewardWeights)
    execute_until_index: int = 4

    def goal_tip_and_axis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Goal from task_context + scene (static GT)."""
        tc = self.planner_input.task_context
        fsm = int(tc.fsm_state)
        berries = list(self.planner_input.berries or [])
        clusters = list(self.planner_input.clusters or [])

        if fsm == PlannerTaskContext.FSM_APPROACH_FRUIT and int(tc.fruit_id) >= 0:
            for b in berries:
                if int(b.id) == int(tc.fruit_id):
                    goal = np.asarray(b.position, dtype=np.float64)
                    if b.normal_valid:
                        n = np.asarray(b.surface_normal, dtype=np.float64)
                        nn = float(np.linalg.norm(n))
                        axis = -n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
                    else:
                        tip0 = tip_xyz(
                            self.planner_input.ego.q_rad, tip_offset_link6=self.tip_offset_link6)
                        d = goal - tip0
                        dn = float(np.linalg.norm(d))
                        axis = d / dn if dn > 1e-9 else np.array([0.0, 0.0, 1.0])
                    return goal, axis
            if berries:
                b = berries[0]
                goal = np.asarray(b.position, dtype=np.float64)
                tip0 = tip_xyz(
                    self.planner_input.ego.q_rad, tip_offset_link6=self.tip_offset_link6)
                d = goal - tip0
                dn = float(np.linalg.norm(d))
                axis = d / dn if dn > 1e-9 else np.array([0.0, 0.0, 1.0])
                return goal, axis

        cid = int(tc.cluster_id)
        cluster = None
        for c in clusters:
            if int(c.id) == cid:
                cluster = c
                break
        if cluster is None and clusters:
            cluster = clusters[0]
        if cluster is None:
            tip0 = tip_xyz(
                self.planner_input.ego.q_rad, tip_offset_link6=self.tip_offset_link6)
            return tip0 + np.array([0.05, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])

        center = np.asarray(cluster.position, dtype=np.float64)
        tip0 = tip_xyz(self.planner_input.ego.q_rad, tip_offset_link6=self.tip_offset_link6)
        d = center - tip0
        dn = float(np.linalg.norm(d))
        axis = d / dn if dn > 1e-9 else np.array([0.0, 0.0, 1.0])
        standoff = float(getattr(self.planner_input.end_effector, 'approach_standoff_m', 0.07) or 0.07)
        if standoff < 0.03:
            standoff = 0.07
        goal = center - axis * standoff
        return goal, axis

    def _occ_arrays(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, Tuple[int, int, int]]]:
        oc = getattr(self.planner_input, 'occupancy_local', None)
        if oc is None:
            return None
        size = list(oc.size_xyz) if oc.size_xyz is not None else [0, 0, 0]
        size = [int(v) for v in size]
        if len(size) < 3 or size[0] <= 0:
            return None
        nx, ny, nz = size[0], size[1], size[2]
        n = nx * ny * nz
        labels = np.asarray(oc.labels, dtype=np.uint8)
        esdf = np.asarray(oc.esdf, dtype=np.float32)
        if labels.size < n or esdf.size < n:
            return None
        labels = labels[:n].reshape(nx, ny, nz)
        esdf = esdf[:n].reshape(nx, ny, nz)
        origin = np.asarray(oc.origin_xyz, dtype=np.float64)
        voxel = float(oc.voxel_m) if float(oc.voxel_m) > 1e-6 else 0.02
        return origin, labels, esdf, voxel, (nx, ny, nz)

    def query_occupancy(self, tip: np.ndarray) -> Tuple[int, float]:
        """Return (label, esdf_m) from occupancy_local; UNKNOWN/nan if missing."""
        pack = self._occ_arrays()
        if pack is None:
            return UNKNOWN, float('nan')
        origin, labels, esdf, voxel, (nx, ny, nz) = pack
        rel = (np.asarray(tip, dtype=np.float64) - origin) / voxel
        i, j, k = [int(np.floor(x)) for x in rel]
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return UNKNOWN, float('nan')
        return int(labels[i, j, k]), float(esdf[i, j, k])

    def min_clearance(self, tip: np.ndarray) -> float:
        lab, d = self.query_occupancy(tip)
        if lab == OCCUPIED:
            return 0.0
        if not math.isfinite(d):
            return float('inf')
        return float(d)

    def check_collision(self, tip: np.ndarray) -> bool:
        return not self.is_free(tip)

    def is_free(self, tip: np.ndarray) -> bool:
        """ESDF free-space: not OCCUPIED/UNKNOWN, clearance ≥ margin.

        EGO/BERRY voxels are traversable (arm is ego; fruit is goal). Only
        OCCUPIED depth hits are hard obstacles.
        """
        m = self.weights.collision_margin_m
        ee_r = float(getattr(self.planner_input.ego, 'ee_radius_m', 0.02) or 0.02)
        need = m + ee_r * 0.25
        lab, d = self.query_occupancy(tip)
        if lab == OCCUPIED:
            return False
        if lab == UNKNOWN:
            return False
        if not math.isfinite(d):
            return False
        return d >= need

    def rollout_tool_positions(
        self,
        tip_positions: Sequence[Sequence[float]],
        approach_axes: Optional[Sequence[Sequence[float]]] = None,
    ) -> SimRolloutResult:
        q_seed = list(self.planner_input.ego.q_rad)
        goal, goal_axis = self.goal_tip_and_axis()
        tip0 = tip_xyz(q_seed, tip_offset_link6=self.tip_offset_link6)

        path_len = 0.0
        min_clear = float('inf')
        ik_fail = 0
        collision = False
        q = q_seed
        prev_tip = tip0

        axes = approach_axes
        for i, tip in enumerate(tip_positions):
            tip_a = np.asarray(tip, dtype=np.float64)
            axis = goal_axis
            if axes is not None and i < len(axes):
                a = np.asarray(axes[i], dtype=np.float64)
                if float(np.linalg.norm(a)) > 1e-9:
                    axis = a / float(np.linalg.norm(a))

            path_len += float(np.linalg.norm(tip_a - prev_tip))
            prev_tip = tip_a
            min_clear = min(min_clear, self.min_clearance(tip_a))
            if self.check_collision(tip_a):
                collision = True

            q_new = tool_pose_ik(
                tip_a.tolist(), axis.tolist(), q,
                tip_offset_link6=self.tip_offset_link6)
            if q_new is None:
                ik_fail += 1
            else:
                q = q_new

        final_err = float(np.linalg.norm(prev_tip - goal))
        w = self.weights
        reward = (
            -w.dist * final_err
            -w.path * path_len
            -w.collision * (1.0 if collision else 0.0)
            -w.ik_fail * ik_fail
        )
        success = (final_err < 0.025 and not collision and ik_fail == 0)
        if success:
            reward += w.success

        return SimRolloutResult(
            ok=success,
            reward=float(reward),
            tip_path_m=float(path_len),
            min_obstacle_clearance_m=float(min_clear),
            ik_failures=int(ik_fail),
            final_tip_err_m=float(final_err),
            collision=collision,
            detail=f'err={final_err*1e3:.1f}mm path={path_len:.3f}m ik_fail={ik_fail}',
        )

    def rollout_from_tool_trajectory(self, traj_msg) -> SimRolloutResult:
        exec_i = min(int(traj_msg.execute_until_index), len(traj_msg.waypoints) - 1)
        exec_i = max(0, exec_i)
        tips = [wp.position for wp in traj_msg.waypoints[: exec_i + 1]]
        axes = [wp.approach_axis for wp in traj_msg.waypoints[: exec_i + 1]]
        return self.rollout_tool_positions(tips, axes)

    def teacher_straight_line(self, *, move_duration_s: float = 1.0):
        """PBVS-style straight tip interp (for smoke / BC baseline)."""
        tip0 = tip_xyz(
            self.planner_input.ego.q_rad, tip_offset_link6=self.tip_offset_link6)
        goal, axis = self.goal_tip_and_axis()
        return build_tool_trajectory_4s(
            tip0, goal, axis,
            move_duration_s=move_duration_s,
            execute_until_index=self.execute_until_index,
        )

    def evaluate_teacher(self) -> SimRolloutResult:
        traj = self.teacher_straight_line()
        return self.rollout_from_tool_trajectory(traj)


def planner_input_from_dict(d: Dict[str, Any]) -> PlannerInput:
    """Minimal synthetic PlannerInput for unit tests."""
    from picking_msgs.msg import (
        OccupancyLocal,
        PlannerEgo,
        PlannerEndEffector,
        PlannerTaskContext,
        SceneBerry,
        SceneCluster,
    )

    msg = PlannerInput()
    msg.header.frame_id = 'base_link'
    ego = PlannerEgo()
    ego.q_rad = [float(v) for v in d.get('q_rad', [0.1, 0.8, -1.2, 0.0, -0.5, 0.0])]
    ego.qd_rad = [0.0] * 6
    msg.ego = ego

    tc = PlannerTaskContext()
    tc.fsm_state = int(d.get('fsm_state', PlannerTaskContext.FSM_CLUSTER_ALIGN))
    tc.cluster_id = int(d.get('cluster_id', 0))
    tc.fruit_id = int(d.get('fruit_id', -1))
    msg.task_context = tc

    ee = PlannerEndEffector()
    ee.approach_standoff_m = float(d.get('approach_standoff_m', 0.07))
    ee.cup_radius_m = 0.012
    ee.tip_offset_link6 = list(DEFAULT_TIP_OFFSET)
    msg.end_effector = ee

    c = SceneCluster()
    c.id = 0
    c.position = [float(v) for v in d.get('cluster', [0.35, 0.0, 0.25])]
    c.confidence = 0.9
    msg.clusters = [c]

    if 'berry' in d:
        b = SceneBerry()
        b.id = int(d.get('fruit_id', 1))
        b.position = [float(v) for v in d['berry']]
        b.confidence = 0.9
        b.visible_wrist = True
        b.normal_valid = False
        msg.berries = [b]

    n = 16
    oc = OccupancyLocal()
    oc.origin_xyz = [0.2, -0.2, 0.0]
    oc.voxel_m = 0.02
    oc.size_xyz = [n, n, n]
    labels = np.full((n, n, n), FREE, dtype=np.uint8)
    esdf = np.full((n, n, n), 0.25, dtype=np.float32)
    # Optional hard obstacle block in crop for collision tests.
    for o in d.get('occ_blocks', []):
        # o: {ijk: [i,j,k]} or position in world
        if 'ijk' in o:
            i, j, k = [int(v) for v in o['ijk']]
            if 0 <= i < n and 0 <= j < n and 0 <= k < n:
                labels[i, j, k] = OCCUPIED
                esdf[i, j, k] = 0.0
    oc.labels = labels.reshape(-1).tolist()
    oc.esdf = esdf.reshape(-1).tolist()
    oc.crop_center_xyz = [float(v) for v in d.get('berry', d.get('cluster', [0.35, 0.0, 0.25]))]
    msg.occupancy_local = oc
    msg.obstacles = []
    return msg


def load_planner_inputs_from_bag(bag_dir: str, *, max_msgs: int = 200) -> List[PlannerInput]:
    """Load PlannerInput messages from rosbag2 directory."""
    try:
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rclpy.serialization import deserialize_message
    except ImportError as exc:
        raise RuntimeError('rosbag2_py required: source ROS 2 setup.bash') from exc

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'),
    )
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if '/planning/planner_input' not in topics:
        raise FileNotFoundError(f'no /planning/planner_input in {bag_dir}')

    out: List[PlannerInput] = []
    while reader.has_next() and len(out) < max_msgs:
        topic, data, _ = reader.read_next()
        if topic != '/planning/planner_input':
            continue
        out.append(deserialize_message(data, PlannerInput))
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', default='', help='rosbag2 dir with planner_input')
    parser.add_argument('--synthetic', action='store_true', help='run built-in fixture')
    args = parser.parse_args()

    if args.bag:
        msgs = load_planner_inputs_from_bag(args.bag, max_msgs=20)
        if not msgs:
            print('no planner_input in bag')
            return 1
        env = PlannerSimEnv(msgs[0])
        print(f'bag samples={len(msgs)}')
    else:
        env = PlannerSimEnv(planner_input_from_dict({
            'cluster': [0.32, 0.05, 0.28],
        }))

    teacher = env.evaluate_teacher()
    print(
        f'teacher rollout: reward={teacher.reward:.3f} ok={teacher.ok} '
        f'{teacher.detail} min_clear={teacher.min_obstacle_clearance_m*1e3:.1f}mm')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
