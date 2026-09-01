#!/usr/bin/env python3
"""Assemble /planning/planner_input @ 1 Hz from existing perception + FSM topics.

P0: verify input data chain; see docs/PICK_TRAJECTORY_PLANNER_DATA.md

Usage:
  export PYTHONPATH="$(pwd)/scripts:${PYTHONPATH}"
  python3 scripts/planner_input_assembler.py --end-effector suction_cup_v1
"""

from __future__ import annotations

import argparse
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from end_effector_profile import load_profile
from piper_position_ik import fk_link_origins, tip_approach_axis, tip_xyz
from picking_msgs.msg import (
    DetectedBerry,
    DetectedBerryArray,
    OccupancyLocal,
    PerceptionScene,
    PlannerEgo,
    PlannerEndEffector,
    PlannerInput,
    PlannerTaskContext,
  SceneBerry,
  SceneCluster,
  SceneObstacle,
)
try:
    from picking_msgs.msg import SceneObstacleArray
except ImportError:
    SceneObstacleArray = None  # type: ignore

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

# PlannerInput.input_flags bits
FLAG_NO_CLUSTERS = 1 << 0
FLAG_NO_BERRIES = 1 << 1
FLAG_NO_JOINTS = 1 << 2
FLAG_CLUSTER_ID_UNKNOWN = 1 << 3
FLAG_FRUIT_ID_UNKNOWN = 1 << 4
FLAG_NO_OCCUPANCY = 1 << 5

MAX_CLUSTERS = 8
MAX_BERRIES = 10  # model input cap; ACTIVE + PENDING queue
MAX_OBSTACLES = 12

# Upper hemisphere about base_link (matches occupancy_map defaults).
DEFAULT_WORKSPACE_CENTER = (0.0, 0.0, 0.0)
DEFAULT_WORKSPACE_RADIUS_M = 0.85
DEFAULT_WORKSPACE_Z_MIN = 0.02
DEFAULT_WORKSPACE_AABB = (-0.85, 0.85, -0.85, 0.85, 0.02, 0.85)
DEFAULT_LINK_RADIUS_M = 0.05

PICK_ROLE_NAME = {
    0: 'pending',   # 待采集
    1: 'active',    # 采集（当前接近目标）
    2: 'done',      # 已采
}


def _xyz(berry: DetectedBerry) -> Optional[Tuple[float, float, float]]:
    try:
        p = berry.pose.pose.position
        x, y, z = float(p.x), float(p.y), float(p.z)
        if all(math.isfinite(v) for v in (x, y, z)):
            return x, y, z
    except Exception:
        pass
    return None


def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def map_fsm_state(reach_status: str, pick_status: str) -> int:
    pick = (pick_status or '').split(':')[0].strip().upper()
    reach = (reach_status or '').split(':')[0].strip().upper()

    if pick.startswith('FRUIT_RETRACT'):
        return PlannerTaskContext.FSM_RETRACT
    if pick.startswith('FRUIT_') or reach == 'REFINING':
        return PlannerTaskContext.FSM_APPROACH_FRUIT
    if pick.startswith('CLUSTER_') or reach in ('ALIGNING', 'LOCKING', 'PLANNING', 'APPROACHING'):
        return PlannerTaskContext.FSM_CLUSTER_ALIGN
    if reach in ('WAIT_CONFIRM', 'HOLD'):
        return PlannerTaskContext.FSM_HOLD
    if pick == 'IDLE' and reach in ('', 'IDLE'):
        return PlannerTaskContext.FSM_IDLE
    if pick or reach:
        return PlannerTaskContext.FSM_HOLD
    return PlannerTaskContext.FSM_IDLE


def infer_cluster_id(
    clusters: List[SceneCluster],
    target_lock: Optional[DetectedBerry],
) -> Tuple[int, bool]:
    if not clusters or target_lock is None:
        return -1, True
    lock_xyz = _xyz(target_lock)
    if lock_xyz is None or float(target_lock.confidence) <= 0.0:
        return -1, True
    best_i = -1
    best_d = float('inf')
    for i, c in enumerate(clusters):
        d = _dist3(lock_xyz, (c.position[0], c.position[1], c.position[2]))
        if d < best_d:
            best_d = d
            best_i = c.id
    if best_i < 0 or best_d > 0.35:
        return -1, True
    return int(best_i), False


class PlannerInputAssembler(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('planner_input_assembler')
        self._profile = load_profile(args.end_effector)
        self._seq = 0
        self._scene: Optional[PerceptionScene] = None
        self._global: Optional[DetectedBerryArray] = None
        self._fine: Optional[DetectedBerryArray] = None
        self._target_lock: Optional[DetectedBerry] = None
        self._joints: Optional[List[float]] = None
        self._joint_vel: Optional[List[float]] = None
        self._reach_status = ''
        self._pick_status = ''
        self._task_ctx: Optional[PlannerTaskContext] = None
        self._obstacles: List[SceneObstacle] = []
        self._occupancy_local: Optional[OccupancyLocal] = None
        self._fruit_id_override = int(getattr(args, 'fruit_id', -1))

        self.create_subscription(
            PerceptionScene, '/perception/scene_graph', self._on_scene, 10)
        self.create_subscription(
            DetectedBerryArray, '/perception/global/berries', self._on_global, 10)
        self.create_subscription(
            DetectedBerryArray, '/perception/fine/berries', self._on_fine, 10)
        self.create_subscription(
            DetectedBerry, '/perception/target_lock', self._on_lock, 10)
        self.create_subscription(JointState, '/feedback/joint_states', self._on_joints, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.create_subscription(String, '/reach/status', self._on_reach_status, 10)
        self.create_subscription(String, '/pick/status', self._on_pick_status, 10)
        self.create_subscription(
            PlannerTaskContext, '/planning/task_context', self._on_task_context, 10)
        self.create_subscription(
            OccupancyLocal, '/perception/occupancy_local', self._on_occupancy_local, 10)
        try:
            if SceneObstacleArray is not None:
                self.create_subscription(
                    SceneObstacleArray, '/perception/obstacles', self._on_obstacles, 10)
        except Exception:
            self.get_logger().warn('SceneObstacleArray unavailable — obstacles=[]')

        self._pub = self.create_publisher(PlannerInput, '/planning/planner_input', 10)
        period = max(float(args.rate_hz), 0.1)
        self.create_timer(1.0 / period, self._tick)
        self.get_logger().info(
            f'planner_input_assembler @ {1.0 / period:.1f}s  end_effector={args.end_effector} '
            f'(prefer /perception/scene_graph)')

    def _on_scene(self, msg: PerceptionScene) -> None:
        self._scene = msg

    def _on_global(self, msg: DetectedBerryArray) -> None:
        self._global = msg

    def _on_fine(self, msg: DetectedBerryArray) -> None:
        self._fine = msg

    def _on_lock(self, msg: DetectedBerry) -> None:
        self._target_lock = msg

    def _on_joints(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))
        if not all(j in name_to_pos for j in ARM_JOINTS):
            return
        self._joints = [float(name_to_pos[j]) for j in ARM_JOINTS]
        self._joint_vel = [
            float(name_to_vel[j]) if j in name_to_vel else 0.0 for j in ARM_JOINTS
        ]

    def _on_reach_status(self, msg: String) -> None:
        self._reach_status = msg.data.strip()

    def _on_pick_status(self, msg: String) -> None:
        self._pick_status = msg.data.strip()

    def _on_task_context(self, msg: PlannerTaskContext) -> None:
        self._task_ctx = msg

    def _on_obstacles(self, msg) -> None:
        # Deprecated freespace path; kept empty-compatible for old bags.
        self._obstacles = list(msg.obstacles or [])[:MAX_OBSTACLES]

    def _on_occupancy_local(self, msg: OccupancyLocal) -> None:
        self._occupancy_local = msg

    def _build_clusters(self) -> List[SceneCluster]:
        if self._scene is not None and self._scene.clusters:
            return list(self._scene.clusters[:MAX_CLUSTERS])
        out: List[SceneCluster] = []
        if self._global is None:
            return out
        for b in self._global.berries[:MAX_CLUSTERS]:
            xyz = _xyz(b)
            if xyz is None:
                continue
            c = SceneCluster()
            tid = int(b.track_id)
            c.id = tid if tid >= 0 else len(out)
            c.position = [xyz[0], xyz[1], xyz[2]]
            c.confidence = float(b.confidence)
            c.bbox_u0 = float(getattr(b, 'bbox_u0', -1.0))
            c.bbox_v0 = float(getattr(b, 'bbox_v0', -1.0))
            c.bbox_u1 = float(getattr(b, 'bbox_u1', -1.0))
            c.bbox_v1 = float(getattr(b, 'bbox_v1', -1.0))
            c.track_source = 'live'
            out.append(c)
        return out

    def _build_berries(self) -> List[SceneBerry]:
        if self._scene is not None and self._scene.berries:
            out = list(self._scene.berries[:MAX_BERRIES])
        else:
            out = []
            if self._fine is None:
                return out
            for b in self._fine.berries[:MAX_BERRIES]:
                xyz = _xyz(b)
                if xyz is None:
                    continue
                sb = SceneBerry()
                sb.id = int(b.track_id) if int(b.track_id) >= 0 else len(out)
                sb.position = [xyz[0], xyz[1], xyz[2]]
                sb.confidence = float(b.confidence)
                sb.visible_wrist = True
                sb.visible_global = False
                sb.bbox_u0 = float(getattr(b, 'bbox_u0', -1.0))
                sb.bbox_v0 = float(getattr(b, 'bbox_v0', -1.0))
                sb.bbox_u1 = float(getattr(b, 'bbox_u1', -1.0))
                sb.bbox_v1 = float(getattr(b, 'bbox_v1', -1.0))
                sb.image_u = float(getattr(b, 'image_u', -1.0))
                sb.image_v = float(getattr(b, 'image_v', -1.0))
                sb.depth_mode = str(getattr(b, 'depth_mode', '') or '')
                sb.z_depth_m = float(getattr(b, 'z_depth_m', -1.0))
                sb.z_mono_m = float(getattr(b, 'z_mono_m', -1.0))
                cd = float(getattr(b, 'center_depth_m', -1.0))
                if cd <= 0.05:
                    if sb.z_depth_m > 0.05:
                        cd = sb.z_depth_m
                    elif sb.z_mono_m > 0.05:
                        cd = sb.z_mono_m
                    else:
                        cd = -1.0
                sb.center_depth_m = cd
                raw_n = getattr(b, 'surface_normal_base', None)
                if raw_n is None:
                    n = [0.0, 0.0, 0.0]
                else:
                    try:
                        n = [float(raw_n[0]), float(raw_n[1]), float(raw_n[2])]
                    except (TypeError, IndexError, ValueError):
                        n = [0.0, 0.0, 0.0]
                if len(n) < 3:
                    n = [0.0, 0.0, 0.0]
                sb.surface_normal = [float(n[0]), float(n[1]), float(n[2])]
                sb.normal_valid = bool(getattr(b, 'normal_valid', False))
                src = str(getattr(b, 'depth_mode', '') or '')
                sb.track_source = 'coast' if src == 'coast' else 'live'
                out.append(sb)
        return out

    def _apply_pick_roles(
        self, berries: List[SceneBerry], fruit_id: int
    ) -> List[SceneBerry]:
        """Stamp pick_role from task_context.fruit_id; cap to MAX_BERRIES.

        ACTIVE = 采集目标（模型/PBVS 只接近这颗）；其余 PENDING=待采集。
        """
        fid = int(fruit_id)
        for b in berries:
            if fid >= 0 and int(b.id) == fid:
                b.pick_role = SceneBerry.PICK_ROLE_ACTIVE
            elif int(getattr(b, 'pick_role', 0)) == SceneBerry.PICK_ROLE_DONE:
                b.pick_role = SceneBerry.PICK_ROLE_DONE
            else:
                b.pick_role = SceneBerry.PICK_ROLE_PENDING
        # Prefer ACTIVE, then higher confidence; hard cap 10 for the model.
        berries.sort(
            key=lambda b: (
                0 if int(b.pick_role) == SceneBerry.PICK_ROLE_ACTIVE else 1,
                -float(b.confidence),
            ))
        return berries[:MAX_BERRIES]

    def _tick(self) -> None:
        flags = 0
        clusters = self._build_clusters()
        if not clusters:
            flags |= FLAG_NO_CLUSTERS

        task = PlannerTaskContext()
        if self._task_ctx is not None:
            task = self._task_ctx
        else:
            task.fsm_state = map_fsm_state(self._reach_status, self._pick_status)
            cid, unknown = infer_cluster_id(clusters, self._target_lock)
            task.cluster_id = cid
            task.fruit_id = -1
            if unknown:
                flags |= FLAG_CLUSTER_ID_UNKNOWN
            flags |= FLAG_FRUIT_ID_UNKNOWN
        if self._fruit_id_override >= 0:
            task.fruit_id = int(self._fruit_id_override)
            flags &= ~FLAG_FRUIT_ID_UNKNOWN

        berries = self._apply_pick_roles(self._build_berries(), int(task.fruit_id))
        # Keep fruit_id consistent with ACTIVE berry when override/queue set it.
        for b in berries:
            if int(b.pick_role) == SceneBerry.PICK_ROLE_ACTIVE:
                task.fruit_id = int(b.id)
                break
        if not berries:
            flags |= FLAG_NO_BERRIES

        ego = PlannerEgo()
        if self._joints is None:
            flags |= FLAG_NO_JOINTS
            ego.q_rad = [0.0] * 6
            ego.qd_rad = [0.0] * 6
            ego.tip_xyz = [0.0, 0.0, 0.0]
            ego.approach_axis = [0.0, 0.0, 1.0]
            ego.ee_radius_m = float(self._profile.cup_radius_m or 0.02)
            ego.link_xyz = [0.0] * 18
            ego.link_radius_m = float(DEFAULT_LINK_RADIUS_M)
        else:
            q = [float(v) for v in self._joints[:6]]
            ego.q_rad = q
            ego.qd_rad = [float(v) for v in (self._joint_vel or [0.0] * 6)[:6]]
            tip = tip_xyz(q, tip_offset_link6=self._profile.tip_offset_link6)
            ego.tip_xyz = [float(tip[0]), float(tip[1]), float(tip[2])]
            axis = tip_approach_axis(q)
            ego.approach_axis = [float(axis[0]), float(axis[1]), float(axis[2])]
            ego.ee_radius_m = float(self._profile.cup_radius_m or 0.02)
            origins = fk_link_origins(q)
            flat: List[float] = []
            for i in range(6):
                flat.extend([
                    float(origins[i, 0]), float(origins[i, 1]), float(origins[i, 2])])
            ego.link_xyz = flat
            ego.link_radius_m = float(DEFAULT_LINK_RADIUS_M)
        ego.workspace_aabb = [float(v) for v in DEFAULT_WORKSPACE_AABB]
        ego.workspace_center = [float(v) for v in DEFAULT_WORKSPACE_CENTER]
        ego.workspace_radius_m = float(DEFAULT_WORKSPACE_RADIUS_M)
        ego.workspace_z_min = float(DEFAULT_WORKSPACE_Z_MIN)

        ee = PlannerEndEffector()
        fields = self._profile.to_planner_msg_fields()
        ee.tool_type_id = int(fields['tool_type_id'])
        ee.profile_name = str(fields['profile_name'])
        ee.tip_offset_link6 = [float(v) for v in fields['tip_offset_link6']]
        ee.approach_standoff_m = float(fields['approach_standoff_m'])
        ee.cup_radius_m = float(fields['cup_radius_m'])

        msg = PlannerInput()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.seq = self._seq
        self._seq += 1
        msg.clusters = clusters
        msg.berries = berries
        msg.obstacles = []  # freespace via occupancy_local (spheres deprecated)
        if self._occupancy_local is not None:
            msg.occupancy_local = self._occupancy_local
        else:
            flags |= FLAG_NO_OCCUPANCY
            msg.occupancy_local = OccupancyLocal()
        msg.ego = ego
        msg.task_context = task
        msg.end_effector = ee
        msg.input_flags = flags
        self._pub.publish(msg)

        if flags:
            bits = []
            if flags & FLAG_NO_CLUSTERS:
                bits.append('no_clusters')
            if flags & FLAG_NO_BERRIES:
                bits.append('no_berries')
            if flags & FLAG_NO_JOINTS:
                bits.append('no_joints')
            if flags & FLAG_CLUSTER_ID_UNKNOWN:
                bits.append('cluster_id_unknown')
            if flags & FLAG_FRUIT_ID_UNKNOWN:
                bits.append('fruit_id_unknown')
            if flags & FLAG_NO_OCCUPANCY:
                bits.append('no_occupancy')
            self.get_logger().debug(f'planner_input flags={",".join(bits)} seq={msg.seq}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--end-effector', default='suction_cup_v1')
    parser.add_argument('--rate-hz', type=float, default=1.0,
                        help='planner_input publish rate (default 1 Hz replan)')
    parser.add_argument(
        '--fruit-id', type=int, default=-1,
        help='QA override: mark this berry.id as ACTIVE (采集); others PENDING')
    args = parser.parse_args()

    rclpy.init()
    node = PlannerInputAssembler(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
