#!/usr/bin/env python3
"""Real-robot suction pipeline: YOLO+FP detect -> nearest berry -> plan -> optional move."""

from __future__ import annotations

import argparse
import os
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, MotionPlanRequest, MoveItErrorCodes, PlanningOptions, PositionConstraint
from picking_msgs.srv import PlanSuction, TriggerFineDetection
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for rel in (
    'install/picking_msgs/lib/python3.12/site-packages',
    'install/picking_perception/lib/python3.12/site-packages',
):
    p = os.path.join(_WS, rel)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _print_pose(label: str, ps: PoseStamped) -> None:
    p = ps.pose.position
    print(f'  {label} [{ps.header.frame_id}] ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')


class ApproachNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('real_suction_approach')
        self._args = args
        self._fp = self.create_client(TriggerFineDetection, 'trigger_fine_detection')
        self._plan = self.create_client(PlanSuction, 'plan_suction')
        self._move = ActionClient(self, MoveGroup, args.move_action)

    def _wait(self, client, name: str, timeout: float = 30.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f'{name} unavailable')
            return False
        return True

    def run(self) -> int:
        if not self._wait(self._fp, 'trigger_fine_detection'):
            return 1
        if not self._wait(self._plan, 'plan_suction'):
            print('Start grasp planner: bash scripts/run_grasp_planner.sh', file=sys.stderr)
            return 1

        fut = self._fp.call_async(TriggerFineDetection.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=self._args.fp_timeout)
        det = fut.result()
        if det is None or not det.success or not det.detected_berries:
            print(f'[approach] detect failed: {det.message if det else "timeout"}')
            return 1
        lock_i = int(getattr(det, 'locked_berry_index', -1))
        print(f'[approach] detected {len(det.detected_berries)} berries — {det.message}')
        if lock_i >= 0:
            berries_for_plan = [det.detected_berries[lock_i]]
            print(f'[approach] using locked berry index={lock_i}')
        else:
            berries_for_plan = list(det.detected_berries)

        pfut = self._plan.call_async(PlanSuction.Request(berries=berries_for_plan))
        rclpy.spin_until_future_complete(self, pfut, timeout_sec=30.0)
        plan_res = pfut.result()
        if plan_res is None or not plan_res.success:
            print(f'[approach] plan_suction failed: {plan_res.message if plan_res else "timeout"}')
            return 1

        plan = plan_res.plan
        print('[approach] suction plan (nearest berry):')
        _print_pose('pre_grasp', plan.pre_grasp)
        _print_pose('grasp', plan.grasp)
        _print_pose('post_grasp', plan.post_grasp)

        if not self._args.move:
            print('[approach] dry-run only (add --move to execute arm motion)')
            return 0

        if not self._move.wait_for_server(timeout_sec=10.0):
            print('[approach] move_action unavailable — start arm bringup first', file=sys.stderr)
            return 1

        for label, target in ('pre_grasp', plan.pre_grasp), ('grasp', plan.grasp):
            print(f'[approach] moving to {label} ...')
            if not self._send_pose(target):
                return 1
        print('[approach] at grasp pose — enable suction manually (GPIO not wired yet)')
        return 0

    def _send_pose(self, target: PoseStamped) -> bool:
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self._args.move_group
        req.num_planning_attempts = 5
        req.allowed_planning_time = 8.0
        req.max_velocity_scaling_factor = self._args.velocity_scale
        req.max_acceleration_scaling_factor = self._args.velocity_scale

        pc = PositionConstraint()
        pc.header = target.header
        pc.link_name = self._args.ee_link
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [0.02]
        pc.constraint_region.primitives = [region]
        pc.constraint_region.primitive_poses = [target.pose]
        pc.weight = 1.0
        req.goal_constraints = [Constraints(position_constraints=[pc])]
        goal.request = req
        opts = PlanningOptions()
        opts.plan_only = False
        opts.replan = True
        goal.planning_options = opts

        send = self._move.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=30.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('move goal rejected')
            return False
        result_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=120.0)
        result = result_fut.result()
        if result is None or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = result.result.error_code.val if result else -1
            self.get_logger().error(f'move failed code={code}')
            return False
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--move', action='store_true', help='Execute pre_grasp and grasp via move_action')
    parser.add_argument('--fp-timeout', type=float, default=180.0)
    parser.add_argument('--move-action', default='/move_action')
    parser.add_argument('--move-group', default='arm')
    parser.add_argument('--ee-link', default='eef_link')
    parser.add_argument('--velocity-scale', type=float, default=0.15)
    args = parser.parse_args()

    rclpy.init()
    node = ApproachNode(args)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
