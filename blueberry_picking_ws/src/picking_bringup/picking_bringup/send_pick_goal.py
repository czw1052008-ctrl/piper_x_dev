#!/usr/bin/env python3
"""Send a PickBlueberry action goal for integration testing."""

import sys
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from picking_msgs.action import PickBlueberry
from rclpy.action import ActionClient
from rclpy.node import Node


class SendPickGoal(Node):
    def __init__(self):
        super().__init__('send_pick_goal')
        self.declare_parameter('end_effector_mode', 'suction')
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('controller_wait_sec', 90.0)
        self.declare_parameter('goal_timeout_sec', 120.0)
        self._client = ActionClient(self, PickBlueberry, 'pick_blueberry')
        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

    def _pick_server_process_count(self) -> int:
        import subprocess

        try:
            out = subprocess.check_output(
                ['pgrep', '-af', 'lib/picking_task/pick_action_server'],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, OSError, ValueError):
            return 0
        count = 0
        for line in out.splitlines():
            if not line.strip():
                continue
            if 'pgrep' in line or 'send_pick_goal' in line:
                continue
            if 'lib/picking_task/pick_action_server' in line:
                count += 1
        return count

    def _duplicate_action_servers(self) -> bool:
        proc_count = self._pick_server_process_count()
        if proc_count > 1:
            self.get_logger().error(
                f'Found {proc_count} pick_action_server processes. '
                'Stop the old launch (Ctrl+C) then: bash scripts/kill_stale_nodes.sh')
            return True
        return False

    def _wait_for_arm_controller(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self._arm_client.wait_for_server(timeout_sec=1.0):
                self.get_logger().info('arm_controller is ready')
                return True
            self.get_logger().info('Waiting for arm_controller/follow_joint_trajectory ...')
        return False

    def run(self):
        wait_sec = float(self.get_parameter('controller_wait_sec').value)
        if not self._wait_for_arm_controller(wait_sec):
            self.get_logger().error('arm_controller not available')
            return 1

        if not self._client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('pick_blueberry action server not available')
            return 1

        if self._duplicate_action_servers():
            return 1

        goal = PickBlueberry.Goal()
        goal.end_effector_mode = self.get_parameter('end_effector_mode').value
        goal.max_retries = int(self.get_parameter('max_retries').value)

        goal_timeout = float(self.get_parameter('goal_timeout_sec').value)
        self.get_logger().info(
            f'[pick_pipeline:pick_goal] sending mode={goal.end_effector_mode} '
            f'max_retries={goal.max_retries} timeout_sec={goal_timeout:.0f}')

        send = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        goal_handle = send.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return 1

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + goal_timeout
        last_progress = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=1.0)
            if result_future.done():
                break
            if time.monotonic() >= deadline:
                self.get_logger().error(
                    f'[pick_pipeline:pick_goal] timeout after {goal_timeout:.0f}s')
                goal_handle.cancel_goal_async()
                return 1
            if time.monotonic() - last_progress >= 30.0:
                elapsed = goal_timeout - (deadline - time.monotonic())
                self.get_logger().info(
                    f'[pick_pipeline:pick_goal] waiting elapsed={elapsed:.0f}s')
                last_progress = time.monotonic()

        result = result_future.result().result
        self.get_logger().info(
            f'[pick_pipeline:pick_result] success={result.success} '
            f'message={result.message} picked={result.total_picked}')
        return 0 if result.success else 1


def main():
    rclpy.init(args=sys.argv)
    node = SendPickGoal()
    code = node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
