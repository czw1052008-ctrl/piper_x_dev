#!/usr/bin/env python3
"""Verify FoundationPose fine detection (weights + optional live Gazebo service)."""

from __future__ import annotations

import argparse
import os
import sys
import time


def _ws_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _setup_ros_pythonpath() -> None:
    """Allow running without manually sourcing install/setup.bash."""
    ws = _ws_root()
    candidates = [
        os.path.join(ws, 'install', 'picking_msgs', 'lib', 'python3.12', 'site-packages'),
        os.path.join(ws, 'install', 'picking_perception', 'lib', 'python3.12', 'site-packages'),
        '/opt/ros/jazzy/lib/python3.12/site-packages',
        '/opt/ros/humble/lib/python3.10/site-packages',
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def check_weights(fp_root: str) -> bool:
    ok = True
    for stamp in ('2024-01-11-20-02-45', '2023-10-28-18-33-37'):
        ckpt = os.path.join(fp_root, 'weights', stamp, 'model_best.pth')
        if os.path.isfile(ckpt):
            print(f'weights OK: {ckpt}')
        else:
            print(f'weights MISSING: {ckpt}')
            ok = False
    return ok


def check_wrapper(fp_root: str, mesh_path: str) -> bool:
    sys.path.insert(0, os.path.join(_repo_root(), 'blueberry_picking_ws', 'src'))
    from picking_perception.foundation_pose_wrapper import FoundationPoseWrapper

    fp = FoundationPoseWrapper(mesh_path=mesh_path, foundation_pose_root=fp_root)
    if not fp.ready:
        print('FoundationPoseWrapper not ready (torch/GPU/deps missing)')
        return False
    print('FoundationPoseWrapper initialized')
    return True


def check_ros_service(timeout_sec: float) -> bool:
    _setup_ros_pythonpath()
    import rclpy
    from picking_msgs.srv import TriggerFineDetection

    rclpy.init()
    node = rclpy.create_node('verify_fine_detection')
    client = node.create_client(TriggerFineDetection, 'trigger_fine_detection')
    if not client.wait_for_service(timeout_sec=timeout_sec):
        print('trigger_fine_detection service not available')
        node.destroy_node()
        rclpy.shutdown()
        return False

    req = TriggerFineDetection.Request()
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done():
        print('Fine detection call timed out')
        node.destroy_node()
        rclpy.shutdown()
        return False

    res = future.result()
    _print_detected_berries(res)
    node.destroy_node()
    rclpy.shutdown()
    return bool(res.success)


def _print_detected_berries(res) -> None:
    print(f'\n=== trigger_fine_detection (base_link) ===')
    print(f'success={res.success}  message={res.message!r}  count={len(res.detected_berries)}')
    for i, berry in enumerate(res.detected_berries):
        p = berry.pose.pose.position
        q = berry.pose.pose.orientation
        frame = berry.pose.header.frame_id or berry.header.frame_id or 'base_link'
        print(
            f'  [{i + 1}] frame={frame}  '
            f'x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}  '
            f'qx={q.x:.4f}  qy={q.y:.4f}  qz={q.z:.4f}  qw={q.w:.4f}  '
            f'confidence={berry.confidence:.3f}')
    if not res.detected_berries:
        print('  (no berries)')
    print('==========================================\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--foundation-pose-root',
        default=os.environ.get('FOUNDATIONPOSE_ROOT', os.path.join(_repo_root(), 'FoundationPose')),
    )
    parser.add_argument(
        '--mesh',
        default=os.path.join(
            _repo_root(),
            'blueberry_picking_ws',
            'src',
            'picking_perception',
            'meshes',
            'blueberry.obj',
        ),
    )
    parser.add_argument('--skip-wrapper', action='store_true')
    parser.add_argument('--ros-service', action='store_true', help='Call live trigger_fine_detection')
    parser.add_argument('--service-timeout', type=float, default=120.0)
    args = parser.parse_args()

    if not check_weights(args.foundation_pose_root):
        print('Run: bash blueberry_picking_ws/scripts/download_foundationpose_weights.sh')
        return 1

    if not args.skip_wrapper:
        if not check_wrapper(args.foundation_pose_root, args.mesh):
            return 1

    if args.ros_service:
        time.sleep(0.5)
        if not check_ros_service(args.service_timeout):
            return 1

    print('Fine detection verification passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
