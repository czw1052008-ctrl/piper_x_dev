#!/usr/bin/env python3
"""Capture entry-pose wrist image for manual target bbox labeling.

  python3 scripts/prepare_entry_bbox_label.py --restore

Writes log/real_robot/qa/<session>/entry_label_wrist.png and target_bbox_request.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore', action='store_true')
    p.add_argument('--traj-s', type=float, default=6.0)
    p.add_argument('--settle-s', type=float, default=2.0)
    p.add_argument('--wait-s', type=float, default=6.0)
    args = p.parse_args()

    if args.restore:
        subprocess.run(
            [sys.executable, str(SCRIPTS / 'refine_entry_pose.py'), 'restore',
             '--traj-s', str(args.traj_s), '--settle-s', str(args.settle_s)],
            cwd=str(ROOT), check=False)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = ROOT / 'log' / 'real_robot' / 'qa' / session
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Node('prepare_entry_bbox_label')
    state = {'rgb': None, 'K': None}

    def on_rgb(msg: Image) -> None:
        try:
            from cv_bridge import CvBridge
            state['rgb'] = CvBridge().imgmsg_to_cv2(msg, 'rgb8')
        except Exception:
            pass

    def on_info(msg: CameraInfo) -> None:
        k = msg.k
        if len(k) >= 9:
            state['K'] = [float(k[0]), float(k[4]), float(k[2]), float(k[5])]
            state['wh'] = [int(msg.width), int(msg.height)]

    node.create_subscription(Image, '/camera_wrist/color/image_raw', on_rgb,
                           qos_profile_sensor_data)
    node.create_subscription(CameraInfo, '/camera_wrist/color/camera_info', on_info, 10)

    t0 = time.time()
    while time.time() - t0 < args.wait_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        if state['rgb'] is not None and state.get('K'):
            break

    node.destroy_node()
    rclpy.shutdown()

    if state['rgb'] is None:
        print('ERROR: no wrist RGB', file=sys.stderr)
        return 2

    import cv2  # type: ignore
    img_path = out_dir / 'entry_label_wrist.png'
    cv2.imwrite(str(img_path), cv2.cvtColor(state['rgb'], cv2.COLOR_RGB2BGR))

    req = {
        'session': session,
        'image': 'entry_label_wrist.png',
        'image_size_wh': state.get('wh'),
        'intrinsics_fx_fy_cx_cy': state.get('K'),
        'instructions': (
            'Label the contact berry in entry_label_wrist.png. '
            'BBox pixels (x0,y0,x1,y1) top-left origin. Then run:\n'
            f'  python3 scripts/write_target_bbox_decision.py '
            f'--session-dir log/real_robot/qa/{session} '
            f'--bbox X0 Y0 X1 Y1 --also-global'
        ),
    }
    (out_dir / 'target_bbox_request.json').write_text(
        json.dumps(req, indent=2), encoding='utf-8')

    print(f'session={session}')
    print(f'image={img_path}')
    print(f'request={out_dir / "target_bbox_request.json"}')
    print('请在图上标出接触果的框，把 x0 y0 x1 y1 像素坐标发给我。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
