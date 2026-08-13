#!/usr/bin/env bash
# Snapshot fixed/wrist/global_viz/fine_viz for visual QA.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-snap}"
OUT="${ROOT}/log/real_robot/qa/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
export PATH=/usr/bin:/bin
source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"
export OUT
timeout 25 /usr/bin/python3 - <<'PY'
import os, time, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import numpy as np, cv2
out = os.environ['OUT']
rclpy.init()
node = rclpy.create_node('qa_snap_only')
got = {}

def make_cb(key):
    def cb(msg):
        enc = msg.encoding.lower()
        if enc == 'rgb8':
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        elif enc == 'bgr8':
            bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            rgb = bgr[:, :, ::-1].copy()
        elif enc in ('yuv422_yuy2', 'yuyv', 'yuyv422'):
            yuyv = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 2)
            rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
        else:
            return
        got[key] = rgb
    return cb

topics = [
    ('/perception/global/detection_viz', 'global_viz'),
    ('/perception/fine/detection_viz', 'fine_viz'),
    ('/camera_fixed/color/image_raw', 'fixed'),
    ('/camera_wrist/color/image_raw', 'wrist'),
]
for t, k in topics:
    node.create_subscription(Image, t, make_cb(k), qos_profile_sensor_data)
t0 = time.time()
while time.time() - t0 < 15 and len(got) < 4:
    rclpy.spin_once(node, timeout_sec=0.2)
for k, v in got.items():
    path = os.path.join(out, f'{k}.png')
    cv2.imwrite(path, cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    print('saved', path, v.shape)
print('got', sorted(got))
open(os.path.join(out, 'keys.txt'), 'w').write(','.join(sorted(got)))
node.destroy_node()
rclpy.shutdown()
PY
echo "OUT=$OUT"
ls -la "$OUT"
