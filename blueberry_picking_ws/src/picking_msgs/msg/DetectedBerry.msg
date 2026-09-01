std_msgs/Header header
geometry_msgs/PoseStamped pose
float32 confidence
# BoT-SORT / MOT track id; -1 = none (e.g. global detect-only).
int32 track_id
# Wrist RGB-D depth diagnostics (fine detector).
# depth_mode: depth | mono | coast | iou_pin | base_coast | hsv | ""
# z_*_m = -1.0 when that source is unavailable.
string depth_mode
float32 z_depth_m
float32 z_mono_m
# Image-plane bbox center (pixels in wrist RGB). -1 = unknown.
float32 image_u
float32 image_v
# Unified YOLO: 0=berry, 1=cluster, -1=unknown
int32 class_id
# Fixed-camera / wrist bbox (pixels). -1 = unknown.
float32 bbox_u0
float32 bbox_v0
float32 bbox_u1
float32 bbox_v1
# Preferred center depth along camera Z (m). Prefer z_depth_m; -1 unknown.
float32 center_depth_m
# Local surface normal in base_link (unit). Outward / toward camera.
# Zero + normal_valid=false when depth plane fit fails.
float64[3] surface_normal_base
bool normal_valid
