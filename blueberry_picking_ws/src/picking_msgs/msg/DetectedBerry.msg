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
