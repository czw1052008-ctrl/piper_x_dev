# One pickable berry — encoding-friendly for trajectory planner.
#
# Minimal token fields the model should see:
#   position (3) + surface_normal (3) + bbox (4) + center_depth + conf + visibility
# Missing geometric fields use sentinels: depth=-1, bbox=-1, normal_valid=false.

int32 id
float64[3] position
float32 confidence
bool visible_wrist
bool visible_global

# Image-plane bbox (pixels in source camera: wrist if visible_wrist). -1 = unknown.
float32 bbox_u0
float32 bbox_v0
float32 bbox_u1
float32 bbox_v1
float32 image_u
float32 image_v

# Center depth along camera optical axis (m). Prefer RGB-D; -1 = unknown.
float32 center_depth_m
string depth_mode
float32 z_depth_m
float32 z_mono_m

# Local fruit surface normal in base_link (unit). Convention: outward,
# flipped toward camera / approach side. Zero vector if normal_valid=false.
float64[3] surface_normal
bool normal_valid
# Track provenance for bag / debug: "live" (detector update) | "coast" (world hold)
string track_source

# Pick queue role (rules/FSM set; model does not choose).
# PENDING = 待采集；ACTIVE = 当前采集目标（接近/触达只跟这颗）；DONE = 本轮已采。
uint8 PICK_ROLE_PENDING=0
uint8 PICK_ROLE_ACTIVE=1
uint8 PICK_ROLE_DONE=2
uint8 pick_role
