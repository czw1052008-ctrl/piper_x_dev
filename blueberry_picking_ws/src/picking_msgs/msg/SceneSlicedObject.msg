# One semantic instance lifted to stacked XY polygons (P3 world object).
# class_id matches DINOv3: 1 berry, 2 branch, 3 rigid, 4 ego; 5 = residual depth.

uint8 CLASS_BERRY=1
uint8 CLASS_BRANCH=2
uint8 CLASS_RIGID=3
uint8 CLASS_EGO=4
uint8 CLASS_RESIDUAL=5

int32 id
uint8 class_id
int32 n_pts
float32 xyz_std
float64[3] centroid_xyz
string source
bool visible_wrist
int8 pick_role
float32 fuse_delta_m
SceneZSlice[] slices
