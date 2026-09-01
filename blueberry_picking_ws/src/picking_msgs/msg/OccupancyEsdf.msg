# Full workspace occupancy + ESDF (base_link).
# labels: 0=UNKNOWN 1=FREE 2=OCCUPIED 3=EGO 4=BERRY
# layout: row-major i(x) + nx*(j(y) + ny*k(z)); world = origin + (i+0.5)*voxel

std_msgs/Header header
float64[3] origin_xyz
float32 voxel_m
uint32[3] size_xyz
uint8[] labels
float32[] esdf
uint32 occupied_count
uint32 free_count
