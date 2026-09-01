# Local occupancy/ESDF crop for planner model input (tip/fruit neighborhood).
# Same label/ESDF semantics as OccupancyEsdf; typically size 32^3.

std_msgs/Header header
float64[3] origin_xyz
float32 voxel_m
uint32[3] size_xyz
uint8[] labels
float32[] esdf
float64[3] crop_center_xyz
