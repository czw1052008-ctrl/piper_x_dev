# Sampled-layer occupancy for the planner (same payload RViz draws).
# Uniform layer_dz_m (default 0.02 m): one contour set per occupied z slice.
# polygons always starts with the z=floor_z_m workspace disk (is_floor=true).

std_msgs/Header header
float32 layer_dz_m
float32 floor_z_m
uint32 n_polygons
OccupancyPolygon[] polygons
