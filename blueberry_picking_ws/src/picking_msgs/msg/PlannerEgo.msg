# Arm / EE as ego (NOT obstacles). Sim free-space query uses complement:
#   free(p) = in_workspace(p) ∧ ¬obstacles ∧ ¬berries ∧ ¬ego_capsules(q)
#
# q_rad is enough to rebuild arm capsules via FK; tip_* are cached conveniences.

float64[6] q_rad
float64[6] qd_rad

# Tool tip in base_link + approach (+Z of tool) at this replan stamp.
float64[3] tip_xyz
float64[3] approach_axis
float32 ee_radius_m

# Coarse link origins after each joint (6×3 flattened row-major) for arm body
# exclusion / self-collision. link_radius_m ≈ capsule radius for all links.
float64[18] link_xyz
float32 link_radius_m

# Workspace: upper hemisphere in base_link (|p-center|<=radius, z>=z_min).
# workspace_aabb is the enclosing box of that hemisphere (compat / viz).
float64[6] workspace_aabb
float64[3] workspace_center
float32 workspace_radius_m
float32 workspace_z_min
