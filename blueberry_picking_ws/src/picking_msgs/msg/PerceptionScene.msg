# Unified perception output: tracked clusters + berries in base_link.
# Prefer this over separate /perception/global|fine/berries for planner / bag.
std_msgs/Header header
SceneCluster[] clusters
SceneBerry[] berries
# Bitmask: 1=no_clusters 2=no_berries 4=global_stale 8=fine_stale 16=has_coast
uint32 flags
