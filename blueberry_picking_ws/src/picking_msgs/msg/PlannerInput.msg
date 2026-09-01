# Full structured input for pick trajectory planner @ 1 Hz replan.
std_msgs/Header header
uint64 seq

SceneCluster[] clusters
SceneBerry[] berries
SceneObstacle[] obstacles   # DEPRECATED for freespace; prefer occupancy_local

OccupancyLocal occupancy_local

PlannerEgo ego
PlannerTaskContext task_context
PlannerEndEffector end_effector

# Bitmask of missing optional fields (debug); 0 = assembler considers input complete enough.
uint32 input_flags
