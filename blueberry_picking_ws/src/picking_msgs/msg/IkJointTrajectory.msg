# Executor IK result for one replan of /planning/tool_trajectory_4s.
# Record this for debug: tool 6D → joint path before tracking.

std_msgs/Header header
uint64 seq
# Matches ToolTrajectory4s.seq that was solved.
uint64 tool_traj_seq

float32 horizon_s
float32 dt_s
uint8 execute_until_index

string[] joint_names
JointWaypoint[] waypoints

# Aggregate status for this solve.
bool ok
uint8 n_ok
uint8 n_fail
string detail
