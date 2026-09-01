# Trajectory executor status (debug + FSM gate).

std_msgs/Header header
uint64 seq
uint64 tool_traj_seq
uint64 ik_traj_seq

# status: ok | ik_fail | jump_reject | track_timeout | idle | blending
string status
string detail

# Tracking residual at last tick (m / rad).
float64 tip_err_m
float64 axis_err_rad
float64 max_joint_err_rad
