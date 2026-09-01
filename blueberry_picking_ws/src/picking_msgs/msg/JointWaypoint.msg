# One IK-solved joint sample (absolute rad).
builtin_interfaces/Duration time_from_start
float64[6] q_rad
# Optional: tip FK after IK for residual debug. NaN if unused.
float64[3] tip_xyz
float64[3] tip_approach
bool ik_ok
string ik_reason
