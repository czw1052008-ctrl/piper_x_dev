#include "picking_grasp/vibration_planner.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace picking_grasp
{

namespace
{

Eigen::Vector3d toVec3(const geometry_msgs::msg::Point & p)
{
  return {p.x, p.y, p.z};
}

Eigen::Vector3d toVec3(const geometry_msgs::msg::Vector3 & v)
{
  return {v.x, v.y, v.z};
}

geometry_msgs::msg::Quaternion quatFromAxes(
  const Eigen::Vector3d & x_axis,
  const Eigen::Vector3d & y_axis,
  const Eigen::Vector3d & z_axis)
{
  Eigen::Matrix3d R;
  R.col(0) = x_axis.normalized();
  R.col(1) = y_axis.normalized();
  R.col(2) = z_axis.normalized();
  Eigen::Quaterniond q(R);
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

bool isDuplicateDirection(const Eigen::Vector3d & d, const std::vector<Eigen::Vector3d> & existing)
{
  for (const auto & e : existing) {
    if (d.dot(e) > 0.92) {
      return true;
    }
  }
  return false;
}

void addDirection(std::vector<Eigen::Vector3d> & out, const Eigen::Vector3d & d)
{
  if (d.norm() < 0.2) {
    return;
  }
  Eigen::Vector3d n = d.normalized();
  if (!isDuplicateDirection(n, out)) {
    out.push_back(n);
  }
}

std::vector<Eigen::Vector3d> generateApproachAxes(
  const Eigen::Vector3d & rod_axis,
  const Eigen::Vector3d & tip_at_slot)
{
  std::vector<Eigen::Vector3d> axes;
  const Eigen::Vector3d uz(0, 0, 1);

  Eigen::Vector3d horiz_to_arm = -tip_at_slot;
  horiz_to_arm.z() = 0.0;
  if (horiz_to_arm.norm() > 0.05) {
    horiz_to_arm.normalize();
    addDirection(axes, horiz_to_arm);
    addDirection(axes, (uz + horiz_to_arm).normalized());
    addDirection(axes, (uz * 0.7 + horiz_to_arm * 0.3).normalized());
  }

  addDirection(axes, uz);

  Eigen::Vector3d rod_horiz = rod_axis;
  rod_horiz.z() = 0.0;
  if (rod_horiz.norm() > 0.05) {
    rod_horiz.normalize();
    Eigen::Vector3d perp = rod_horiz.cross(uz);
    if (perp.norm() > 0.05) {
      perp.normalize();
      addDirection(axes, perp);
      addDirection(axes, -perp);
    }
  }

  addDirection(axes, Eigen::Vector3d(1, 0, 0));
  addDirection(axes, Eigen::Vector3d(-1, 0, 0));
  addDirection(axes, Eigen::Vector3d(0, 1, 0));
  addDirection(axes, Eigen::Vector3d(0, -1, 0));

  return axes;
}

double segmentTrunkClearance(
  const Eigen::Vector3d & a, const Eigen::Vector3d & b,
  double trunk_x = 0.45, double trunk_y = 0.0, double trunk_r = 0.012)
{
  double min_clear = 1e9;
  for (int i = 0; i <= 8; ++i) {
    const double t = static_cast<double>(i) / 8.0;
    const Eigen::Vector3d p = a + t * (b - a);
    const double dist_xy = std::hypot(p.x() - trunk_x, p.y() - trunk_y);
    min_clear = std::min(min_clear, dist_xy - trunk_r);
  }
  return min_clear;
}

double scoreApproachAxis(
  const Eigen::Vector3d & d,
  const Eigen::Vector3d & pre_pos,
  const Eigen::Vector3d & tip_at_slot,
  const VibrationPlanner::Params & params)
{
  double score = 0.0;
  if (d.z() > 0.15) {
    score += 4.0;
  } else if (d.z() > 0.0) {
    score += 2.0;
  }
  if (d.z() < -0.1) {
    score -= 6.0;
  }

  const double min_z = params.table_top_z + params.table_clearance_m;
  if (pre_pos.z() > min_z + 0.04) {
    score += 2.0;
  }

  score += 1.0 / (tip_at_slot.norm() + 0.15);

  const double trunk_clear = segmentTrunkClearance(pre_pos, tip_at_slot);
  if (trunk_clear < 0.0) {
    score -= 12.0;
  } else if (trunk_clear < 0.025) {
    score -= 5.0;
  } else {
    score += trunk_clear * 8.0;
  }

  // No penalty for d not perpendicular to rod — branch may enter slot at an angle.
  (void)params;
  return score;
}

}  // namespace

VibrationPlanner::VibrationPlanner(const Params & params)
: params_(params)
{
}

bool VibrationPlanner::plan(
  const std::vector<picking_msgs::msg::DetectedBerry> & berries,
  const geometry_msgs::msg::Vector3 & stem_direction,
  const bool stem_direction_valid,
  const geometry_msgs::msg::PoseStamped & contact_pose,
  const bool contact_pose_valid,
  picking_msgs::msg::VibrationPlan & plan,
  geometry_msgs::msg::PoseStamped & dump_pose,
  std::string & message) const
{
  if (static_cast<int>(berries.size()) < params_.min_berries) {
    message = "Need at least " + std::to_string(params_.min_berries) + " berries in cluster";
    return false;
  }
  if (!contact_pose_valid) {
    message = "Branch-cluster contact_pose required from perception (no centroid guess)";
    return false;
  }
  if (!stem_direction_valid) {
    message = "stem_direction required from perception (no PCA / berry layout guess)";
    return false;
  }

  Eigen::Vector3d branch_dir = toVec3(stem_direction);
  if (branch_dir.norm() < 0.5) {
    message = "Invalid stem_direction from perception";
    return false;
  }
  branch_dir.normalize();

  Eigen::Vector3d slot_center = toVec3(contact_pose.pose.position);
  const Eigen::Vector3d to_base = -slot_center;
  if (to_base.norm() > 0.05 && branch_dir.dot(to_base.normalized()) < 0.0) {
    branch_dir = -branch_dir;
  }
  const Eigen::Vector3d rod_axis = branch_dir;

  plan.header = contact_pose.header;
  plan.branch_dir.x = branch_dir.x();
  plan.branch_dir.y = branch_dir.y();
  plan.branch_dir.z = branch_dir.z();
  plan.perception_valid = true;
  plan.vibration_duration_sec = static_cast<float>(params_.vibration_duration_sec);

  auto makePose = [&](const Eigen::Vector3d & pos, const Eigen::Vector3d & z_axis_in) {
      const Eigen::Vector3d z = z_axis_in.normalized();
      const Eigen::Vector3d ref = (std::abs(z.dot(Eigen::Vector3d::UnitZ())) < 0.9)
        ? Eigen::Vector3d::UnitZ()
        : Eigen::Vector3d::UnitX();
      const Eigen::Vector3d x = (ref - ref.dot(z) * ref).normalized();
      const Eigen::Vector3d y = z.cross(x);
      geometry_msgs::msg::PoseStamped ps;
      ps.header = plan.header;
      ps.pose.position.x = pos.x();
      ps.pose.position.y = pos.y();
      ps.pose.position.z = pos.z();
      ps.pose.orientation = quatFromAxes(x, y, z);
      return ps;
    };

  const double min_slot_z = params_.table_top_z + params_.table_clearance_m;
  const double slot_z_before = slot_center.z();
  if (slot_center.z() < min_slot_z) {
    slot_center.z() = min_slot_z;
  }

  // eef tip = contact + slot_center_from_tip * rod_axis (rod +Z toward robot).
  // Physical slot center sits slot_center_from_tip back from tip → aligns with branch contact.
  const Eigen::Vector3d tip_at_slot = slot_center + params_.slot_center_from_tip * rod_axis;
  plan.slot_pose = makePose(tip_at_slot, rod_axis);

  struct ScoredCandidate
  {
    geometry_msgs::msg::PoseStamped pose;
    double score;
    Eigen::Vector3d axis;
  };

  std::vector<ScoredCandidate> scored;
  const auto approach_axes = generateApproachAxes(rod_axis, tip_at_slot);
  for (const auto & d : approach_axes) {
    const Eigen::Vector3d pre_pos = tip_at_slot + params_.pre_approach_dist * d;
    if (pre_pos.z() < min_slot_z) {
      continue;
    }
    ScoredCandidate c;
    c.pose = makePose(pre_pos, rod_axis);
    c.score = scoreApproachAxis(d, pre_pos, tip_at_slot, params_);
    c.axis = d;
    scored.push_back(c);
  }

  if (scored.empty()) {
    message = "No valid approach axis (all below table clearance)";
    return false;
  }

  std::sort(scored.begin(), scored.end(), [](const ScoredCandidate & a, const ScoredCandidate & b) {
      return a.score > b.score;
    });

  plan.pre_approach_candidates.clear();
  plan.pre_approach_candidates.reserve(scored.size());
  for (const auto & c : scored) {
    plan.pre_approach_candidates.push_back(c.pose);
  }
  plan.pre_approach = plan.pre_approach_candidates.front();
  plan.retract_pose = plan.pre_approach;

  Eigen::Vector3d dump_tip = tip_at_slot;
  dump_tip.z() = std::max(
    tip_at_slot.z() + params_.dump_lift_z,
    params_.table_top_z + params_.table_clearance_m + params_.rod_radius);
  dump_pose = makePose(dump_tip, rod_axis);
  plan.dump_pose = dump_pose;

  std::ostringstream oss;
  oss << "Vibration plan OK (berries=" << berries.size()
      << ") contact=(" << contact_pose.pose.position.x << ","
      << contact_pose.pose.position.y << "," << contact_pose.pose.position.z << ")"
      << " branch_dir=(" << branch_dir.x() << "," << branch_dir.y()
      << "," << branch_dir.z() << ")"
      << " approach_axes=" << scored.size()
      << " slot_width_m=" << params_.slot_width_m
      << " max_slot_angle_deg=" << params_.max_branch_slot_angle_deg
      << " slot_z_clamped=" << (slot_z_before < min_slot_z ? "yes" : "no");
  message = oss.str();
  return true;
}

}  // namespace picking_grasp
