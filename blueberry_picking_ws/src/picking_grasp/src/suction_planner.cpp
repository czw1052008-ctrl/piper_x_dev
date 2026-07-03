#include "picking_grasp/suction_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace picking_grasp
{

namespace
{

double selectionMetric(
  const picking_msgs::msg::DetectedBerry & berry,
  const std::string & mode,
  const std::string & nearest_frame)
{
  const auto & p = berry.pose.pose.position;
  const auto & frame = berry.pose.header.frame_id;
  if (mode == "nearest") {
    if (frame == nearest_frame || nearest_frame.empty()) {
      return p.z;  // optical frame: smaller z = closer
    }
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
  }
  return -static_cast<double>(berry.confidence);
}

geometry_msgs::msg::Quaternion quatAlignPosZTo(const double ax, const double ay, const double az)
{
  geometry_msgs::msg::Quaternion q;
  const double tx = ax;
  const double ty = ay;
  const double tz = az;
  const double len = std::sqrt(tx * tx + ty * ty + tz * tz);
  if (len < 1e-9) {
    q.w = 1.0;
    return q;
  }
  const double zx = tx / len;
  const double zy = ty / len;
  const double zz = tz / len;
  const double dot = zz;  // tool +Z dotted with target
  if (dot > 1.0 - 1e-9) {
    q.w = 1.0;
    return q;
  }
  if (dot < -1.0 + 1e-9) {
    q.x = 1.0;
    q.w = 0.0;
    return q;
  }
  const double cx = -zy;
  const double cy = zx;
  const double w = 1.0 + dot;
  const double norm = std::sqrt(cx * cx + cy * cy + w * w);
  q.x = cx / norm;
  q.y = cy / norm;
  q.z = 0.0;
  q.w = w / norm;
  return q;
}

void setPose(
  geometry_msgs::msg::PoseStamped & ps,
  const geometry_msgs::msg::PoseStamped & ref,
  double x, double y, double z,
  const geometry_msgs::msg::Quaternion & q)
{
  ps.header = ref.header;
  ps.pose.position.x = x;
  ps.pose.position.y = y;
  ps.pose.position.z = z;
  ps.pose.orientation = q;
}

}  // namespace

SuctionPlanner::SuctionPlanner(const Params & params)
: params_(params)
{
}

bool SuctionPlanner::plan(
  const std::vector<picking_msgs::msg::DetectedBerry> & berries,
  picking_msgs::msg::SuctionGraspPlan & plan,
  std::string & message) const
{
  if (berries.empty()) {
    message = "No berries to plan suction grasp";
    return false;
  }

  const auto best = std::min_element(
    berries.begin(), berries.end(),
    [this](const auto & a, const auto & b) {
      return selectionMetric(a, params_.selection_mode, params_.nearest_frame) <
             selectionMetric(b, params_.selection_mode, params_.nearest_frame);
    });

  const auto & pose = best->pose;
  plan.header = pose.header;

  const double bx = pose.pose.position.x;
  const double by = pose.pose.position.y;
  const double bz = pose.pose.position.z;

  // Eye-in-hand: in camera optical frame, (bx,by,bz) is the camera->berry vector.
  // Pick loop refines in link6 (CS7) with 8cm Y mount; this is a coarse pre-plan.
  double ax = bx;
  double ay = by;
  double az = bz;
  const double alen = std::sqrt(ax * ax + ay * ay + az * az);
  if (alen < 1e-6) {
    ax = 0.0;
    ay = 0.0;
    az = 1.0;
  } else {
    ax /= alen;
    ay /= alen;
    az /= alen;
  }

  const auto q = quatAlignPosZTo(ax, ay, az);

  const double cup = params_.cup_contact_offset_m;
  const double grasp_x = bx - ax * cup;
  const double grasp_y = by - ay * cup;
  const double grasp_z = bz - az * cup;

  const double pre_x = bx - ax * (cup + params_.pre_grasp_offset);
  const double pre_y = by - ay * (cup + params_.pre_grasp_offset);
  const double pre_z = bz - az * (cup + params_.pre_grasp_offset);

  const double post_x = bx - ax * (cup + params_.post_grasp_offset);
  const double post_y = by - ay * (cup + params_.post_grasp_offset);
  const double post_z = bz - az * (cup + params_.post_grasp_offset);

  setPose(plan.grasp, pose, grasp_x, grasp_y, grasp_z, q);
  setPose(plan.pre_grasp, pose, pre_x, pre_y, pre_z, q);
  setPose(plan.post_grasp, pose, post_x, post_y, post_z, q);

  message = "Suction plan OK (approach toward berry)";
  return true;
}

}  // namespace picking_grasp
