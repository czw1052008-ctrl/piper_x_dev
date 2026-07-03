/** @file vibration_planner.hpp */
#pragma once

#include <vector>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "picking_msgs/msg/detected_berry.hpp"
#include "picking_msgs/msg/vibration_plan.hpp"

namespace picking_grasp
{

class VibrationPlanner
{
public:
  struct Params
  {
    double slot_center_from_tip{0.04};
    double slot_width_m{0.04};
    double max_branch_slot_angle_deg{25.0};
    double hover_z{0.12};
    double pre_approach_dist{0.25};
    double vibration_duration_sec{3.0};
    int min_berries{2};
    double dump_lift_z{0.15};
    double table_top_z{0.31};
    double table_clearance_m{0.05};
    double rod_radius{0.008};
  };

  explicit VibrationPlanner(const Params & params);
  VibrationPlanner() : VibrationPlanner(Params()) {}

  bool plan(
    const std::vector<picking_msgs::msg::DetectedBerry> & berries,
    const geometry_msgs::msg::Vector3 & stem_direction,
    bool stem_direction_valid,
    const geometry_msgs::msg::PoseStamped & contact_pose,
    bool contact_pose_valid,
    picking_msgs::msg::VibrationPlan & plan,
    geometry_msgs::msg::PoseStamped & dump_pose,
    std::string & message) const;

private:
  Params params_;
};

}  // namespace picking_grasp
