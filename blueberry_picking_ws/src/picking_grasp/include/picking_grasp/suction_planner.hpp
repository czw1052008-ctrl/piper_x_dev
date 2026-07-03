/** @file suction_planner.hpp */
#pragma once

#include <vector>
#include "picking_msgs/msg/detected_berry.hpp"
#include "picking_msgs/msg/suction_grasp_plan.hpp"

namespace picking_grasp
{

class SuctionPlanner
{
public:
    struct Params
    {
        double berry_radius{0.0075};
        /** Distance from eef_link origin to cup mouth along -eef_Z (suction_eef.urdf.xacro). */
        double cup_contact_offset_m{0.04};
        double pre_grasp_offset{0.15};
        double post_grasp_offset{0.12};
        std::string selection_mode{"nearest"};  // nearest | highest_confidence
        std::string nearest_frame{"camera_wrist_color_optical_frame"};
    };

  explicit SuctionPlanner(const Params & params);
  SuctionPlanner() : SuctionPlanner(Params()) {}

  bool plan(
    const std::vector<picking_msgs::msg::DetectedBerry> & berries,
    picking_msgs::msg::SuctionGraspPlan & plan,
    std::string & message) const;

private:
  Params params_;
};

}  // namespace picking_grasp
