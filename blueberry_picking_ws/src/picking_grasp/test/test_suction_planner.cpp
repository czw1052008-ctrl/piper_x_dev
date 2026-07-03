#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "picking_grasp/suction_planner.hpp"
#include "picking_msgs/msg/detected_berry.hpp"

TEST(SuctionPlanner, SelectsNearestByDepth)
{
  picking_grasp::SuctionPlanner planner;
  std::vector<picking_msgs::msg::DetectedBerry> berries(2);
  berries[0].confidence = 0.9f;
  berries[0].pose.header.frame_id = "base_link";
  berries[0].pose.pose.position.x = 0.3;
  berries[0].pose.pose.position.y = 0.0;
  berries[0].pose.pose.position.z = 0.5;
  berries[1].confidence = 0.3f;
  berries[1].pose.header.frame_id = "base_link";
  berries[1].pose.pose.position.x = 0.3;
  berries[1].pose.pose.position.y = 0.0;
  berries[1].pose.pose.position.z = 0.35;

  picking_msgs::msg::SuctionGraspPlan plan;
  std::string msg;
  ASSERT_TRUE(planner.plan(berries, plan, msg));
  EXPECT_NEAR(plan.grasp.pose.position.z, 0.35 - 0.04 * (0.35 / std::sqrt(0.3 * 0.3 + 0.35 * 0.35)), 1e-4);
  const double dx = plan.grasp.pose.position.x - plan.pre_grasp.pose.position.x;
  const double dy = plan.grasp.pose.position.y - plan.pre_grasp.pose.position.y;
  const double dz = plan.grasp.pose.position.z - plan.pre_grasp.pose.position.z;
  EXPECT_NEAR(std::sqrt(dx * dx + dy * dy + dz * dz), 0.15, 1e-4);
  EXPECT_GT(plan.grasp.pose.orientation.w, 0.0);
}

TEST(SuctionPlanner, SelectsHighestConfidenceWhenConfigured)
{
  picking_grasp::SuctionPlanner::Params params;
  params.selection_mode = "highest_confidence";
  picking_grasp::SuctionPlanner planner(params);
  std::vector<picking_msgs::msg::DetectedBerry> berries(2);
  berries[0].confidence = 0.3f;
  berries[0].pose.pose.position.z = 0.5;
  berries[1].confidence = 0.9f;
  berries[1].pose.pose.position.z = 0.6;

  picking_msgs::msg::SuctionGraspPlan plan;
  std::string msg;
  ASSERT_TRUE(planner.plan(berries, plan, msg));
  EXPECT_NEAR(plan.grasp.pose.position.z, 0.6 - 0.04, 1e-4);
}
