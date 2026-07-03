#include <cmath>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "picking_grasp/vibration_planner.hpp"
#include "picking_msgs/msg/detected_berry.hpp"

namespace
{

constexpr double kPreApproachDist = 0.25;
constexpr double kSlotFromTip = 0.04;

double dist3(
  const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
{
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::vector<picking_msgs::msg::DetectedBerry> tipClusterBerries()
{
  const std::vector<std::tuple<double, double, double>> fruit = {
    {0.305, 0.020, 0.545},
    {0.298, 0.032, 0.540},
    {0.292, 0.015, 0.550},
    {0.287, 0.028, 0.542},
  };
  std::vector<picking_msgs::msg::DetectedBerry> berries;
  for (const auto & [x, y, z] : fruit) {
    picking_msgs::msg::DetectedBerry b;
    b.pose.header.frame_id = "base_link";
    b.pose.pose.position.x = x;
    b.pose.pose.position.y = y;
    b.pose.pose.position.z = z;
    b.confidence = 0.9f;
    berries.push_back(b);
  }
  return berries;
}

}  // namespace

TEST(VibrationPlanner, ContactPoseDrivesSlotNotBerryCentroid)
{
  picking_grasp::VibrationPlanner planner;
  const auto berries = tipClusterBerries();

  geometry_msgs::msg::PoseStamped contact;
  contact.header.frame_id = "base_link";
  contact.pose.position.x = 0.305;
  contact.pose.position.y = 0.020;
  contact.pose.position.z = 0.545;

  geometry_msgs::msg::Vector3 stem;
  stem.x = -0.945f;
  stem.y = 0.167f;
  stem.z = 0.278f;

  picking_msgs::msg::VibrationPlan plan;
  geometry_msgs::msg::PoseStamped dump;
  std::string msg;
  ASSERT_TRUE(planner.plan(
    berries, stem, true, contact, true, plan, dump, msg)) << msg;

  EXPECT_TRUE(plan.perception_valid);
  EXPECT_GE(plan.pre_approach_candidates.size(), 3u);

  // Each candidate is collinear with slot: pre = slot + 0.25m × approach_axis.
  for (const auto & cand : plan.pre_approach_candidates) {
    const double d = dist3(cand.pose.position, plan.slot_pose.pose.position);
    EXPECT_NEAR(d, kPreApproachDist, 0.02);
  }

  const double lateral = dist3(plan.pre_approach.pose.position, plan.slot_pose.pose.position);
  EXPECT_NEAR(lateral, kPreApproachDist, 0.02);

  // slot tip is offset back along rod axis from contact (toward robot).
  const double slot_from_contact = dist3(plan.slot_pose.pose.position, contact.pose.position);
  EXPECT_NEAR(slot_from_contact, kSlotFromTip, 0.02);
}

TEST(VibrationPlanner, MissingContactFails)
{
  picking_grasp::VibrationPlanner planner;
  geometry_msgs::msg::Vector3 stem;
  stem.x = 1.0f;
  geometry_msgs::msg::PoseStamped contact;
  picking_msgs::msg::VibrationPlan plan;
  geometry_msgs::msg::PoseStamped dump;
  std::string msg;
  EXPECT_FALSE(planner.plan(
    tipClusterBerries(), stem, true, contact, false, plan, dump, msg));
  EXPECT_NE(msg.find("contact_pose"), std::string::npos);
}

TEST(VibrationPlanner, MissingStemFails)
{
  picking_grasp::VibrationPlanner planner;
  geometry_msgs::msg::Vector3 stem;
  geometry_msgs::msg::PoseStamped contact;
  contact.header.frame_id = "base_link";
  contact.pose.position.x = 0.3;
  picking_msgs::msg::VibrationPlan plan;
  geometry_msgs::msg::PoseStamped dump;
  std::string msg;
  EXPECT_FALSE(planner.plan(
    tipClusterBerries(), stem, false, contact, true, plan, dump, msg));
  EXPECT_NE(msg.find("stem_direction"), std::string::npos);
}
