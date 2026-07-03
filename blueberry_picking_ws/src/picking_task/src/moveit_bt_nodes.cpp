#include "picking_task/moveit_bt_nodes.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <Eigen/Dense>

#include <behaviortree_cpp/action_node.h>
#include <behaviortree_cpp/bt_factory.h>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <rclcpp/rclcpp.hpp>

#include "picking_msgs/msg/vibration_plan.hpp"
#include "picking_task/pipeline_log.hpp"

namespace picking_task
{

namespace
{

// Must match vibration_eef.urdf.xacro: link6 → flange(3mm) → motor(10mm) → rod(300mm) → eef.
constexpr double kFlangeThicknessM = 0.003;
constexpr double kMotorZM = 0.010;
constexpr double kRodSpanM = 0.30;
constexpr double kLink6ToRodOriginM = kFlangeThicknessM + kMotorZM;
constexpr double kLink6ToEefTipM = kLink6ToRodOriginM + kRodSpanM;
constexpr double kBoltFromTipNearM = 0.02;
constexpr double kBoltFromTipFarM = 0.06;
constexpr double kSlotCenterFromTipM = 0.5 * (kBoltFromTipNearM + kBoltFromTipFarM);
constexpr double kSlotCenterFromRodOriginM = kRodSpanM - kSlotCenterFromTipM;
constexpr double kLink6ToSlotCenterM = kLink6ToRodOriginM + kSlotCenterFromRodOriginM;
// TRAC-IK / MoveIt arm group tip (piper_vibration.srdf); Cartesian IK only works on this link.
constexpr const char * kMotionTipLink = "link6";
// Gazebo sim: compress MoveIt time-parameterized trajectories so pick fits in 120s wall/sim budget.
constexpr double kSimTrajectoryTimeScale = 0.28;
// blueberry_plant model in blueberry_picking.sdf (base_link / world-aligned).
constexpr double kPlantOriginX = 0.45;
constexpr double kPlantOriginY = 0.0;
constexpr double kPlantOriginZ = 0.41;
constexpr double kTrunkLocalZ = 0.26;
constexpr double kTrunkRadius = 0.012;
constexpr double kTrunkHeight = 0.28;
constexpr double kTrunkAvoidRadius = 0.035;

void scalePlanTimestamps(
  moveit::planning_interface::MoveGroupInterface::Plan & plan, double factor)
{
  for (auto & pt : plan.trajectory.joint_trajectory.points) {
    const double t = rclcpp::Duration(pt.time_from_start).seconds() * factor;
    pt.time_from_start = rclcpp::Duration::from_seconds(std::max(t, 0.001));
  }
}

const char * moveItErrorName(int code)
{
  using moveit_msgs::msg::MoveItErrorCodes;
  switch (code) {
    case MoveItErrorCodes::SUCCESS: return "SUCCESS";
    case MoveItErrorCodes::FAILURE: return "FAILURE";
    case MoveItErrorCodes::PLANNING_FAILED: return "PLANNING_FAILED";
    case MoveItErrorCodes::INVALID_MOTION_PLAN: return "INVALID_MOTION_PLAN";
    case MoveItErrorCodes::START_STATE_IN_COLLISION: return "START_STATE_IN_COLLISION";
    case MoveItErrorCodes::GOAL_IN_COLLISION: return "GOAL_IN_COLLISION";
    case MoveItErrorCodes::START_STATE_INVALID: return "START_STATE_INVALID";
    case MoveItErrorCodes::GOAL_STATE_INVALID: return "GOAL_STATE_INVALID";
    case MoveItErrorCodes::UNABLE_TO_AQUIRE_SENSOR_DATA: return "UNABLE_TO_AQUIRE_SENSOR_DATA";
    case MoveItErrorCodes::TIMED_OUT: return "TIMED_OUT";
    case MoveItErrorCodes::PREEMPTED: return "PREEMPTED";
    default: return "UNKNOWN";
  }
}

void logCurrentJointState(
  const rclcpp::Logger & logger,
  const std::string & label,
  const std::map<std::string, double> & joints)
{
  std::ostringstream oss;
  oss << label << " joints:";
  for (const auto & [name, pos] : joints) {
    oss << ' ' << name << '=' << pos;
  }
  RCLCPP_INFO(logger, "%s", oss.str().c_str());
}

}  // namespace

class MoveItInterface
{
public:
  static void init(const rclcpp::Node::SharedPtr & node)
  {
    if (!instance_) {
      instance_.reset(new MoveItInterface(node));
    }
  }

  static MoveItInterface & get()
  {
    return *instance_;
  }

  std::map<std::string, double> currentJointPositions() const
  {
    std::map<std::string, double> out;
    const auto names = move_group_->getActiveJoints();
    const auto positions = move_group_->getCurrentJointValues();
    for (size_t i = 0; i < names.size() && i < positions.size(); ++i) {
      out[names[i]] = positions[i];
    }
    return out;
  }

  void ensureValidStartState()
  {
    move_group_->setStartStateToCurrentState();
    const moveit::core::RobotStatePtr state_ptr = move_group_->getCurrentState();
    if (!state_ptr) {
      RCLCPP_WARN_ONCE(
        node_->get_logger(),
        "ensureValidStartState: joint_states not ready, seeding named target 'home'");
      move_group_->setNamedTarget("home");
      return;
    }
    moveit::core::RobotState state = *state_ptr;
    state.enforceBounds();
    move_group_->setStartState(state);
  }

  bool moveHorizontalXYOmpl(double target_x, double target_y, const char * label)
  {
    ensureValidStartState();
    useMotionTipLink();
    const auto current = move_group_->getCurrentPose().pose;
    const double keep_z = current.position.z;

    RCLCPP_INFO(
      node_->get_logger(), "%s OMPL from=(%.3f, %.3f) to=(%.3f, %.3f) z=%.3f",
      label, current.position.x, current.position.y, target_x, target_y, keep_z);

    move_group_->setGoalPositionTolerance(0.03);
    move_group_->setGoalOrientationTolerance(6.28);
    move_group_->clearPoseTargets();
    move_group_->setPlanningTime(15.0);
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);
    move_group_->setPositionTarget(target_x, target_y, keep_z);

    const auto code = move_group_->move();
    const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    RCLCPP_INFO(
      node_->get_logger(), "%s -> %s (%s)",
      label, ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    if (!ok) {
      logCurrentJointState(node_->get_logger(), label, currentJointPositions());
    }
    return ok;
  }

  bool moveToNamed(const std::string & name)
  {
    move_group_->setStartStateToCurrentState();
    move_group_->setNamedTarget(name);
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);

    // Already at named target — skip (avoids START_STATE_IN_COLLISION replan from same pose).
    const auto current = move_group_->getCurrentJointValues();
    std::vector<double> target;
    move_group_->getJointValueTarget(target);
    if (target.size() == current.size()) {
      bool at_target = true;
      for (size_t i = 0; i < current.size(); ++i) {
        if (std::abs(current[i] - target[i]) > 0.02) {
          at_target = false;
          break;
        }
      }
      if (at_target) {
        RCLCPP_INFO(node_->get_logger(), "MoveToNamed('%s') -> SUCCESS (already there)", name.c_str());
        return true;
      }
    }

    const auto code = move_group_->move();
    const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    RCLCPP_INFO(
      node_->get_logger(), "MoveToNamed('%s') -> %s (%s)",
      name.c_str(), ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    if (!ok) {
      logCurrentJointState(node_->get_logger(), "MoveToNamed fail", currentJointPositions());
    }
    return ok;
  }

  bool moveToPose(
    const geometry_msgs::msg::PoseStamped & target, double offset_z,
    bool enforce_orientation, const std::string & planning_link)
  {
    auto pose = target;
    if (!pose.header.frame_id.empty() && pose.header.frame_id != "base_link") {
      RCLCPP_WARN(
        node_->get_logger(), "MoveToPose frame %s, assuming base_link",
        pose.header.frame_id.c_str());
    }
    pose.header.frame_id = "base_link";
    pose.pose.position.z += offset_z;

    RCLCPP_INFO(
      node_->get_logger(),
      "MoveToPose ee=%s target=(%.3f, %.3f, %.3f) offset_z=%.3f orient=%s",
      planning_link.c_str(),
      pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
      offset_z, enforce_orientation ? "true" : "false");

    if (planning_link == "eef_link" || planning_link == kMotionTipLink) {
      useMotionTipLink();
      if (planning_link == "eef_link") {
        pose.pose = poseEefToLink6(pose.pose);
      }
    } else {
      move_group_->setEndEffectorLink(planning_link);
    }
    move_group_->setStartStateToCurrentState();
    move_group_->setGoalPositionTolerance(0.03);
    move_group_->setGoalOrientationTolerance(enforce_orientation ? 0.5 : 6.28);
    move_group_->setPlanningTime(enforce_orientation ? 10.0 : 15.0);
    move_group_->clearPoseTargets();
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);

    moveit::core::MoveItErrorCode code = moveit::core::MoveItErrorCode::FAILURE;
    if (enforce_orientation) {
      move_group_->setPoseTarget(pose.pose);
      code = move_group_->move();
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Pose goal with orientation failed, retrying position-only on %s",
          planning_link.c_str());
        move_group_->setStartStateToCurrentState();
        move_group_->clearPoseTargets();
        move_group_->setPositionTarget(
          pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
        code = move_group_->move();
      }
    } else {
      move_group_->setPositionTarget(
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
      code = move_group_->move();
    }
    useMotionTipLink();

    const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    RCLCPP_INFO(
      node_->get_logger(), "MoveToPose -> %s (%s, code=%d)",
      ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val), code.val);
    if (!ok) {
      const auto eef = move_group_->getCurrentPose();
      RCLCPP_WARN(
        node_->get_logger(), "MoveToPose fail eef=(%.3f, %.3f, %.3f)",
        eef.pose.position.x, eef.pose.position.y, eef.pose.position.z);
      logCurrentJointState(node_->get_logger(), "MoveToPose fail", currentJointPositions());
    }
    return ok;
  }

  bool moveToPose(
    const geometry_msgs::msg::PoseStamped & target, double offset_z = 0.0,
    bool enforce_orientation = false)
  {
    return moveToPose(target, offset_z, enforce_orientation, kMotionTipLink);
  }

  bool executeCartesianWaypoints(
    const std::vector<geometry_msgs::msg::Pose> & waypoints,
    const char * label, double min_fraction = 0.85, bool avoid_collisions = true)
  {
    useMotionTipLink();
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);

    auto attempt = [&](bool check_collisions) -> bool {
        moveit_msgs::msg::RobotTrajectory trajectory;
        moveit_msgs::msg::MoveItErrorCodes error_code;
        const double fraction = move_group_->computeCartesianPath(
          waypoints, 0.008, trajectory, check_collisions, &error_code);
        if (fraction < min_fraction) {
          RCLCPP_WARN(
            node_->get_logger(), "%s path only %.0f%% (collide=%s, err=%d)",
            label, fraction * 100.0, check_collisions ? "true" : "false", error_code.val);
          return false;
        }
        const auto code = move_group_->execute(trajectory);
        const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
        RCLCPP_INFO(
          node_->get_logger(), "%s execute -> %s (%s, fraction=%.0f%%, collide=%s)",
          label, ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val), fraction * 100.0,
          check_collisions ? "true" : "false");
        return ok;
      };

    move_group_->setStartStateToCurrentState();
    if (attempt(avoid_collisions)) {
      return true;
    }
    if (avoid_collisions) {
      RCLCPP_WARN(node_->get_logger(), "%s retry without collision checks", label);
      move_group_->setStartStateToCurrentState();
      if (attempt(false)) {
        return true;
      }
    }
    logCurrentJointState(node_->get_logger(), label, currentJointPositions());
    return false;
  }

  static double posDist(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
  {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  static geometry_msgs::msg::Vector3 rodAxisWorld(
    const geometry_msgs::msg::Quaternion & q)
  {
    // R * (0,0,1) from unit quaternion
    const double qw = q.w;
    const double qx = q.x;
    const double qy = q.y;
    const double qz = q.z;
    geometry_msgs::msg::Vector3 axis;
    axis.x = 2.0 * (qx * qz + qw * qy);
    axis.y = 2.0 * (qy * qz - qw * qx);
    axis.z = 1.0 - 2.0 * (qx * qx + qy * qy);
    return axis;
  }

  static geometry_msgs::msg::Point offsetAlongRod(
    const geometry_msgs::msg::Point & origin,
    const geometry_msgs::msg::Quaternion & q, double distance_m)
  {
    const auto axis = rodAxisWorld(q);
    geometry_msgs::msg::Point out = origin;
    out.x += distance_m * axis.x;
    out.y += distance_m * axis.y;
    out.z += distance_m * axis.z;
    return out;
  }

  /** Rod tip pose (planner eef / slot_pose) → link6 pose for MoveIt arm tip. */
  static geometry_msgs::msg::Pose link6PoseFromRodTip(const geometry_msgs::msg::Pose & rod_tip)
  {
    geometry_msgs::msg::Pose out = rod_tip;
    out.position = offsetAlongRod(rod_tip.position, rod_tip.orientation, -kLink6ToEefTipM);
    return out;
  }

  static geometry_msgs::msg::Point slotCenterFromLink6(const geometry_msgs::msg::Pose & link6)
  {
    return offsetAlongRod(link6.position, link6.orientation, kLink6ToSlotCenterM);
  }

  static geometry_msgs::msg::Point rodTipFromLink6(const geometry_msgs::msg::Pose & link6)
  {
    return offsetAlongRod(link6.position, link6.orientation, kLink6ToEefTipM);
  }

  static geometry_msgs::msg::Pose poseEefToLink6(const geometry_msgs::msg::Pose & eef)
  {
    return link6PoseFromRodTip(eef);
  }

  void useMotionTipLink()
  {
    move_group_->setEndEffectorLink(kMotionTipLink);
  }

  void setFastMotionScaling()
  {
    move_group_->setMaxVelocityScalingFactor(1.0);
    move_group_->setMaxAccelerationScalingFactor(1.0);
  }

  bool moveZOnlyOmpl(double target_z, const char * label)
  {
    useMotionTipLink();
    setFastMotionScaling();
    ensureValidStartState();
    const auto current = move_group_->getCurrentPose().pose;
    move_group_->setGoalPositionTolerance(0.015);
    move_group_->setGoalOrientationTolerance(6.28);
    move_group_->clearPoseTargets();
    move_group_->setPlanningTime(8.0);
    move_group_->setNumPlanningAttempts(2);
    move_group_->setPositionTarget(
      current.position.x, current.position.y, target_z);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto plan_code = move_group_->plan(plan);
    if (plan_code != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(
        node_->get_logger(), "%s OMPL z -> FAILED (plan %s) z=%.3f target=%.3f",
        label, moveItErrorName(plan_code.val),
        move_group_->getCurrentPose().pose.position.z, target_z);
      return false;
    }
    scalePlanTimestamps(plan, kSimTrajectoryTimeScale);
    const auto code = move_group_->execute(plan);
    const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_link6", "%s -> %s (%s)",
      label, ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    RCLCPP_INFO(
      node_->get_logger(), "%s OMPL z -> %s (%s) z=%.3f target=%.3f",
      label, ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val),
      move_group_->getCurrentPose().pose.position.z, target_z);
    return ok;
  }

  bool moveLink6Position(double target_x, double target_y, double target_z, const char * label)
  {
    ensureValidStartState();
    move_group_->setEndEffectorLink("link6");
    const auto current = move_group_->getCurrentPose().pose;

    RCLCPP_INFO(
      node_->get_logger(),
      "%s link6 from=(%.3f, %.3f, %.3f) to=(%.3f, %.3f, %.3f)",
      label, current.position.x, current.position.y, current.position.z,
      target_x, target_y, target_z);

    const bool is_push = std::strstr(label, "Push") != nullptr;
    const bool is_approach = std::strstr(label, "Approach") != nullptr;
    move_group_->setGoalPositionTolerance(
      is_push ? 0.04 : (is_approach ? 0.02 : 0.03));
    move_group_->setGoalOrientationTolerance(6.28);
    move_group_->clearPoseTargets();
    move_group_->setPlanningTime(is_push ? 5.0 : 8.0);
    move_group_->setNumPlanningAttempts(is_push ? 3 : 2);
    setFastMotionScaling();
    move_group_->setPositionTarget(target_x, target_y, target_z);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto plan_code = move_group_->plan(plan);
    if (plan_code != moveit::core::MoveItErrorCode::SUCCESS) {
      PICK_PIPELINE_INFO(
        node_->get_logger(), "motion_link6", "%s -> FAILED (plan %s)",
        label, moveItErrorName(plan_code.val));
      return false;
    }
    scalePlanTimestamps(plan, kSimTrajectoryTimeScale);
    const auto code = move_group_->execute(plan);
    bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    if (!ok && is_push) {
      const auto after = move_group_->getCurrentPose().pose;
      const double err = std::hypot(
        after.position.x - target_x, after.position.y - target_y,
        after.position.z - target_z);
      const double moved = std::hypot(
        after.position.x - current.position.x, after.position.y - current.position.y);
      if (err <= 0.038 && moved >= 0.008) {
        ok = true;
        PICK_PIPELINE_WARN(
          node_->get_logger(), "motion_link6",
          "%s execute %s partial err=%.3f moved=%.3f",
          label, moveItErrorName(code.val), err, moved);
      }
    }
  PICK_PIPELINE_INFO(
    node_->get_logger(), "motion_link6", "%s -> %s (%s)",
    label, ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    RCLCPP_INFO(
      node_->get_logger(), "%s -> %s (%s)",
      label, ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    if (!ok) {
      logCurrentJointState(node_->get_logger(), label, currentJointPositions());
    }
    return ok;
  }

  bool moveLink6Pose(
    const geometry_msgs::msg::Pose & target, const char * label, bool is_push = false)
  {
    ensureValidStartState();
    move_group_->setEndEffectorLink("link6");
    const auto current = move_group_->getCurrentPose().pose;
    RCLCPP_INFO(
      node_->get_logger(),
      "%s link6 from=(%.3f, %.3f, %.3f) to=(%.3f, %.3f, %.3f) (pose+orient)",
      label, current.position.x, current.position.y, current.position.z,
      target.position.x, target.position.y, target.position.z);

    move_group_->setGoalPositionTolerance(is_push ? 0.04 : 0.025);
    move_group_->setGoalOrientationTolerance(max_rod_yaw_tol_rad_);
    move_group_->clearPoseTargets();
    move_group_->setPlanningTime(is_push ? 5.0 : 8.0);
    move_group_->setNumPlanningAttempts(is_push ? 3 : 2);
    setFastMotionScaling();
    move_group_->setPoseTarget(target);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto plan_code = move_group_->plan(plan);
    if (plan_code != moveit::core::MoveItErrorCode::SUCCESS) {
      PICK_PIPELINE_INFO(
        node_->get_logger(), "motion_link6", "%s -> FAILED (plan %s)",
        label, moveItErrorName(plan_code.val));
      return false;
    }
    scalePlanTimestamps(plan, kSimTrajectoryTimeScale);
    const auto code = move_group_->execute(plan);
    const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_link6", "%s -> %s (%s)",
      label, ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    if (!ok) {
      logCurrentJointState(node_->get_logger(), label, currentJointPositions());
    }
    return ok;
  }

  static double segmentMinClearanceToTrunk(
    const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
  {
    double min_clear = 1e9;
    for (int i = 0; i <= 8; ++i) {
      const double t = static_cast<double>(i) / 8.0;
      const double x = a.x + t * (b.x - a.x);
      const double y = a.y + t * (b.y - a.y);
      const double dist_xy = std::hypot(x - kPlantOriginX, y - kPlantOriginY);
      min_clear = std::min(min_clear, dist_xy - kTrunkRadius);
    }
    return min_clear;
  }

  void applyPlantCollisionObstacles()
  {
    if (plant_collision_applied_) {
      return;
    }
    moveit_msgs::msg::CollisionObject trunk;
    trunk.id = "plant_trunk";
    trunk.header.frame_id = "base_link";
    trunk.operation = moveit_msgs::msg::CollisionObject::ADD;

    shape_msgs::msg::SolidPrimitive cylinder;
    cylinder.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    cylinder.dimensions = {kTrunkHeight, kTrunkRadius * 2.0};

    geometry_msgs::msg::Pose trunk_pose;
    trunk_pose.position.x = kPlantOriginX;
    trunk_pose.position.y = kPlantOriginY;
    trunk_pose.position.z = kPlantOriginZ + kTrunkLocalZ;
    trunk_pose.orientation.w = 1.0;

    trunk.primitives.push_back(cylinder);
    trunk.primitive_poses.push_back(trunk_pose);
    planning_scene_interface_.applyCollisionObject(trunk);
    plant_collision_applied_ = true;
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_collision",
      "plant trunk collision at (%.3f, %.3f, %.3f) r=%.3f h=%.3f",
      trunk_pose.position.x, trunk_pose.position.y, trunk_pose.position.z,
      kTrunkRadius, kTrunkHeight);
  }

  bool pushAlongApproachLine(
    const geometry_msgs::msg::Pose & goal, const char * label,
    double pos_tol = 0.035, int max_iters = 8, double step_m = 0.045)
  {
    setFastMotionScaling();
    double last_remain = 1e9;
    int stall_iters = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
      ensureValidStartState();
      useMotionTipLink();
      const auto current = move_group_->getCurrentPose().pose;
      const double remain = posDist(current.position, goal.position);
      if (remain <= pos_tol) {
        PICK_PIPELINE_INFO(
          node_->get_logger(), "motion_push",
          "%s lateral final remain=%.4f OK iter=%d", label, remain, iter);
        return true;
      }

      const double clearance = segmentMinClearanceToTrunk(current.position, goal.position);
      if (clearance < kTrunkAvoidRadius * 0.5) {
        PICK_PIPELINE_WARN(
          node_->get_logger(), "motion_push",
          "%s path too close to trunk clearance=%.3f", label, clearance);
      }

      Eigen::Vector3d delta(
        goal.position.x - current.position.x,
        goal.position.y - current.position.y,
        goal.position.z - current.position.z);
      const double dist = delta.norm();
      if (dist < 1e-4) {
        return true;
      }
      delta /= dist;
      const double move = std::min(step_m, dist);

      geometry_msgs::msg::Pose next = current;
      next.position.x += move * delta.x();
      next.position.y += move * delta.y();
      next.position.z += move * delta.z();
      next.orientation = goal.orientation;

      auto try_step = [&](double scale) {
        geometry_msgs::msg::Pose step_pose = current;
        step_pose.position.x += scale * move * delta.x();
        step_pose.position.y += scale * move * delta.y();
        step_pose.position.z += scale * move * delta.z();
        step_pose.orientation = goal.orientation;
        if (moveLink6Pose(step_pose, label, true)) {
          return true;
        }
        move_group_->stop();
        ensureValidStartState();
        return moveLink6Position(
          step_pose.position.x, step_pose.position.y, step_pose.position.z, label);
      };

      if (!try_step(1.0)) {
        move_group_->stop();
        ensureValidStartState();
        if (!try_step(0.5)) {
          PICK_PIPELINE_WARN(
            node_->get_logger(), "motion_push",
            "%s lateral step failed iter=%d remain=%.3f", label, iter, remain);
          return remain <= pos_tol + 0.04;
        }
      }

      const double remain_after = posDist(
        move_group_->getCurrentPose().pose.position, goal.position);
      PICK_PIPELINE_INFO(
        node_->get_logger(), "motion_push",
        "%s lateral step %d remain=%.3f pos=(%.3f, %.3f, %.3f)",
        label, iter, remain_after,
        move_group_->getCurrentPose().pose.position.x,
        move_group_->getCurrentPose().pose.position.y,
        move_group_->getCurrentPose().pose.position.z);

      if (remain_after >= last_remain - 0.005) {
        if (++stall_iters >= 2) {
          return remain_after <= pos_tol + 0.03;
        }
      } else {
        stall_iters = 0;
      }
      last_remain = remain_after;
    }
    const auto end_pose = move_group_->getCurrentPose().pose;
    return posDist(end_pose.position, goal.position) <= pos_tol + 0.04;
  }

  bool stepCartesianTo(
    const geometry_msgs::msg::Pose & goal, const char * label,
    double pos_tol = 0.03, int max_iters = 12)
  {
    useMotionTipLink();
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);

    for (int iter = 0; iter < max_iters; ++iter) {
      move_group_->setStartStateToCurrentState();
      const auto current = move_group_->getCurrentPose().pose;
      if (posDist(current.position, goal.position) <= pos_tol) {
        RCLCPP_INFO(node_->get_logger(), "%s reached (iter=%d)", label, iter);
        return true;
      }

      std::vector<geometry_msgs::msg::Pose> waypoints{current, goal};
      moveit_msgs::msg::RobotTrajectory trajectory;
      moveit_msgs::msg::MoveItErrorCodes error_code;
      const double fraction = move_group_->computeCartesianPath(
        waypoints, 0.008, trajectory, false, &error_code);
      if (fraction < 0.08) {
        RCLCPP_WARN(
          node_->get_logger(), "%s stalled iter=%d frac=%.0f%% err=%d",
          label, iter, fraction * 100.0, error_code.val);
        return false;
      }
      const auto code = move_group_->execute(trajectory);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_WARN(
          node_->get_logger(), "%s execute failed iter=%d (%s)",
          label, iter, moveItErrorName(code.val));
        return false;
      }
      RCLCPP_INFO(
        node_->get_logger(), "%s iter=%d frac=%.0f%% pos=(%.3f, %.3f, %.3f)",
        label, iter, fraction * 100.0,
        move_group_->getCurrentPose().pose.position.x,
        move_group_->getCurrentPose().pose.position.y,
        move_group_->getCurrentPose().pose.position.z);
      if (fraction >= 0.98) {
        return true;
      }
    }
    RCLCPP_WARN(node_->get_logger(), "%s max iters reached", label);
    return false;
  }

  bool moveLink6XY(const geometry_msgs::msg::PoseStamped & slot_pose)
  {
    move_group_->setEndEffectorLink("link6");
    move_group_->setStartStateToCurrentState();
    const auto current = move_group_->getCurrentPose();
    move_group_->setGoalPositionTolerance(0.025);
    move_group_->setGoalOrientationTolerance(6.28);
    move_group_->clearPoseTargets();
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);
    move_group_->setPositionTarget(
      slot_pose.pose.position.x, slot_pose.pose.position.y, current.pose.position.z);

    RCLCPP_INFO(
      node_->get_logger(),
      "MoveLink6XY target=(%.3f, %.3f, %.3f) from link6 z=%.3f",
      slot_pose.pose.position.x, slot_pose.pose.position.y, current.pose.position.z,
      current.pose.position.z);

    const auto code = move_group_->move();
    const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    RCLCPP_INFO(
      node_->get_logger(), "MoveLink6XY -> %s (%s)",
      ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    return ok;
  }

  bool moveHorizontalXY(double target_x, double target_y, const char * label)
  {
    useMotionTipLink();
    setFastMotionScaling();
    move_group_->setStartStateToCurrentState();
    const auto current = move_group_->getCurrentPose().pose;
    geometry_msgs::msg::Pose goal = current;
    goal.position.x = target_x;
    goal.position.y = target_y;

    RCLCPP_INFO(
      node_->get_logger(), "%s from=(%.3f, %.3f) to=(%.3f, %.3f) z=%.3f",
      label, current.position.x, current.position.y, target_x, target_y, current.position.z);

    return stepCartesianTo(goal, label, 0.025, 15);
  }

  bool alignRodYaw(const geometry_msgs::msg::Quaternion & orientation)
  {
    ensureValidStartState();
    useMotionTipLink();
    const auto current = move_group_->getCurrentPose();
    geometry_msgs::msg::Pose target = current.pose;
    target.orientation = orientation;

    RCLCPP_INFO(
      node_->get_logger(),
      "AlignRodYaw at pos=(%.3f, %.3f, %.3f) tol=%.1fdeg (branch may enter slot at angle)",
      current.pose.position.x, current.pose.position.y, current.pose.position.z,
      max_rod_yaw_tol_deg_);

    move_group_->setGoalPositionTolerance(0.02);
    move_group_->setGoalOrientationTolerance(max_rod_yaw_tol_rad_);
    move_group_->clearPoseTargets();
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);
    move_group_->setPoseTarget(target);
    const auto code = move_group_->move();
    const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    RCLCPP_INFO(
      node_->get_logger(), "AlignRodYaw -> %s (%s)",
      ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    return ok;
  }

  bool cartesianPositionStep(const geometry_msgs::msg::Pose & goal, const char * label)
  {
    useMotionTipLink();
    move_group_->setStartStateToCurrentState();
    const auto current = move_group_->getCurrentPose().pose;
    if (posDist(current.position, goal.position) <= 0.015) {
      return true;
    }
    geometry_msgs::msg::Pose next = current;
    next.position = goal.position;

    std::vector<geometry_msgs::msg::Pose> waypoints{current, next};
    moveit_msgs::msg::RobotTrajectory trajectory;
    moveit_msgs::msg::MoveItErrorCodes error_code;
    const double fraction = move_group_->computeCartesianPath(
      waypoints, 0.008, trajectory, false, &error_code);
    if (fraction < 0.08) {
      RCLCPP_WARN(
        node_->get_logger(), "%s cartesian frac=%.0f%% err=%d",
        label, fraction * 100.0, error_code.val);
      return false;
    }
    const auto code = move_group_->execute(trajectory);
    return code == moveit::core::MoveItErrorCode::SUCCESS;
  }

  bool pushToward(
    const geometry_msgs::msg::Pose & goal, const char * label, const char * stage,
    double pos_tol = 0.03, int max_iters = 22)
  {
    const geometry_msgs::msg::Pose link6_goal = poseEefToLink6(goal);
    useMotionTipLink();
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);
    const double step_m = 0.04;
    double last_remain = 1e9;
    int stall_iters = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
      ensureValidStartState();
      const auto current = move_group_->getCurrentPose().pose;
      const double remain = posDist(current.position, link6_goal.position);
      if (remain <= pos_tol) {
        PICK_PIPELINE_INFO(
          node_->get_logger(), stage, "%s final remain=%.4f OK iter=%d",
          label, remain, iter);
        return true;
      }

      Eigen::Vector3d delta(
        link6_goal.position.x - current.position.x,
        link6_goal.position.y - current.position.y,
        link6_goal.position.z - current.position.z);
      const double dist = delta.norm();
      if (dist < 1e-4) {
        return true;
      }
      delta /= dist;
      const double move = std::min(step_m, dist);

      geometry_msgs::msg::Pose next = current;
      next.position.x += move * delta.x();
      next.position.y += move * delta.y();
      next.position.z += move * delta.z();

      std::vector<geometry_msgs::msg::Pose> waypoints{current, next};
      moveit_msgs::msg::RobotTrajectory trajectory;
      moveit_msgs::msg::MoveItErrorCodes error_code;
      const double fraction = move_group_->computeCartesianPath(
        waypoints, 0.008, trajectory, false, &error_code);
      if (fraction < 0.12) {
        PICK_PIPELINE_WARN(
          node_->get_logger(), stage, "%s stalled iter=%d frac=%.0f%% remain=%.3f",
          label, iter, fraction * 100.0, remain);
        return remain <= pos_tol + 0.025;
      }
      const auto code = move_group_->execute(trajectory);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        PICK_PIPELINE_WARN(
          node_->get_logger(), stage, "%s execute failed iter=%d (%s)",
          label, iter, moveItErrorName(code.val));
        return false;
      }

      const double remain_after = posDist(
        move_group_->getCurrentPose().pose.position, link6_goal.position);
      PICK_PIPELINE_INFO(
        node_->get_logger(), stage,
        "%s step %d remain=%.3f pos=(%.3f, %.3f, %.3f)",
        label, iter, remain_after,
        move_group_->getCurrentPose().pose.position.x,
        move_group_->getCurrentPose().pose.position.y,
        move_group_->getCurrentPose().pose.position.z);

      if (remain_after >= last_remain - 0.008) {
        ++stall_iters;
        if (stall_iters >= 3) {
          PICK_PIPELINE_WARN(
            node_->get_logger(), stage, "%s no progress remain=%.3f", label, remain_after);
          return remain_after <= pos_tol + 0.02;
        }
      } else {
        stall_iters = 0;
      }
      last_remain = remain_after;
    }
    PICK_PIPELINE_WARN(node_->get_logger(), stage, "%s max iters", label);
    return false;
  }

  bool correctLink6ZIfNeeded(double target_z, const char * label, double tol = 0.025)
  {
    useMotionTipLink();
    const double cur_z = move_group_->getCurrentPose().pose.position.z;
    if (std::abs(cur_z - target_z) <= tol) {
      return true;
    }
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_push",
      "%s Z drift correct cur=%.3f target=%.3f", label, cur_z, target_z);
    const double dz = std::abs(cur_z - target_z);
    if (dz <= 0.04 && moveZOnly(target_z, label, true)) {
      return true;
    }
    return moveZOnlyOmpl(target_z, label);
  }

  bool pushTowardXY(
    const geometry_msgs::msg::Point & goal, const char * label, const char * stage,
    double pos_tol = 0.035, int max_iters = 8, double step_m = 0.08)
  {
    setFastMotionScaling();
    double last_remain = 1e9;
    int stall_iters = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
      ensureValidStartState();
      useMotionTipLink();
      const auto current = move_group_->getCurrentPose().pose;
      const double lock_z = goal.z;
      const double remain_xy = std::hypot(
        goal.x - current.position.x, goal.y - current.position.y);
      if (remain_xy <= pos_tol) {
        PICK_PIPELINE_INFO(
          node_->get_logger(), stage, "%s XY final remain=%.4f OK iter=%d",
          label, remain_xy, iter);
        return true;
      }

      const double dx = goal.x - current.position.x;
      const double dy = goal.y - current.position.y;
      const double step = std::min(step_m, remain_xy);
      const double nx = current.position.x + step * dx / remain_xy;
      const double ny = current.position.y + step * dy / remain_xy;

      auto try_step = [&](double sx, double sy) {
        return moveLink6Position(sx, sy, lock_z, label);
      };
      if (!try_step(nx, ny)) {
        move_group_->stop();
        ensureValidStartState();
        const double half = step * 0.5;
        const double hx = current.position.x + half * dx / remain_xy;
        const double hy = current.position.y + half * dy / remain_xy;
        if (!try_step(hx, hy)) {
          PICK_PIPELINE_WARN(
            node_->get_logger(), stage, "%s XY OMPL step failed iter=%d remain=%.3f",
            label, iter, remain_xy);
          return remain_xy <= pos_tol + 0.05;
        }
      }

      auto after = move_group_->getCurrentPose().pose;
      if (std::abs(after.position.z - lock_z) > 0.025) {
        correctLink6ZIfNeeded(lock_z, "PushStepZ", 0.02);
        after = move_group_->getCurrentPose().pose;
      }
      const double remain_after = std::hypot(
        goal.x - after.position.x, goal.y - after.position.y);
      PICK_PIPELINE_INFO(
        node_->get_logger(), stage,
        "%s XY OMPL step %d remain=%.3f pos=(%.3f, %.3f, %.3f)",
        label, iter, remain_after, after.position.x, after.position.y, after.position.z);

      if (remain_after >= last_remain - 0.005) {
        if (++stall_iters >= 2) {
          return remain_after <= pos_tol + 0.03;
        }
      } else {
        stall_iters = 0;
      }
      last_remain = remain_after;
    }
    const auto end_pose = move_group_->getCurrentPose().pose;
    const double end_remain = std::hypot(
      goal.x - end_pose.position.x, goal.y - end_pose.position.y);
    return end_remain <= pos_tol + 0.04;
  }

  bool snapLink6ToSlotXY(
    const geometry_msgs::msg::Point & slot, const char * label, double max_remain = 0.25)
  {
    useMotionTipLink();
    const auto cur = move_group_->getCurrentPose().pose;
    const double remain = std::hypot(slot.x - cur.position.x, slot.y - cur.position.y);
    if (remain <= 0.03) {
      return true;
    }
    if (remain > max_remain) {
      return false;
    }
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_push",
      "%s stepped snap remain=%.3f", label, remain);
    const double step = std::clamp(remain / 4.0, 0.03, 0.05);
    return pushTowardXY(slot, label, "motion_push", 0.03, 6, step);
  }

  bool approachLink6Fast(double target_x, double target_y, double target_z, const char * label)
  {
    useMotionTipLink();
    const auto cur = move_group_->getCurrentPose().pose;
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_link6",
      "%s fast from=(%.3f,%.3f,%.3f) to=(%.3f,%.3f,%.3f)",
      label, cur.position.x, cur.position.y, cur.position.z,
      target_x, target_y, target_z);

    geometry_msgs::msg::Pose xy_goal = cur;
    xy_goal.position.x = target_x;
    xy_goal.position.y = target_y;
    if (std::hypot(target_x - cur.position.x, target_y - cur.position.y) > 0.02) {
      if (!stepCartesianTo(xy_goal, "ApproachLink6XY", 0.03, 10)) {
        if (!moveLink6Position(target_x, target_y, cur.position.z, "ApproachLink6XYOmpl")) {
          return false;
        }
      }
    }
    if (std::abs(target_z - move_group_->getCurrentPose().pose.position.z) > 0.015) {
      if (!moveZOnlyOmpl(target_z, "ApproachLink6Z")) {
        if (!moveZOnly(target_z, "ApproachLink6Z", true)) {
          return moveLink6Position(target_x, target_y, target_z, "ApproachLink6ZOmpl");
        }
      }
    }
    return true;
  }

  bool pushAlongLine(
    const geometry_msgs::msg::Pose & goal, const char * label,
    double pos_tol = 0.03, int max_iters = 20)
  {
    return pushToward(goal, label, "motion_push", pos_tol, max_iters);
  }

  int selectApproachCandidate(
    const std::vector<geometry_msgs::msg::PoseStamped> & candidates,
    const geometry_msgs::msg::PoseStamped & slot_pose)
  {
    if (candidates.empty()) {
      return -1;
    }
    move_group_->setEndEffectorLink("link6");
    ensureValidStartState();
    const auto cur = move_group_->getCurrentPose().pose.position;
    const auto link6_slot = link6PoseFromRodTip(slot_pose.pose).position;
    const auto cur_link6 = move_group_->getCurrentPose().pose;

    const moveit::core::RobotStatePtr base_state = move_group_->getCurrentState();
    if (!base_state) {
      RCLCPP_WARN(node_->get_logger(), "SelectApproach: joint_states not ready");
      return -1;
    }
    const moveit::core::JointModelGroup * jmg =
      base_state->getRobotModel()->getJointModelGroup("arm");
    if (!jmg) {
      RCLCPP_WARN(node_->get_logger(), "SelectApproach: arm joint model group missing");
      return -1;
    }

    struct Ranked
    {
      int idx;
      double dist;
      double trunk_clear;
    };
    std::vector<Ranked> feasible;

    for (size_t idx = 0; idx < candidates.size(); ++idx) {
      const auto & pose = candidates[idx].pose;
      const auto link6_pose = poseEefToLink6(pose);
      const auto link6_pt = link6_pose.position;

      auto ikAtLink6 = [&](const geometry_msgs::msg::Pose & target) {
          moveit::core::RobotState trial(*base_state);
          Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
          iso.translation() = Eigen::Vector3d(
            target.position.x, target.position.y, target.position.z);
          Eigen::Quaterniond q(
            target.orientation.w, target.orientation.x,
            target.orientation.y, target.orientation.z);
          iso.linear() = q.toRotationMatrix();
          return trial.setFromIK(jmg, iso, "link6", 0.25) && trial.satisfiesBounds(jmg);
        };

      geometry_msgs::msg::Pose ik_pose = link6_pose;
      ik_pose.orientation = cur_link6.orientation;
      bool ik_ok = ikAtLink6(ik_pose);
      if (!ik_ok) {
        ik_ok = ikAtLink6(link6_pose);
      }
      if (!ik_ok) {
        RCLCPP_INFO(
          node_->get_logger(),
          "SelectApproach: candidate %zu/%zu link6 IK failed at (%.3f, %.3f, %.3f)",
          idx + 1, candidates.size(), link6_pt.x, link6_pt.y, link6_pt.z);
        continue;
      }
      const double trunk_clear = segmentMinClearanceToTrunk(link6_pt, link6_slot);
      feasible.push_back({
        static_cast<int>(idx),
        posDist(cur, pose.position),
        trunk_clear});
    }

    if (feasible.empty()) {
      RCLCPP_WARN(
        node_->get_logger(),
        "SelectApproach: no IK-feasible candidate among %zu rule-ranked axes",
        candidates.size());
      return -1;
    }

    std::sort(feasible.begin(), feasible.end(), [](const Ranked & a, const Ranked & b) {
        if (a.trunk_clear >= kTrunkAvoidRadius && b.trunk_clear < kTrunkAvoidRadius) {
          return true;
        }
        if (b.trunk_clear >= kTrunkAvoidRadius && a.trunk_clear < kTrunkAvoidRadius) {
          return false;
        }
        if (std::abs(a.trunk_clear - b.trunk_clear) > 0.01) {
          return a.trunk_clear > b.trunk_clear;
        }
        return a.dist < b.dist;
      });

    const int chosen = feasible.front().idx;
    const auto & pose = candidates[chosen].pose;
    useMotionTipLink();
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_select",
      "chosen=%d/%zu link6 IK OK trunk_clear=%.3f eef_pre=(%.3f, %.3f, %.3f)",
      chosen + 1, candidates.size(), feasible.front().trunk_clear,
      pose.position.x, pose.position.y, pose.position.z);
    return chosen;
  }

  bool verifyAlignment(
    const geometry_msgs::msg::PoseStamped & slot_pose,
    const geometry_msgs::msg::PoseStamped & contact_pose,
    const geometry_msgs::msg::Vector3 & branch_dir,
    double tolerance)
  {
    useMotionTipLink();
    const auto current = move_group_->getCurrentPose().pose;
    const auto link6_target = link6PoseFromRodTip(slot_pose.pose);
    const double dist_xy = std::hypot(
      current.position.x - link6_target.position.x,
      current.position.y - link6_target.position.y);
    const double dist_z = std::abs(current.position.z - link6_target.position.z);
    const double dist = posDist(current.position, link6_target.position);

    const auto slot_center = slotCenterFromLink6(current);
    const double slot_contact = posDist(slot_center, contact_pose.pose.position);
    const auto rod_tip = rodTipFromLink6(current);
    const double tip_err = posDist(rod_tip, slot_pose.pose.position);

    const auto rod_axis = rodAxisWorld(current.orientation);

    const double bx = branch_dir.x;
    const double by = branch_dir.y;
    const double bz = branch_dir.z;
    const double b_norm = std::sqrt(bx * bx + by * by + bz * bz);
    double angle_deg = 0.0;
    if (b_norm > 0.5) {
      const double dot = std::clamp(
        (rod_axis.x * bx + rod_axis.y * by + rod_axis.z * bz) / b_norm, -1.0, 1.0);
      angle_deg = std::acos(dot) * 180.0 / M_PI;
    }

    const bool ok_link6 = dist_xy <= tolerance && dist_z <= tolerance;
    const bool ok_slot = slot_contact <= tolerance;
    const bool ok_angle = angle_deg <= max_rod_yaw_tol_deg_;
    const bool ok = ok_link6 && ok_slot && ok_angle;
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_verify",
      "link6-slot dist=%.4f xy=%.4f z=%.4f slot-contact=%.4f tip-err=%.4f rod-branch=%.1fdeg tol=%.3f %s",
      dist, dist_xy, dist_z, slot_contact, tip_err, angle_deg, tolerance, ok ? "OK" : "FAIL");
    RCLCPP_INFO(
      node_->get_logger(),
      "VerifyAlignment: link6-slot dist=%.4f slot-contact=%.4f rod-branch=%.1fdeg %s",
      dist, slot_contact, angle_deg, ok ? "OK" : "FAIL");
    return ok;
  }

  bool moveZOnly(double target_z, const char * label, bool allow_up = false)
  {
    useMotionTipLink();
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);

    const double z_tol = 0.012;
    const double step = 0.025;

    for (int iter = 0; iter < 25; ++iter) {
      ensureValidStartState();
      useMotionTipLink();
      const auto current = move_group_->getCurrentPose().pose;
      const double dz = target_z - current.position.z;

      if (std::abs(dz) <= z_tol) {
        RCLCPP_INFO(
          node_->get_logger(), "%s reached z=%.3f (iter=%d)", label, current.position.z, iter);
        return true;
      }
      if (!allow_up && dz > 0.0) {
        RCLCPP_INFO(node_->get_logger(), "%s at z=%.3f above target %.3f", label, current.position.z, target_z);
        return true;
      }

      geometry_msgs::msg::Pose next = current;
      if (dz > 0.0) {
        next.position.z = std::min(target_z, current.position.z + step);
      } else {
        next.position.z = std::max(target_z, current.position.z - step);
      }

      std::vector<geometry_msgs::msg::Pose> waypoints{current, next};
      moveit_msgs::msg::RobotTrajectory trajectory;
      moveit_msgs::msg::MoveItErrorCodes error_code;
      const double fraction = move_group_->computeCartesianPath(
        waypoints, 0.005, trajectory, false, &error_code);
      if (fraction < 0.5) {
        RCLCPP_WARN(
          node_->get_logger(), "%s stalled iter=%d frac=%.0f%% z=%.3f target=%.3f",
          label, iter, fraction * 100.0, current.position.z, target_z);
        if (std::abs(current.position.z - target_z) <= 0.04) {
          return true;
        }
        return moveZOnlyOmpl(target_z, label);
      }
      const auto code = move_group_->execute(trajectory);
      if (code != moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_WARN(node_->get_logger(), "%s execute failed iter=%d", label, iter);
        return false;
      }
      RCLCPP_INFO(
        node_->get_logger(), "%s iter=%d z=%.3f -> %.3f",
        label, iter, current.position.z, next.position.z);
    }
    RCLCPP_WARN(node_->get_logger(), "%s max iters", label);
    return false;
  }

  bool moveToPoseFull(const geometry_msgs::msg::PoseStamped & target, const char * label)
  {
    ensureValidStartState();
    useMotionTipLink();
    geometry_msgs::msg::PoseStamped link6_target = target;
    link6_target.pose = poseEefToLink6(target.pose);
    move_group_->setGoalPositionTolerance(0.025);
    move_group_->setGoalOrientationTolerance(0.12);
    move_group_->clearPoseTargets();
    move_group_->setPlanningTime(20.0);
    move_group_->setNumPlanningAttempts(5);
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(accel_scale_);
    move_group_->setPoseTarget(link6_target.pose);

    RCLCPP_INFO(
      node_->get_logger(), "%s link6 target=(%.3f, %.3f, %.3f)",
      label, link6_target.pose.position.x, link6_target.pose.position.y,
      link6_target.pose.position.z);

    const auto code = move_group_->move();
    const bool ok = code == moveit::core::MoveItErrorCode::SUCCESS;
    RCLCPP_INFO(
      node_->get_logger(), "%s -> %s (%s)",
      label, ok ? "SUCCESS" : "FAILED", moveItErrorName(code.val));
    if (!ok) {
      logCurrentJointState(node_->get_logger(), label, currentJointPositions());
    }
    return ok;
  }

  bool cartesianPush(
    const geometry_msgs::msg::PoseStamped & from,
    const geometry_msgs::msg::PoseStamped & to, const char * label)
  {
    ensureValidStartState();
    useMotionTipLink();
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(poseEefToLink6(from.pose));
    waypoints.push_back(poseEefToLink6(to.pose));

    const double dist = posDist(from.pose.position, to.pose.position);
    RCLCPP_INFO(
      node_->get_logger(), "%s push dist=%.3f m from=(%.3f, %.3f, %.3f) to=(%.3f, %.3f, %.3f)",
      label, dist,
      from.pose.position.x, from.pose.position.y, from.pose.position.z,
      to.pose.position.x, to.pose.position.y, to.pose.position.z);

    return executeCartesianWaypoints(waypoints, label, 0.90, false);
  }

  bool executeLateralApproach(
    const geometry_msgs::msg::PoseStamped & pre_approach,
    const geometry_msgs::msg::PoseStamped & slot_pose)
  {
    useMotionTipLink();

    const geometry_msgs::msg::Pose link6_pre_pose = link6PoseFromRodTip(pre_approach.pose);
    const geometry_msgs::msg::Pose link6_slot_pose = link6PoseFromRodTip(slot_pose.pose);
    const auto current = move_group_->getCurrentPose();

    const auto slot_center_goal = offsetAlongRod(
      slot_pose.pose.position, slot_pose.pose.orientation, -kSlotCenterFromTipM);

    const double trunk_clear = segmentMinClearanceToTrunk(
      link6_pre_pose.position, link6_slot_pose.position);
    const double approach_len = posDist(link6_pre_pose.position, link6_slot_pose.position);

    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_lateral",
      "rod_tip_goal=(%.3f, %.3f, %.3f) slot_center_goal=(%.3f, %.3f, %.3f)",
      slot_pose.pose.position.x, slot_pose.pose.position.y, slot_pose.pose.position.z,
      slot_center_goal.x, slot_center_goal.y, slot_center_goal.z);
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_lateral",
      "link6_from=(%.3f, %.3f, %.3f) link6_pre=(%.3f, %.3f, %.3f) link6_slot=(%.3f, %.3f, %.3f) "
      "(link6_to_tip=%.3f link6_to_slot=%.3f)",
      current.pose.position.x, current.pose.position.y, current.pose.position.z,
      link6_pre_pose.position.x, link6_pre_pose.position.y, link6_pre_pose.position.z,
      link6_slot_pose.position.x, link6_slot_pose.position.y, link6_slot_pose.position.z,
      kLink6ToEefTipM, kLink6ToSlotCenterM);
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_lateral",
      "lateral_push dist=%.3f trunk_clear=%.3f (avoid trunk, rod aligned)",
      approach_len, trunk_clear);

    auto link6_at = [&](const geometry_msgs::msg::Pose & ref, double x, double y, double z) {
        geometry_msgs::msg::Pose p = ref;
        p.position.x = x;
        p.position.y = y;
        p.position.z = z;
        return p;
      };

    geometry_msgs::msg::Pose safe_pose = link6_pre_pose;
    safe_pose.position.z += 0.04;
    if (!moveLink6Pose(safe_pose, "ApproachSafeZ")) {
      PICK_PIPELINE_WARN(
        node_->get_logger(), "motion_link6",
        "ApproachSafeZ pose failed, trying XY-only safe_z");
      if (!moveLink6Position(
          link6_pre_pose.position.x, link6_pre_pose.position.y,
          safe_pose.position.z, "ApproachSafeZXY"))
      {
        PICK_PIPELINE_ERROR(node_->get_logger(), "motion_link6", "ApproachSafeZ failed");
        return false;
      }
    }

    if (!moveLink6Pose(link6_pre_pose, "ApproachPre")) {
      PICK_PIPELINE_WARN(
        node_->get_logger(), "motion_link6",
        "ApproachPre pose failed, trying position-only");
      if (!moveLink6Position(
          link6_pre_pose.position.x, link6_pre_pose.position.y,
          link6_pre_pose.position.z, "ApproachPreXY"))
      {
        PICK_PIPELINE_ERROR(node_->get_logger(), "motion_lateral", "ApproachPre failed");
        return false;
      }
    }

    if (trunk_clear < 0.02) {
      PICK_PIPELINE_INFO(
        node_->get_logger(), "motion_lateral",
        "trunk_clear=%.3f low, staged push: align Y then X (rod orient locked)",
        trunk_clear);
      const auto align_y = link6_at(
        link6_slot_pose,
        link6_pre_pose.position.x, link6_slot_pose.position.y, link6_pre_pose.position.z);
      if (!moveLink6Pose(align_y, "PushAlignY", true)) {
        PICK_PIPELINE_WARN(
          node_->get_logger(), "motion_lateral",
          "PushAlignY failed, trying direct lateral push");
      }
      for (int i = 0; i < 8; ++i) {
        ensureValidStartState();
        useMotionTipLink();
        const auto cur = move_group_->getCurrentPose().pose;
        const double dx = link6_slot_pose.position.x - cur.position.x;
        const double remain = posDist(cur.position, link6_slot_pose.position);
        if (remain <= 0.035) {
          break;
        }
        const double step = std::clamp(std::abs(dx), 0.03, 0.045) * (dx >= 0.0 ? 1.0 : -1.0);
        const auto step_pose = link6_at(
          link6_slot_pose, cur.position.x + step, cur.position.y, cur.position.z);
        if (!moveLink6Pose(step_pose, "PushIntoSlotX", true)) {
          break;
        }
      }
    } else if (!pushAlongApproachLine(link6_slot_pose, "PushIntoSlot", 0.035, 8, 0.045)) {
      PICK_PIPELINE_WARN(
        node_->get_logger(), "motion_push",
        "PushIntoSlot lateral incomplete, VerifyAlignment will decide");
    }

    const auto after_push = move_group_->getCurrentPose().pose;
    const double push_remain = posDist(after_push.position, link6_slot_pose.position);
    if (push_remain > 0.025) {
      moveLink6Pose(link6_slot_pose, "PushIntoSlotFinal", true);
    }

    const auto final_pose = move_group_->getCurrentPose().pose;
    const auto final_slot = slotCenterFromLink6(final_pose);
    const auto final_tip = rodTipFromLink6(final_pose);
    const double slot_remain = posDist(final_pose.position, link6_slot_pose.position);
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_push",
      "PushIntoSlot final link6_remain=%.4f slot_center=(%.3f,%.3f,%.3f) rod_tip=(%.3f,%.3f,%.3f) %s",
      slot_remain,
      final_slot.x, final_slot.y, final_slot.z,
      final_tip.x, final_tip.y, final_tip.z,
      slot_remain < 0.03 ? "OK" : "WARN");
    return true;
  }

  bool executeLateralRetract(const geometry_msgs::msg::PoseStamped & retract_pose)
  {
    ensureValidStartState();
    useMotionTipLink();
    const geometry_msgs::msg::Pose link6_retract = poseEefToLink6(retract_pose.pose);
    PICK_PIPELINE_INFO(
      node_->get_logger(), "motion_push",
      "LateralRetract link6 target=(%.3f, %.3f, %.3f)",
      link6_retract.position.x, link6_retract.position.y, link6_retract.position.z);
    if (!pushAlongApproachLine(link6_retract, "LateralRetract", 0.04, 10, 0.05)) {
      PICK_PIPELINE_WARN(
        node_->get_logger(), "motion_push",
        "LateralRetract lateral path incomplete, continuing to dump");
    }
    return true;
  }

  bool cartesianLift(double lift_z)
  {
    useMotionTipLink();
    move_group_->setStartStateToCurrentState();
    const auto current = move_group_->getCurrentPose();
    geometry_msgs::msg::Pose up = current.pose;
    up.position.z += lift_z;

    RCLCPP_INFO(
      node_->get_logger(),
      "CartesianLift from z=%.3f to z=%.3f (dz=%.3f)",
      current.pose.position.z, up.position.z, lift_z);

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(current.pose);
    waypoints.push_back(up);
    return executeCartesianWaypoints(waypoints, "CartesianLift", 0.95);
  }

  bool cartesianMove(const geometry_msgs::msg::PoseStamped & target)
  {
    useMotionTipLink();
    move_group_->setStartStateToCurrentState();
    const auto current = move_group_->getCurrentPose();
    const auto link6_target = poseEefToLink6(target.pose);
    RCLCPP_INFO(
      node_->get_logger(),
      "CartesianMove link6 descend from=(%.3f, %.3f, %.3f) to=(%.3f, %.3f, %.3f)",
      current.pose.position.x, current.pose.position.y, current.pose.position.z,
      link6_target.position.x, link6_target.position.y, link6_target.position.z);

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(current.pose);
    waypoints.push_back(link6_target);
    return executeCartesianWaypoints(waypoints, "CartesianMove");
  }

private:
  explicit MoveItInterface(const rclcpp::Node::SharedPtr & node)
  : node_(node),
    move_group_(std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, "arm")),
    planning_scene_interface_()
  {
    move_group_->setPoseReferenceFrame("base_link");
    useMotionTipLink();
    move_group_->setPlanningTime(15.0);
    move_group_->setNumPlanningAttempts(5);
    velocity_scale_ = node_->declare_parameter<double>("moveit_velocity_scale", 1.0);
    accel_scale_ = node_->declare_parameter<double>("moveit_accel_scale", 1.0);
    max_rod_yaw_tol_deg_ = node_->declare_parameter<double>("max_branch_slot_angle_deg", 25.0);
    max_rod_yaw_tol_rad_ = max_rod_yaw_tol_deg_ * M_PI / 180.0;
    RCLCPP_INFO(
      node_->get_logger(),
      "MoveIt interface ready (motion_tip=%s, max_slot_angle=%.0f deg)",
      kMotionTipLink, max_rod_yaw_tol_deg_);
  }

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  bool plant_collision_applied_{false};
  double velocity_scale_{0.4};
  double accel_scale_{0.4};
  double max_rod_yaw_tol_deg_{25.0};
  double max_rod_yaw_tol_rad_{0.436};
  static std::unique_ptr<MoveItInterface> instance_;
};

std::unique_ptr<MoveItInterface> MoveItInterface::instance_ = nullptr;

class MoveToNamedNode : public BT::SyncActionNode
{
public:
  MoveToNamedNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto target = getInput<std::string>("target_name");
    if (!target) {
      return BT::NodeStatus::FAILURE;
    }
    return MoveItInterface::get().moveToNamed(target.value()) ?
           BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("target_name")};
  }
};

class MoveToPoseMoveItNode : public BT::SyncActionNode
{
public:
  MoveToPoseMoveItNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto target = getInput<geometry_msgs::msg::PoseStamped>("target");
    if (!target) {
      return BT::NodeStatus::FAILURE;
    }
    double offset_z = 0.0;
    if (auto offset = getInput<double>("offset_z")) {
      offset_z = offset.value();
    }
    bool enforce_orientation = false;
    if (auto orient = getInput<bool>("use_orientation")) {
      enforce_orientation = orient.value();
    }
    std::string planning_link = kMotionTipLink;
    if (auto link = getInput<std::string>("planning_link")) {
      planning_link = link.value();
    }
    return MoveItInterface::get().moveToPose(
      target.value(), offset_z, enforce_orientation, planning_link) ?
           BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<geometry_msgs::msg::PoseStamped>("target"),
      BT::InputPort<double>("offset_z"),
      BT::InputPort<bool>("use_orientation"),
      BT::InputPort<std::string>("planning_link"),
    };
  }
};

class CartesianMoveMoveItNode : public BT::SyncActionNode
{
public:
  CartesianMoveMoveItNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto target = getInput<geometry_msgs::msg::PoseStamped>("target");
    if (!target) {
      return BT::NodeStatus::FAILURE;
    }
    return MoveItInterface::get().cartesianMove(target.value()) ?
           BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<geometry_msgs::msg::PoseStamped>("target")};
  }
};

class LateralApproachMoveItNode : public BT::SyncActionNode
{
public:
  LateralApproachMoveItNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto pre = getInput<geometry_msgs::msg::PoseStamped>("pre_approach");
    auto slot = getInput<geometry_msgs::msg::PoseStamped>("slot_pose");
    if (!pre || !slot) {
      return BT::NodeStatus::FAILURE;
    }
    return MoveItInterface::get().executeLateralApproach(pre.value(), slot.value()) ?
           BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<geometry_msgs::msg::PoseStamped>("pre_approach"),
      BT::InputPort<geometry_msgs::msg::PoseStamped>("slot_pose"),
    };
  }
};

class LateralRetractMoveItNode : public BT::SyncActionNode
{
public:
  LateralRetractMoveItNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto retract = getInput<geometry_msgs::msg::PoseStamped>("retract_pose");
    if (!retract) {
      return BT::NodeStatus::FAILURE;
    }
    return MoveItInterface::get().executeLateralRetract(retract.value()) ?
           BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<geometry_msgs::msg::PoseStamped>("retract_pose")};
  }
};

class CartesianLiftMoveItNode : public BT::SyncActionNode
{
public:
  CartesianLiftMoveItNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    double lift_z = 0.12;
    if (auto lift = getInput<double>("lift_z")) {
      lift_z = lift.value();
    }
    return MoveItInterface::get().cartesianLift(lift_z) ?
           BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<double>("lift_z")};
  }
};

class SelectApproachNode : public BT::SyncActionNode
{
public:
  SelectApproachNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto vib_plan = getInput<picking_msgs::msg::VibrationPlan>("vib_plan");
    if (!vib_plan || vib_plan->pre_approach_candidates.empty()) {
      return BT::NodeStatus::FAILURE;
    }
    const int idx = MoveItInterface::get().selectApproachCandidate(
      vib_plan->pre_approach_candidates, vib_plan->slot_pose);
    if (idx < 0) {
      return BT::NodeStatus::FAILURE;
    }
    vib_plan->pre_approach = vib_plan->pre_approach_candidates[idx];
    vib_plan->retract_pose = vib_plan->pre_approach;
    setOutput("vib_plan", vib_plan.value());
    setOutput("pre_approach", vib_plan->pre_approach);
    setOutput("retract_pose", vib_plan->retract_pose);
    return BT::NodeStatus::SUCCESS;
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::BidirectionalPort<picking_msgs::msg::VibrationPlan>("vib_plan"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("pre_approach"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("retract_pose"),
    };
  }
};

class VerifyAlignmentNode : public BT::SyncActionNode
{
public:
  VerifyAlignmentNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto vib_plan = getInput<picking_msgs::msg::VibrationPlan>("vib_plan");
    if (!vib_plan) {
      return BT::NodeStatus::FAILURE;
    }
    double tolerance = 0.025;
    if (auto tol = getInput<double>("tolerance")) {
      tolerance = tol.value();
    }
    auto contact = getInput<geometry_msgs::msg::PoseStamped>("contact_pose");
    if (!contact) {
      return BT::NodeStatus::FAILURE;
    }
    const bool ok = MoveItInterface::get().verifyAlignment(
      vib_plan->slot_pose, contact.value(), vib_plan->branch_dir, tolerance);
    return ok ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<picking_msgs::msg::VibrationPlan>("vib_plan"),
      BT::InputPort<geometry_msgs::msg::PoseStamped>("contact_pose"),
      BT::InputPort<double>("tolerance"),
    };
  }
};

void registerMoveItBtNodes(BT::BehaviorTreeFactory & factory)
{
  factory.registerNodeType<MoveToNamedNode>("MoveToNamed");
  factory.registerNodeType<MoveToPoseMoveItNode>("MoveToPose");
  factory.registerNodeType<CartesianMoveMoveItNode>("CartesianMove");
  factory.registerNodeType<LateralApproachMoveItNode>("LateralApproach");
  factory.registerNodeType<LateralRetractMoveItNode>("LateralRetract");
  factory.registerNodeType<CartesianLiftMoveItNode>("CartesianLift");
  factory.registerNodeType<SelectApproachNode>("SelectApproach");
  factory.registerNodeType<VerifyAlignmentNode>("VerifyAlignment");
}

void initMoveIt(const rclcpp::Node::SharedPtr & node)
{
  MoveItInterface::init(node);
}

}  // namespace picking_task
