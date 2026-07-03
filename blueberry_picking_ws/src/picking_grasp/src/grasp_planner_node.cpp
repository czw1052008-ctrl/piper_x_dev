#include <memory>
#include <mutex>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "picking_grasp/suction_planner.hpp"
#include "picking_grasp/vibration_planner.hpp"
#include "picking_msgs/msg/detected_berry_array.hpp"
#include "picking_msgs/msg/perception_status.hpp"
#include "picking_msgs/msg/vibration_plan.hpp"
#include "picking_msgs/srv/plan_suction.hpp"
#include "picking_msgs/srv/plan_vibration.hpp"
#include "rclcpp/rclcpp.hpp"

namespace picking_grasp
{

namespace
{

void logPose(const rclcpp::Logger & logger, const char * label, const geometry_msgs::msg::PoseStamped & ps)
{
  RCLCPP_INFO(
    logger, "%s [%s] pos=(%.3f, %.3f, %.3f) quat=(%.3f, %.3f, %.3f, %.3f)",
    label, ps.header.frame_id.c_str(),
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z,
    ps.pose.orientation.x, ps.pose.orientation.y,
    ps.pose.orientation.z, ps.pose.orientation.w);
}

}  // namespace

class GraspPlannerNode : public rclcpp::Node
{
public:
  GraspPlannerNode()
  : Node("grasp_planner_node")
  {
    declare_parameter<std::string>("end_effector_mode", "suction");
    declare_parameter("stream_topics", true);
    declare_parameter("slot_center_from_tip", 0.04);
    declare_parameter("slot_width_m", 0.04);
    declare_parameter("max_branch_slot_angle_deg", 25.0);
    declare_parameter("hover_z", 0.12);
    declare_parameter("pre_approach_dist", 0.25);
    declare_parameter("vibration_duration_sec", 3.0);
    declare_parameter("min_berries", 2);
    declare_parameter("dump_lift_z", 0.15);
    declare_parameter("table_top_z", 0.31);
    declare_parameter("table_clearance_m", 0.05);
    declare_parameter("rod_radius", 0.008);
    declare_parameter("berry_radius", 0.0075);
    declare_parameter("cup_contact_offset_m", 0.04);
    declare_parameter("pre_grasp_offset", 0.15);
    declare_parameter("post_grasp_offset", 0.12);
    declare_parameter("berry_selection_mode", "nearest");
    declare_parameter("nearest_frame", "camera_wrist_color_optical_frame");

    mode_ = get_parameter("end_effector_mode").as_string();
    stream_topics_ = get_parameter("stream_topics").as_bool();
    vibration_params_.slot_center_from_tip = get_parameter("slot_center_from_tip").as_double();
    vibration_params_.slot_width_m = get_parameter("slot_width_m").as_double();
    vibration_params_.max_branch_slot_angle_deg = get_parameter("max_branch_slot_angle_deg").as_double();
    vibration_params_.hover_z = get_parameter("hover_z").as_double();
    vibration_params_.pre_approach_dist = get_parameter("pre_approach_dist").as_double();
    vibration_params_.vibration_duration_sec = get_parameter("vibration_duration_sec").as_double();
    vibration_params_.min_berries = get_parameter("min_berries").as_int();
    vibration_params_.dump_lift_z = get_parameter("dump_lift_z").as_double();
    vibration_params_.table_top_z = get_parameter("table_top_z").as_double();
    vibration_params_.table_clearance_m = get_parameter("table_clearance_m").as_double();
    vibration_params_.rod_radius = get_parameter("rod_radius").as_double();
    suction_params_.berry_radius = get_parameter("berry_radius").as_double();
    suction_params_.cup_contact_offset_m = get_parameter("cup_contact_offset_m").as_double();
    suction_params_.pre_grasp_offset = get_parameter("pre_grasp_offset").as_double();
    suction_params_.post_grasp_offset = get_parameter("post_grasp_offset").as_double();
    suction_params_.selection_mode = get_parameter("berry_selection_mode").as_string();
    suction_params_.nearest_frame = get_parameter("nearest_frame").as_string();

    suction_srv_ = create_service<picking_msgs::srv::PlanSuction>(
      "plan_suction",
      std::bind(&GraspPlannerNode::onPlanSuction, this, std::placeholders::_1, std::placeholders::_2));

    vibration_srv_ = create_service<picking_msgs::srv::PlanVibration>(
      "plan_vibration",
      std::bind(&GraspPlannerNode::onPlanVibration, this, std::placeholders::_1, std::placeholders::_2));

    if (stream_topics_ && mode_ == "vibration") {
      plan_pub_ = create_publisher<picking_msgs::msg::VibrationPlan>("/plan/vibration", 10);
      berries_sub_ = create_subscription<picking_msgs::msg::DetectedBerryArray>(
        "/perception/berries", 10,
        std::bind(&GraspPlannerNode::onBerries, this, std::placeholders::_1));
      contact_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/perception/contact_pose", 10,
        std::bind(&GraspPlannerNode::onContact, this, std::placeholders::_1));
      stem_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/perception/stem_direction", 10,
        std::bind(&GraspPlannerNode::onStem, this, std::placeholders::_1));
      status_sub_ = create_subscription<picking_msgs::msg::PerceptionStatus>(
        "/perception/status", 10,
        std::bind(&GraspPlannerNode::onStatus, this, std::placeholders::_1));
    }

    RCLCPP_INFO(
      get_logger(),
      "Grasp planner ready, mode=%s stream=%s table_top_z=%.2f (base_link)",
      mode_.c_str(), stream_topics_ ? "true" : "false", vibration_params_.table_top_z);
  }

private:
  void onPlanSuction(
    const std::shared_ptr<picking_msgs::srv::PlanSuction::Request> req,
    std::shared_ptr<picking_msgs::srv::PlanSuction::Response> res)
  {
    SuctionPlanner planner(suction_params_);
    res->success = planner.plan(req->berries, res->plan, res->message);
    RCLCPP_INFO(
      get_logger(), "plan_suction: success=%s berries=%zu msg=%s",
      res->success ? "true" : "false", req->berries.size(), res->message.c_str());
  }

  void onPlanVibration(
    const std::shared_ptr<picking_msgs::srv::PlanVibration::Request> req,
    std::shared_ptr<picking_msgs::srv::PlanVibration::Response> res)
  {
    if (!req->stem_direction_valid || !req->contact_pose_valid) {
      res->success = false;
      res->message =
        "plan_vibration requires stem_direction and contact_pose from perception";
      return;
    }

    VibrationPlanner planner(vibration_params_);
    geometry_msgs::msg::PoseStamped dump_pose;
    res->success = planner.plan(
      req->berries, req->stem_direction, req->stem_direction_valid,
      req->contact_pose, req->contact_pose_valid,
      res->plan, dump_pose, res->message);
    if (res->success) {
      res->plan.dump_pose = dump_pose;
    }
  }

  void onBerries(const picking_msgs::msg::DetectedBerryArray::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    berries_ = msg->berries;
    tryPublishPlan();
  }

  void onContact(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    contact_ = *msg;
    has_contact_ = true;
    tryPublishPlan();
  }

  void onStem(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stem_ = msg->vector;
    has_stem_ = true;
    tryPublishPlan();
  }

  void onStatus(const picking_msgs::msg::PerceptionStatus::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    perception_ok_ = msg->contact_valid && msg->stem_valid && msg->berries_valid;
    tryPublishPlan();
  }

  void tryPublishPlan()
  {
    if (!plan_pub_ || !perception_ok_ || !has_contact_ || !has_stem_) {
      return;
    }
    if (static_cast<int>(berries_.size()) < vibration_params_.min_berries) {
      return;
    }

    VibrationPlanner planner(vibration_params_);
    geometry_msgs::msg::PoseStamped dump_pose;
    picking_msgs::msg::VibrationPlan plan;
    std::string message;
    const bool ok = planner.plan(
      berries_, stem_, true, contact_, true, plan, dump_pose, message);

    if (!ok) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "stream plan_vibration failed: %s", message.c_str());
      return;
    }
    plan.dump_pose = dump_pose;
    plan_pub_->publish(plan);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "stream /plan/vibration slot=(%.3f, %.3f, %.3f) berries=%zu",
      plan.slot_pose.pose.position.x, plan.slot_pose.pose.position.y,
      plan.slot_pose.pose.position.z, berries_.size());
  }

  std::string mode_;
  bool stream_topics_{false};
  VibrationPlanner::Params vibration_params_;
  SuctionPlanner::Params suction_params_;
  rclcpp::Service<picking_msgs::srv::PlanSuction>::SharedPtr suction_srv_;
  rclcpp::Service<picking_msgs::srv::PlanVibration>::SharedPtr vibration_srv_;
  rclcpp::Publisher<picking_msgs::msg::VibrationPlan>::SharedPtr plan_pub_;
  rclcpp::Subscription<picking_msgs::msg::DetectedBerryArray>::SharedPtr berries_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr contact_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr stem_sub_;
  rclcpp::Subscription<picking_msgs::msg::PerceptionStatus>::SharedPtr status_sub_;

  std::mutex mutex_;
  std::vector<picking_msgs::msg::DetectedBerry> berries_;
  geometry_msgs::msg::PoseStamped contact_;
  geometry_msgs::msg::Vector3 stem_;
  bool has_contact_{false};
  bool has_stem_{false};
  bool perception_ok_{false};
};

}  // namespace picking_grasp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<picking_grasp::GraspPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
