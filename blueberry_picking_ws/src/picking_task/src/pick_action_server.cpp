#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/bt_cout_logger.h>
#include "picking_msgs/action/pick_blueberry.hpp"
#include "picking_task/pipeline_log.hpp"
#include "picking_task/bt_nodes.hpp"
#include "picking_task/moveit_bt_nodes.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"

namespace picking_task
{

class PickActionServer : public rclcpp::Node
{
public:
  using PickAction = picking_msgs::action::PickBlueberry;
  using GoalHandle = rclcpp_action::ServerGoalHandle<PickAction>;

  PickActionServer()
  : Node("pick_action_server")
  {
    declare_parameter<std::string>("bt_xml_suction", "");
    declare_parameter<std::string>("bt_xml_vibration", "");
    declare_parameter<bool>("use_moveit", false);
    declare_parameter<bool>("debug_bt", true);

    registerCommonBtNodes(factory_);
    if (get_parameter("use_moveit").as_bool()) {
      registerMoveItBtNodes(factory_);
    } else {
      registerStubMotionBtNodes(factory_);
    }

    action_server_ = rclcpp_action::create_server<PickAction>(
      this, "pick_blueberry",
      std::bind(&PickActionServer::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&PickActionServer::handleCancel, this, std::placeholders::_1),
      std::bind(&PickActionServer::handleAccepted, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "Pick action server ready (use_moveit=%s, bt_ports=dump_pose)",
      get_parameter("use_moveit").as_bool() ? "true" : "false");
  }

private:
  rclcpp_action::GoalResponse handleGoal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const PickAction::Goal> goal)
  {
    if (goal->end_effector_mode != "suction" && goal->end_effector_mode != "vibration") {
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<GoalHandle>)
  {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(const std::shared_ptr<GoalHandle> goal_handle)
  {
    std::thread{std::bind(&PickActionServer::execute, this, goal_handle)}.detach();
  }

  void execute(const std::shared_ptr<GoalHandle> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<PickAction::Result>();

    const bool suction = goal->end_effector_mode == "suction";
    const std::string xml_param = suction ? "bt_xml_suction" : "bt_xml_vibration";
    std::string xml_path = get_parameter(xml_param).as_string();
    if (xml_path.empty()) {
      xml_path = suction ? "pick_suction.xml" : "pick_vibration.xml";
    }

    BT::Blackboard::Ptr blackboard = BT::Blackboard::create();
    blackboard->set("max_retries", goal->max_retries > 0 ? goal->max_retries : 3);

    geometry_msgs::msg::PoseStamped dump_pose;
    dump_pose.header.frame_id = "base_link";
    dump_pose.pose.position.x = 0.3;
    dump_pose.pose.position.y = -0.2;
    dump_pose.pose.position.z = 0.5;
    dump_pose.pose.orientation.w = 1.0;
    blackboard->set("dump_pose", dump_pose);

    PICK_PIPELINE_INFO(
      get_logger(), "pick_goal",
      "mode=%s max_retries=%d", goal->end_effector_mode.c_str(), goal->max_retries);

    BT::Tree tree;
    try {
      tree = factory_.createTreeFromFile(xml_path, blackboard);
    } catch (const std::exception & e) {
      result->success = false;
      result->message = std::string("BT load failed: ") + e.what();
      goal_handle->abort(result);
      return;
    }

    std::unique_ptr<BT::StdCoutLogger> bt_logger;
    if (get_parameter("debug_bt").as_bool()) {
      bt_logger = std::make_unique<BT::StdCoutLogger>(tree);
    }

    auto feedback = std::make_shared<PickAction::Feedback>();
    feedback->state = "running";
    goal_handle->publish_feedback(feedback);

    const auto status = tree.tickWhileRunning(std::chrono::milliseconds(100));

    if (status == BT::NodeStatus::SUCCESS) {
      result->success = true;
      result->message = "Pick completed";
      result->total_picked = suction ? 1 : 5;
      PICK_PIPELINE_INFO(
        get_logger(), "pick_result", "success=true picked=%d", result->total_picked);
      goal_handle->succeed(result);
    } else {
      result->success = false;
      result->message = "BT failed";
      result->total_picked = 0;
      PICK_PIPELINE_ERROR(
        get_logger(), "pick_result", "success=false status=%d", static_cast<int>(status));
      goal_handle->abort(result);
    }
  }

  BT::BehaviorTreeFactory factory_;
  rclcpp_action::Server<PickAction>::SharedPtr action_server_;
};

}  // namespace picking_task

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<picking_task::PickActionServer>();
  picking_task::RosNodeHolder::init(node);
  if (node->get_parameter("use_moveit").as_bool()) {
    picking_task::initMoveIt(node);
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
