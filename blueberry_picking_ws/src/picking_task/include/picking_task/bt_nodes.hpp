#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/action_node.h>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "picking_msgs/srv/plan_suction.hpp"
#include "picking_msgs/srv/plan_vibration.hpp"
#include "picking_msgs/srv/trigger_fine_detection.hpp"
#include "picking_msgs/srv/trigger_global_detection.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/rclcpp.hpp"

namespace picking_task
{

class RosNodeHolder
{
public:
  static void init(const rclcpp::Node::SharedPtr & node) {node_ = node;}
  static rclcpp::Node::SharedPtr get() {return node_;}

private:
  static rclcpp::Node::SharedPtr node_;
};

rclcpp::Node::SharedPtr RosNodeHolder::node_ = nullptr;

inline void logPoseStamped(
  const char * label, const geometry_msgs::msg::PoseStamped & ps)
{
  RCLCPP_INFO(
    RosNodeHolder::get()->get_logger(),
    "%s [%s] pos=(%.3f, %.3f, %.3f)",
    label, ps.header.frame_id.c_str(),
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
}

template<typename ServiceT>
class ServiceBTNode : public BT::SyncActionNode
{
public:
  ServiceBTNode(const std::string & name, const BT::NodeConfig & config, std::string service_name)
  : BT::SyncActionNode(name, config), service_name_(std::move(service_name)) {}

protected:
  bool waitAndCall(std::function<void(typename ServiceT::Request::SharedPtr)> fill_request,
    std::function<bool(typename ServiceT::Response::SharedPtr)> handle_response)
  {
    if (!client_) {
      client_ = RosNodeHolder::get()->create_client<ServiceT>(service_name_);
    }
    if (!client_->wait_for_service(std::chrono::seconds(2))) {
      RCLCPP_ERROR(
        RosNodeHolder::get()->get_logger(), "[%s] service %s unavailable",
        name().c_str(), service_name_.c_str());
      return false;
    }
    auto req = std::make_shared<typename ServiceT::Request>();
    fill_request(req);
    auto future = client_->async_send_request(req);
    if (future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      RCLCPP_ERROR(
        RosNodeHolder::get()->get_logger(), "[%s] service %s timed out",
        name().c_str(), service_name_.c_str());
      return false;
    }
    const auto res = future.get();
    if (!handle_response(res)) {
      RCLCPP_WARN(
        RosNodeHolder::get()->get_logger(), "[%s] service %s failed",
        name().c_str(), service_name_.c_str());
      return false;
    }
    return true;
  }

  typename rclcpp::Client<ServiceT>::SharedPtr client_;
  std::string service_name_;
};

class GlobalDetectNode : public ServiceBTNode<picking_msgs::srv::TriggerGlobalDetection>
{
public:
  using ServiceBTNode::ServiceBTNode;

  BT::NodeStatus tick() override
  {
    geometry_msgs::msg::PoseStamped cluster_pose;
    constexpr int kMaxAttempts = 3;
    bool ok = false;
    for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
      ok = waitAndCall(
        [](auto) {},
        [&](auto res) {
          if (!res->success) {
            RCLCPP_WARN(
              RosNodeHolder::get()->get_logger(),
              "GlobalDetect failed (attempt %d/%d): %s",
              attempt, kMaxAttempts, res->message.c_str());
            return false;
          }
          cluster_pose = res->cluster_pose;
          return true;
        });
      if (ok) {
        break;
      }
      if (attempt < kMaxAttempts) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
      }
    }
    if (!ok) {return BT::NodeStatus::FAILURE;}
    logPoseStamped("GlobalDetect cluster_pose", cluster_pose);
    setOutput("cluster_pose", cluster_pose);
    return BT::NodeStatus::SUCCESS;
  }

  static BT::PortsList providedPorts()
  {
    return {BT::OutputPort<geometry_msgs::msg::PoseStamped>("cluster_pose")};
  }
};

class FineDetectNode : public ServiceBTNode<picking_msgs::srv::TriggerFineDetection>
{
public:
  using ServiceBTNode::ServiceBTNode;

  BT::NodeStatus tick() override
  {
    picking_msgs::msg::DetectedBerry berry;
    const bool ok = waitAndCall(
      [](auto) {},
      [&](auto res) {
        if (!res->success || res->detected_berries.empty()) {return false;}
        berry = res->detected_berries.front();
        return true;
      });
    if (!ok) {return BT::NodeStatus::FAILURE;}
    setOutput("berry_pose", berry);
    return BT::NodeStatus::SUCCESS;
  }

  static BT::PortsList providedPorts()
  {
    return {BT::OutputPort<picking_msgs::msg::DetectedBerry>("berry_pose")};
  }
};

class ClusterAnalyzeNode : public ServiceBTNode<picking_msgs::srv::TriggerFineDetection>
{
public:
  using ServiceBTNode::ServiceBTNode;

  BT::NodeStatus tick() override
  {
    std::vector<picking_msgs::msg::DetectedBerry> berries;
    geometry_msgs::msg::Vector3 stem_direction;
    geometry_msgs::msg::PoseStamped contact_pose;
    bool stem_valid = false;
    bool contact_valid = false;
    const bool ok = waitAndCall(
      [](auto) {},
      [&](auto res) {
        if (!res->success || res->detected_berries.size() < 2) {
          RCLCPP_WARN(
            RosNodeHolder::get()->get_logger(),
            "ClusterAnalyze failed: %s (berries=%zu)",
            res->message.c_str(), res->detected_berries.size());
          return false;
        }
        berries = res->detected_berries;
        stem_valid = res->stem_direction_valid;
        stem_direction = res->stem_direction;
        contact_valid = res->contact_pose_valid;
        contact_pose = res->contact_pose;
        return true;
      });
    if (!ok) {return BT::NodeStatus::FAILURE;}
    if (!stem_valid || !contact_valid) {
      RCLCPP_WARN(
        RosNodeHolder::get()->get_logger(),
        "ClusterAnalyze reject: stem_valid=%s contact_valid=%s berries=%zu",
        stem_valid ? "true" : "false",
        contact_valid ? "true" : "false",
        berries.size());
      return BT::NodeStatus::FAILURE;
    }
    RCLCPP_INFO(
      RosNodeHolder::get()->get_logger(),
      "ClusterAnalyze: %zu berries contact=(%.3f, %.3f, %.3f) stem=(%.3f, %.3f, %.3f)",
      berries.size(),
      contact_pose.pose.position.x, contact_pose.pose.position.y, contact_pose.pose.position.z,
      stem_direction.x, stem_direction.y, stem_direction.z);
    setOutput("cluster_info", berries);
    setOutput("contact_pose", contact_pose);
    setOutput("stem_direction", stem_direction);
    return BT::NodeStatus::SUCCESS;
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::OutputPort<std::vector<picking_msgs::msg::DetectedBerry>>("cluster_info"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("contact_pose"),
      BT::OutputPort<geometry_msgs::msg::Vector3>("stem_direction"),
    };
  }
};

class PlanSuctionNode : public ServiceBTNode<picking_msgs::srv::PlanSuction>
{
public:
  using ServiceBTNode::ServiceBTNode;

  BT::NodeStatus tick() override
  {
    auto berry = getInput<picking_msgs::msg::DetectedBerry>("berry_pose");
    if (!berry) {return BT::NodeStatus::FAILURE;}
    picking_msgs::msg::SuctionGraspPlan plan;
    const bool ok = waitAndCall(
      [&](auto req) {req->berries = {berry.value()};},
      [&](auto res) {
        if (!res->success) {return false;}
        plan = res->plan;
        return true;
      });
    if (!ok) {return BT::NodeStatus::FAILURE;}
    setOutput("suction_plan", plan);
    setOutput("pre_grasp", plan.pre_grasp);
    setOutput("grasp", plan.grasp);
    setOutput("post_grasp", plan.post_grasp);
    return BT::NodeStatus::SUCCESS;
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<picking_msgs::msg::DetectedBerry>("berry_pose"),
      BT::OutputPort<picking_msgs::msg::SuctionGraspPlan>("suction_plan"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("pre_grasp"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("grasp"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("post_grasp")
    };
  }
};

class PlanVibrationNode : public ServiceBTNode<picking_msgs::srv::PlanVibration>
{
public:
  using ServiceBTNode::ServiceBTNode;

  BT::NodeStatus tick() override
  {
    auto berries = getInput<std::vector<picking_msgs::msg::DetectedBerry>>("cluster_info");
    auto contact = getInput<geometry_msgs::msg::PoseStamped>("contact_pose");
    auto stem = getInput<geometry_msgs::msg::Vector3>("stem_direction");
    if (!berries || !contact || !stem) {return BT::NodeStatus::FAILURE;}
    picking_msgs::msg::VibrationPlan plan;
    const bool ok = waitAndCall(
      [&](auto req) {
        req->berries = berries.value();
        req->contact_pose = contact.value();
        req->contact_pose_valid = true;
        req->stem_direction = stem.value();
        req->stem_direction_valid = true;
      },
      [&](auto res) {
        if (!res->success) {
          RCLCPP_WARN(
            RosNodeHolder::get()->get_logger(), "PlanVibration failed: %s",
            res->message.c_str());
          return false;
        }
        plan = res->plan;
        return true;
      });
    if (!ok) {return BT::NodeStatus::FAILURE;}
    RCLCPP_INFO(
      RosNodeHolder::get()->get_logger(), "PlanVibration: %s",
      plan.perception_valid ? "perception_ok" : "perception_invalid");
    logPoseStamped("slot_pose", plan.slot_pose);
    logPoseStamped("pre_approach", plan.pre_approach);
    setOutput("vib_plan", plan);
    setOutput("pre_approach", plan.pre_approach);
    setOutput("slot_pose", plan.slot_pose);
    setOutput("retract_pose", plan.retract_pose);
    setOutput("dump_pose", plan.dump_pose);
    return BT::NodeStatus::SUCCESS;
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::vector<picking_msgs::msg::DetectedBerry>>("cluster_info"),
      BT::InputPort<geometry_msgs::msg::PoseStamped>("contact_pose"),
      BT::InputPort<geometry_msgs::msg::Vector3>("stem_direction"),
      BT::OutputPort<picking_msgs::msg::VibrationPlan>("vib_plan"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("pre_approach"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("slot_pose"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("retract_pose"),
      BT::OutputPort<geometry_msgs::msg::PoseStamped>("dump_pose")
    };
  }
};

class LogPoseNode : public BT::SyncActionNode
{
public:
  LogPoseNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto target = getInput<geometry_msgs::msg::PoseStamped>("target");
    if (target) {
      RCLCPP_INFO(
        RosNodeHolder::get()->get_logger(), "MoveTo target z=%.3f",
        target->pose.position.z);
    }
    return BT::NodeStatus::SUCCESS;
  }

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<geometry_msgs::msg::PoseStamped>("target")};
  }
};

class ActivateEefNode : public BT::SyncActionNode
{
public:
  ActivateEefNode(const std::string & name, const BT::NodeConfig & config)
  : BT::SyncActionNode(name, config) {}

  BT::NodeStatus tick() override
  {
    auto enable = getInput<bool>("enable");
    RCLCPP_INFO(
      RosNodeHolder::get()->get_logger(), "%s activate=%s",
      name().c_str(), enable && enable.value() ? "true" : "false");
    return BT::NodeStatus::SUCCESS;
  }

  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<bool>("enable")};
  }
};

inline void registerCommonBtNodes(BT::BehaviorTreeFactory & factory)
{
  factory.registerNodeType<GlobalDetectNode>("GlobalDetect", "trigger_global_detection");
  factory.registerNodeType<FineDetectNode>("FineDetect", "trigger_fine_detection");
  factory.registerNodeType<ClusterAnalyzeNode>("ClusterAnalyze", "trigger_fine_detection");
  factory.registerNodeType<PlanSuctionNode>("PlanSuction", "plan_suction");
  factory.registerNodeType<PlanVibrationNode>("PlanVibration", "plan_vibration");
  factory.registerNodeType<ActivateEefNode>("ActivateSuction");
  factory.registerNodeType<ActivateEefNode>("ActivateVibration");
  factory.registerNodeType<LogPoseNode>("DumpBerries");
}

inline void registerStubMotionBtNodes(BT::BehaviorTreeFactory & factory)
{
  factory.registerNodeType<LogPoseNode>("MoveToPose");
  factory.registerNodeType<LogPoseNode>("CartesianMove");
}

inline void registerBtNodes(BT::BehaviorTreeFactory & factory)
{
  registerCommonBtNodes(factory);
  registerStubMotionBtNodes(factory);
}

}  // namespace picking_task
