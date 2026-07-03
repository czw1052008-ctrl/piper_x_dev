#pragma once

#include <behaviortree_cpp/bt_factory.h>
#include <rclcpp/rclcpp.hpp>

namespace picking_task
{

void initMoveIt(const rclcpp::Node::SharedPtr & node);
void registerMoveItBtNodes(BT::BehaviorTreeFactory & factory);

}  // namespace picking_task
