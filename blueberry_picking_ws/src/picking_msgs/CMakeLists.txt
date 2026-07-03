cmake_minimum_required(VERSION 3.8)
project(picking_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedBerry.msg"
  "msg/DetectedBerryArray.msg"
  "msg/PerceptionStatus.msg"
  "msg/SuctionGraspPlan.msg"
  "msg/VibrationPlan.msg"
  "srv/TriggerGlobalDetection.srv"
  "srv/TriggerFineDetection.srv"
  "srv/PlanSuction.srv"
  "srv/PlanVibration.srv"
  "action/PickBlueberry.action"
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
