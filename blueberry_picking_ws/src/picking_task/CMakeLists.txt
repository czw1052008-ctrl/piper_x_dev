cmake_minimum_required(VERSION 3.8)
project(picking_task)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(picking_msgs REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}_moveit src/moveit_bt_nodes.cpp)
ament_target_dependencies(
  ${PROJECT_NAME}_moveit
  rclcpp geometry_msgs picking_msgs behaviortree_cpp moveit_ros_planning_interface moveit_core moveit_msgs)

add_executable(pick_action_server src/pick_action_server.cpp)
target_link_libraries(pick_action_server ${PROJECT_NAME}_moveit)
ament_target_dependencies(
  pick_action_server rclcpp rclcpp_action geometry_msgs picking_msgs behaviortree_cpp)

install(TARGETS ${PROJECT_NAME}_moveit pick_action_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY bt_xml config launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_bt_xml_load test/test_bt_xml_load.cpp)
  ament_target_dependencies(test_bt_xml_load behaviortree_cpp)
  target_compile_definitions(test_bt_xml_load PRIVATE
    "PICKING_BT_XML_SUCTION=\"${CMAKE_CURRENT_SOURCE_DIR}/bt_xml/pick_suction.xml\"")
endif()

ament_package()
