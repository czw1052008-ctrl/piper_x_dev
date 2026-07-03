cmake_minimum_required(VERSION 3.8)
project(picking_grasp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(picking_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}_lib
  src/suction_planner.cpp
  src/vibration_planner.cpp
)
ament_target_dependencies(${PROJECT_NAME}_lib geometry_msgs picking_msgs)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_lib Eigen3::Eigen)

add_executable(grasp_planner_node src/grasp_planner_node.cpp)
target_link_libraries(grasp_planner_node ${PROJECT_NAME}_lib)
ament_target_dependencies(grasp_planner_node rclcpp geometry_msgs picking_msgs)

install(TARGETS ${PROJECT_NAME}_lib grasp_planner_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_suction_planner test/test_suction_planner.cpp)
  target_link_libraries(test_suction_planner ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_suction_planner geometry_msgs picking_msgs)

  ament_add_gtest(test_vibration_planner test/test_vibration_planner.cpp)
  target_link_libraries(test_vibration_planner ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_vibration_planner geometry_msgs picking_msgs)
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_lib)
ament_package()
