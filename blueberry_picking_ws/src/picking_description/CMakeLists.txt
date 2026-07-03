cmake_minimum_required(VERSION 3.8)
project(picking_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes calibration launch worlds models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
