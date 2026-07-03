#pragma once

#include "rclcpp/rclcpp.hpp"

#define PICK_PIPELINE_INFO(logger, stage, fmt, ...) \
  RCLCPP_INFO(logger, "[pick_pipeline:%s] " fmt, stage, ##__VA_ARGS__)

#define PICK_PIPELINE_WARN(logger, stage, fmt, ...) \
  RCLCPP_WARN(logger, "[pick_pipeline:%s] " fmt, stage, ##__VA_ARGS__)

#define PICK_PIPELINE_ERROR(logger, stage, fmt, ...) \
  RCLCPP_ERROR(logger, "[pick_pipeline:%s] " fmt, stage, ##__VA_ARGS__)
