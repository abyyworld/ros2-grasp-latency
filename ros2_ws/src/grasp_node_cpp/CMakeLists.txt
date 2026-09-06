cmake_minimum_required(VERSION 3.16)
project(grasp_node_cpp)

# A debug build would make the C++/Python comparison meaningless, so the
# default is pinned rather than inherited from whatever colcon was invoked with.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(grasp_msgs REQUIRED)
find_package(grasp_core_cpp REQUIRED)

add_executable(grasp_node src/grasp_node.cpp)
target_compile_features(grasp_node PRIVATE cxx_std_17)
target_link_libraries(grasp_node PRIVATE
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
  ${grasp_msgs_TARGETS}
  grasp_core_cpp::grasp_core_cpp
)

add_executable(frame_publisher src/frame_publisher.cpp)
target_compile_features(frame_publisher PRIVATE cxx_std_17)
target_link_libraries(frame_publisher PRIVATE
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
)

install(TARGETS grasp_node frame_publisher
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
