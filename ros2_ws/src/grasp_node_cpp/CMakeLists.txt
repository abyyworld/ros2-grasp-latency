cmake_minimum_required(VERSION 3.16)
project(grasp_node_cpp)

# A debug build would make the C++/Python comparison meaningless, so the
# default is pinned rather than inherited from whatever colcon was invoked with.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(grasp_msgs REQUIRED)
find_package(grasp_core_cpp REQUIRED)

# Both nodes are built as component libraries first and wrapped in an
# executable second. The library is what a container loads, and loading both of
# them into one container is the only way the depth image reaches the consumer
# without a trip through the RMW; the executables are the inter-process arm the
# intra-process numbers are compared against.
add_library(grasp_node_component SHARED src/grasp_node.cpp)
target_compile_features(grasp_node_component PUBLIC cxx_std_17)
target_include_directories(grasp_node_component PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
target_link_libraries(grasp_node_component PUBLIC
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${lifecycle_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${trajectory_msgs_TARGETS}
  ${grasp_msgs_TARGETS}
  grasp_core_cpp::grasp_core_cpp
)
target_link_libraries(grasp_node_component PRIVATE rclcpp_components::component)
rclcpp_components_register_nodes(grasp_node_component "grasp_node_cpp::GraspNode")

add_library(frame_publisher_component SHARED src/frame_publisher.cpp)
target_compile_features(frame_publisher_component PUBLIC cxx_std_17)
target_include_directories(frame_publisher_component PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
target_link_libraries(frame_publisher_component PUBLIC
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${lifecycle_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
)
target_link_libraries(frame_publisher_component PRIVATE rclcpp_components::component)
rclcpp_components_register_nodes(frame_publisher_component "grasp_node_cpp::FramePublisher")

add_executable(grasp_node src/grasp_node_main.cpp)
target_link_libraries(grasp_node PRIVATE grasp_node_component)

add_executable(frame_publisher src/frame_publisher_main.cpp)
target_link_libraries(frame_publisher PRIVATE frame_publisher_component)

# rclcpp_components_register_nodes records the library as lib/<file name>, so
# the shared objects have to land in lib and not in lib/${PROJECT_NAME}.
install(TARGETS grasp_node_component frame_publisher_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS grasp_node frame_publisher
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
