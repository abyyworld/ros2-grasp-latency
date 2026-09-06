cmake_minimum_required(VERSION 3.16)
project(grasp_core_cpp)

# The pipeline itself is a plain CMake project under cpp/ with no ROS dependency
# of any kind -- that independence is what lets the same binary be measured
# in-process and inside a node. This package is only the ament shim that makes
# grasp_core::grasp_core visible to colcon-built packages; it deliberately
# contains no source of its own.
find_package(ament_cmake REQUIRED)
find_package(grasp_core REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE grasp_core::grasp_core)

install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(grasp_core)

ament_package()
