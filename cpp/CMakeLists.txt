cmake_minimum_required(VERSION 3.16)
project(grasp_core VERSION 0.1.0 LANGUAGES CXX)

# This project has no ROS dependency of any kind, which is what lets the same
# object code be measured in-process by bench_pipeline and inside an rclcpp
# node. ros2_ws/src/grasp_core_cpp is only an ament shim over the exported
# grasp_core::grasp_core target that this file installs.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Pinned rather than inherited: docs/TOOLCHAIN.md quotes these flags next to
# every latency figure, so a default that changed with the CMake version would
# silently invalidate a published number.
#
# CACHE, not a bare set(). A bare set() overrode the cache, so passing
# -DCMAKE_CXX_FLAGS_RELEASE on the command line was accepted and then ignored:
# all three of -O2, -O3 and -O3 -march=native produced a byte-identical binary.
# That is a bad property in any project and a disqualifying one here, where the
# headline is a ratio against this build. See results/compiler_flags.json:
# -march=native is 1.58x faster than the pinned default and moves the ratio
# from 1.65x to 2.60x.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG"
    CACHE STRING "Release flags. The published numbers use the default; see docs/METHOD.md.")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(grasp_core STATIC
  src/json.cpp
  src/config.cpp
  src/chain.cpp
  src/timing.cpp
  src/kinematics.cpp
  src/plane.cpp
  src/cluster.cpp
  src/trajectory.cpp
  src/pipeline.cpp
  src/sha256.cpp)
add_library(grasp_core::grasp_core ALIAS grasp_core)

target_include_directories(grasp_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(grasp_core PRIVATE Eigen3::Eigen)
target_compile_options(grasp_core PRIVATE -Wall -Wextra)
# The archive is linked into a ROS component library, which is shared.
set_target_properties(grasp_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(bench_pipeline bench/bench_pipeline.cpp)
target_link_libraries(bench_pipeline PRIVATE grasp_core)
target_compile_options(bench_pipeline PRIVATE -Wall -Wextra)

enable_testing()

set(GRASP_CORE_TESTS
  test_kinematics
  test_plane
  test_cluster
  test_trajectory
  test_pipeline
  test_no_allocation)

foreach(test_name IN LISTS GRASP_CORE_TESTS)
  add_executable(${test_name} tests/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE grasp_core Eigen3::Eigen)
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)
  # Assets are found by absolute path so ctest works from any directory.
  target_compile_definitions(${test_name} PRIVATE
    GRASP_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS grasp_core
  EXPORT grasp_coreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT grasp_coreTargets
  FILE grasp_coreTargets.cmake
  NAMESPACE grasp_core::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/grasp_core)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/grasp_coreConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/grasp_coreConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/grasp_core)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/grasp_coreConfigVersion.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/grasp_coreConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/grasp_coreConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/grasp_core)
