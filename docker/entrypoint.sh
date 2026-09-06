#!/usr/bin/env bash
set -e
# shellcheck disable=SC1090,SC1091  # sourced at run time, inside the image
source /opt/ros/"${ROS_DISTRO}"/setup.bash
# shellcheck disable=SC1091
source /workspace/ros2_ws/install/setup.bash
exec "$@"
