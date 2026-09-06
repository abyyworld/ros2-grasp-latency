# The ROS 2 half of the measurement. ROS is deliberately absent from the
# in-process benchmark, so this image is the only place the two nodes can run
# and the only place the DDS/executor/rclpy costs are visible.
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ros:jazzy-ros-base already carries build-essential, colcon, rosbag2 and
# common_interfaces. These are the additions the pipeline itself needs: Eigen
# for the C++ core, NumPy/SciPy for the Python core, and both rosbag2 storage
# plugins so a bag recorded on any machine can be replayed here.
# The lifecycle, component and launch packages all arrive with ros-base, three
# of them only transitively (rclcpp_components through rosbag2_transport,
# ros2lifecycle through ros2cli_common_extensions). They are named again here
# because this image will not build without them and a transitive dependency is
# not a promise.
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake \
      libeigen3-dev \
      python3-numpy \
      python3-scipy \
      python3-pip \
      python3-yaml \
      ros-jazzy-lifecycle-msgs \
      ros-jazzy-launch-ros \
      ros-jazzy-rclcpp-components \
      ros-jazzy-rclcpp-lifecycle \
      ros-jazzy-ros2lifecycle \
      ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-rosbag2-storage-sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Reads recorded /grasp/latency bags without rclpy, which is what lets the same
# converter run outside the container and be tested without ROS installed.
RUN pip3 install --no-cache-dir --break-system-packages rosbags==0.11.5

WORKDIR /workspace

# Copied piecemeal rather than as the whole tree: data/ holds a multi-gigabyte
# frame store that is regenerated rather than shipped, and belongs on a bind
# mount at run time (see docs/ROS2.md).
COPY assets/ /workspace/assets/
COPY cpp/ /workspace/cpp/
COPY python/ /workspace/python/
COPY tools/ /workspace/tools/
COPY harness/ /workspace/harness/
COPY docs/ /workspace/docs/
COPY docker/ /workspace/docker/
COPY ros2_ws/src/ /workspace/ros2_ws/src/

# The C++ core is a plain CMake project with no ROS dependency; building it the
# same way here as on the host is the point, so colcon never sees its sources.
# grasp_core_cpp then picks up the exported grasp_core::grasp_core target off
# CMAKE_PREFIX_PATH.
RUN cmake -S /workspace/cpp -B /tmp/build-core \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/grasp_core \
    && cmake --build /tmp/build-core --parallel "$(nproc)" \
    && cmake --install /tmp/build-core \
    && rm -rf /tmp/build-core

ENV CMAKE_PREFIX_PATH=/opt/grasp_core:${CMAKE_PREFIX_PATH}
ENV LD_LIBRARY_PATH=/opt/grasp_core/lib:${LD_LIBRARY_PATH}
# The Python core is imported straight from the repository, so the code the
# node measures is byte-for-byte the code the in-process benchmark measures.
ENV PYTHONPATH=/workspace/python:${PYTHONPATH}

RUN source /opt/ros/"${ROS_DISTRO}"/setup.bash \
    && colcon build \
         --base-paths /workspace/ros2_ws/src \
         --build-base /workspace/ros2_ws/build \
         --install-base /workspace/ros2_ws/install \
         --merge-install \
         --cmake-args -DCMAKE_BUILD_TYPE=Release

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
