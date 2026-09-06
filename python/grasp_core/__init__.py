"""The ROS-independent Python grasp pipeline.

`GraspPipeline` is the whole public surface: construct it with the three asset
paths, then call `run` once per frame. The rclpy node in
`ros2_ws/src/grasp_node_py` and `python/bench/bench_pipeline.py` are its only
callers, and they see the same object, so the code measured inside a node and
the code measured in process are the same code.
"""
from .pipeline import (STAGE_KEYS, GraspPipeline, GraspResult,
                       calibrate_timer_ns)

__all__ = ['GraspPipeline', 'GraspResult', 'STAGE_KEYS', 'calibrate_timer_ns']
