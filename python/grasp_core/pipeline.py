"""The eight stages of ALGORITHM.md, wired together and instrumented.

Everything the stages need is allocated when the resolution is first seen, so a
steady-state frame allocates only what NumPy allocates for the temporaries it
will not let you preallocate. The garbage collector stays on: a real rclpy node
runs with it on, and turning it off here would hide exactly the tail this
repository exists to measure. What the pipeline does instead is let the caller
read the collection counters per frame, so a slow frame can be attributed
rather than guessed at.

Timing marks are plain locals, not a NumPy array: a `float64` store through a
one-element array costs more than the clock read it is recording.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .cluster import Clusterer
from .config import (COLOUR_DTYPE, DEPTH_DTYPE, Chain, load_deviates,
                     load_json)
from .deproject import Cropper, Deprojector
from .grasp import GraspSynthesiser
from .kinematics import IKSolver
from .plane import PlaneRemover
from .trajectory import TrajectoryPlanner

# The key order docs/FORMATS.md fixes for stage_ns, which is also the order
# grasp_msgs/GraspLatency indexes its stage array in.
STAGE_KEYS = ('decode', 'deproject', 'transform_crop', 'plane',
              'cluster', 'grasp', 'ik', 'traj')

_COLOUR_CHANNELS = 3


def calibrate_timer_ns(samples: int = 4096) -> dict:
    """Cost of reading the clock, so a stage total can be judged against it.

    Reported rather than subtracted. Eight marks per frame at a few tens of
    nanoseconds each is far below the resolution of anything measured here, and
    a reader who wants it removed has the number.
    """
    clock = time.perf_counter_ns
    deltas = np.empty(samples, dtype=np.int64)
    previous = clock()
    for i in range(samples):
        now = clock()
        deltas[i] = now - previous
        previous = now
    return {
        'samples': samples,
        'median_ns': float(np.median(deltas)),
        'min_ns': int(deltas.min()),
        'p99_ns': float(np.percentile(deltas, 99.0)),
        'resolution_ns': float(time.get_clock_info('perf_counter').resolution * 1e9),
    }


class GraspResult:
    """One frame's answer. Reused across frames; copy what you keep."""

    __slots__ = ('stage_ns', 'total_ns', 'points', 'cluster_points',
                 'plane_found', 'plane', 'graspable', 'width', 'tcp',
                 'converged', 'iterations', 'q', 'duration_s', 'trajectory',
                 'positions', 'velocities', 'accelerations', 'times')

    def __init__(self):
        self.stage_ns = dict.fromkeys(STAGE_KEYS, 0)
        self.total_ns = 0
        self.points = 0
        self.cluster_points = 0
        self.plane_found = False
        self.graspable = False
        self.converged = False
        self.iterations = 0
        self.width = 0.0
        self.duration_s = 0.0


class GraspPipeline:
    """S0 to S7 for one RGB-D frame.

    Construction reads the config, the chain and the RANSAC deviate table, and
    nothing else ever does: determinism rule 5.
    """

    def __init__(self, config_path, chain_path, ransac_table_path):
        config = load_json(config_path)
        chain = Chain(load_json(chain_path))
        deviates = load_deviates(ransac_table_path,
                                 config['ransac_table']['count'])

        self.config = config
        self.chain = chain
        self._deprojector = Deprojector(config['camera'], config['deproject'])
        self._cropper = Cropper(config['camera'], config['workspace'])
        self._plane = PlaneRemover(
            config['plane'], deviates,
            config['implementation']['ransac_block_points'])
        self._clusterer = Clusterer(config['cluster'],
                                    self._voxel_capacity(config))
        self._grasp = GraspSynthesiser(config['grasp'])
        self._ik = IKSolver(chain, config['ik'])
        self._planner = TrajectoryPlanner(config['trajectory'], chain)
        # S5 resolves the closing-axis sign against the wrist's rest
        # orientation, which is one FK call and belongs out here, at
        # construction, where determinism rule 5 wants it.
        self._grasp.set_wrist_reference(self._ik.forward(chain.q_neutral)[:3, 1])

        self._shape = None
        result = GraspResult()
        result.plane = self._plane.plane
        result.tcp = self._grasp.tcp
        result.q = self._ik.q
        result.trajectory = self._planner.waypoints
        result.positions = self._planner.positions
        result.velocities = self._planner.velocities
        result.accelerations = self._planner.accelerations
        result.times = self._planner.times
        self._result = result

    @staticmethod
    def _voxel_capacity(config) -> int:
        """Worst-case voxel count: an object cloud filling the whole crop."""
        workspace = config['workspace']
        voxel = config['cluster']['voxel_size_m']
        span = 1
        for axis in ('x', 'y', 'z'):
            extent = workspace[f'{axis}_max'] - workspace[f'{axis}_min']
            span *= int(math.ceil(extent / voxel)) + 2
        return span

    def resize(self, height: int, width: int) -> None:
        """Bind every buffer to a resolution. Called on the first frame."""
        stride = self.config['deproject']['stride']
        capacity = (((height + stride - 1) // stride)
                    * ((width + stride - 1) // stride))
        self._deprojector.resize(height, width)
        self._cropper.resize(capacity)
        self._plane.resize(capacity)
        self._clusterer.resize(capacity)
        self._grasp.resize(capacity)
        self._shape = (height, width)

    def run(self, depth, rgb, width: int = None, height: int = None):
        """One frame, S0 through S7.

        `depth` and `rgb` are either buffers of the bytes docs/FORMATS.md
        defines, in which case S0 measures the cost of getting arrays out of
        them, or arrays already, which is what the rclpy node hands over after
        wrapping the message data.
        """
        clock = time.perf_counter_ns
        result = self._result

        mark0 = clock()
        if width is None or height is None:
            height, width = depth.shape[0], depth.shape[1]
        if isinstance(depth, np.ndarray):
            depth_image = depth.reshape(height, width)
        else:
            depth_image = np.frombuffer(depth, dtype=DEPTH_DTYPE).reshape(
                height, width)
        if isinstance(rgb, np.ndarray):
            colour = rgb.reshape(height, width, _COLOUR_CHANNELS)
        else:
            colour = np.frombuffer(rgb, dtype=COLOUR_DTYPE).reshape(
                height, width, _COLOUR_CHANNELS)
        mark1 = clock()

        if self._shape != (height, width):
            self.resize(height, width)
            mark0 = mark1 = clock()

        points_cam = self._deprojector.run(depth_image)
        mark2 = clock()
        points_base = self._cropper.run(points_cam)
        mark3 = clock()
        points_object, plane_found = self._plane.run(points_base)
        mark4 = clock()
        indices = self._clusterer.run(points_object)
        mark5 = clock()

        if indices.size > 0:
            graspable = self._grasp.run(points_object, indices,
                                        self._plane.plane, plane_found)
            grasp_width = self._grasp.width
        else:
            graspable = False
            grasp_width = 0.0
        mark6 = clock()

        if graspable:
            converged = self._ik.solve(self._grasp.tcp)
            iterations = self._ik.iterations
        else:
            converged = False
            iterations = 0
            np.copyto(self._ik.q, self.chain.q_neutral)
        mark7 = clock()

        if graspable:
            duration = self._planner.run(self.chain.q_neutral, self._ik.q)
        else:
            duration = 0.0
            self._planner.positions[...] = 0.0
            self._planner.velocities[...] = 0.0
            self._planner.accelerations[...] = 0.0
            self._planner.times[...] = 0.0
        mark8 = clock()

        stage_ns = result.stage_ns
        stage_ns['decode'] = mark1 - mark0
        stage_ns['deproject'] = mark2 - mark1
        stage_ns['transform_crop'] = mark3 - mark2
        stage_ns['plane'] = mark4 - mark3
        stage_ns['cluster'] = mark5 - mark4
        stage_ns['grasp'] = mark6 - mark5
        stage_ns['ik'] = mark7 - mark6
        stage_ns['traj'] = mark8 - mark7
        result.total_ns = mark8 - mark0
        result.points = points_base.shape[0]
        result.cluster_points = int(indices.size)
        result.plane_found = plane_found
        result.graspable = graspable
        result.width = grasp_width
        result.converged = converged
        result.iterations = iterations
        result.duration_s = duration
        return result
