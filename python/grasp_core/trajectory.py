"""S7: quintic joint-space trajectory.

Every waypoint is an independent function of its own normalised time, so the
whole block is three outer products. The shape functions depend only on the
waypoint count, which is a config constant, so they are evaluated once at
construction and only the duration changes per frame.
"""
from __future__ import annotations

import numpy as np


class Waypoint:
    """One trajectory point, holding views into the planner's buffers.

    Views, not copies: the ROS node writes straight from `position` into the
    `array.array` its outgoing message already owns, so a per-frame copy here
    would be a second one nobody asked for.
    """

    __slots__ = ('position', 'velocity', 'acceleration', 'time_from_start')

    def __init__(self, position, velocity, acceleration):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.time_from_start = 0.0


class TrajectoryPlanner:
    __slots__ = ('_count', '_fraction', '_min_duration', '_max_duration',
                 '_limit', '_shape', '_shape_dot', '_shape_ddot', '_scaled',
                 '_delta', 'times', 'positions', 'velocities',
                 'accelerations', 'waypoints', 'duration')

    def __init__(self, trajectory: dict, chain):
        count = trajectory['waypoints']
        self._count = count
        self._fraction = trajectory['velocity_fraction']
        self._min_duration = trajectory['min_duration_s']
        self._max_duration = trajectory['max_duration_s']
        self._limit = chain.velocity * trajectory['velocity_fraction']

        s = np.arange(count, dtype=np.float64) / (count - 1)
        # h(s) = 10 s^3 - 15 s^4 + 6 s^5, and its first two derivatives with
        # respect to s. Writing the derivatives as coefficient times power
        # keeps them provably the derivative of the line above them.
        coefficient = np.array([10.0, -15.0, 6.0])
        power = np.array([3.0, 4.0, 5.0])
        self._shape = (coefficient * s[:, None] ** power).sum(axis=1)
        self._shape_dot = (coefficient * power
                           * s[:, None] ** (power - 1.0)).sum(axis=1)
        self._shape_ddot = (coefficient * power * (power - 1.0)
                            * s[:, None] ** (power - 2.0)).sum(axis=1)
        self._scaled = s

        dof = chain.dof
        self._delta = np.empty(dof, dtype=np.float64)
        self.times = np.empty(count, dtype=np.float64)
        self.positions = np.empty((count, dof), dtype=np.float64)
        self.velocities = np.empty((count, dof), dtype=np.float64)
        self.accelerations = np.empty((count, dof), dtype=np.float64)
        self.waypoints = [
            Waypoint(self.positions[k], self.velocities[k], self.accelerations[k])
            for k in range(count)]
        self.duration = 0.0

    def run(self, start: np.ndarray, goal: np.ndarray) -> float:
        delta = self._delta
        np.subtract(goal, start, out=delta)
        duration = float(np.max(np.abs(delta) / self._limit))
        if duration < self._min_duration:
            duration = self._min_duration
        elif duration > self._max_duration:
            duration = self._max_duration
        self.duration = duration

        np.multiply(self._shape[:, None], delta, out=self.positions)
        np.add(self.positions, start, out=self.positions)
        np.multiply((self._shape_dot / duration)[:, None], delta,
                    out=self.velocities)
        np.multiply((self._shape_ddot / (duration * duration))[:, None], delta,
                    out=self.accelerations)
        np.multiply(self._scaled, duration, out=self.times)

        for waypoint, moment in zip(self.waypoints, self.times.tolist()):
            waypoint.time_from_start = moment
        return duration
