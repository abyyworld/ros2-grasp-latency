"""S5: grasp synthesis.

Small enough that the arithmetic is free and the only thing that matters is
getting the spec's tie-breaks right: the minor axis is the smaller eigenvalue
of the 2-D covariance, its sign comes from determinism rule 3 rather than from
whichever sign LAPACK happened to return, and the grasp height is clamped
against the refit plane rather than against z = 0.
"""
from __future__ import annotations

import numpy as np

from .config import canonicalise_columns

# The frame in step 4 is right-handed by construction; the check is here
# because a silently mirrored frame produces an IK solution that looks
# plausible and puts the gripper in backwards.
DETERMINANT_TOLERANCE = 1e-9


class GraspSynthesiser:
    __slots__ = ('_approach', '_depth', '_min_height', '_clearance',
                 '_max_width', '_capacity', '_cluster', '_projected',
                 '_wrist_reference', 'tcp', 'width')

    def __init__(self, grasp: dict):
        self._approach = np.array(grasp['approach_axis_base'], dtype=np.float64)
        self._depth = grasp['grasp_depth_m']
        self._min_height = grasp['min_height_above_plane_m']
        self._clearance = grasp['finger_clearance_m']
        self._max_width = grasp['max_width_m']
        self._capacity = 0
        self.tcp = np.zeros((4, 4), dtype=np.float64)
        self.tcp[3, 3] = 1.0
        self.tcp[:3, 2] = self._approach
        self._wrist_reference = np.array([1.0, 0.0])
        self.width = 0.0

    def set_wrist_reference(self, y_axis_at_neutral: np.ndarray) -> None:
        """Horizontal part of the TCP y axis at q_neutral, normalised.

        Supplied by the pipeline from one FK call at construction, because S5
        needs it and has no chain of its own. Never recomputed per frame.
        """
        horizontal = np.array([y_axis_at_neutral[0], y_axis_at_neutral[1]])
        norm = np.hypot(horizontal[0], horizontal[1])
        if norm == 0.0:
            raise ValueError('the TCP y axis at q_neutral is vertical, so it '
                             'cannot disambiguate a horizontal closing axis')
        self._wrist_reference = horizontal / norm

    def resize(self, capacity: int) -> None:
        self._capacity = capacity
        self._cluster = np.empty((capacity, 3), dtype=np.float64)
        self._projected = np.empty(capacity, dtype=np.float64)

    def run(self, points: np.ndarray, indices: np.ndarray,
            plane: np.ndarray, plane_found: bool) -> bool:
        count = indices.size
        cluster = self._cluster[:count]
        np.take(points, indices, axis=0, out=cluster)

        centroid = cluster.mean(axis=0)
        planar = cluster[:, :2] - centroid[:2]
        _, axes = np.linalg.eigh((planar.T @ planar) / count)
        canonicalise_columns(axes)
        minor = axes[:, 0]

        projected = self._projected[:count]
        np.matmul(cluster[:, :2], minor, out=projected)
        width = float(projected.max() - projected.min()) + self._clearance
        self.width = width
        if width > self._max_width:
            return False

        tcp = self.tcp
        # y closes the fingers along the narrow direction, z is the approach,
        # x completes a right-handed frame.
        norm = np.hypot(minor[0], minor[1])
        y_axis = tcp[:3, 1]
        y_axis[0] = minor[0] / norm
        y_axis[1] = minor[1] / norm
        y_axis[2] = 0.0
        # A parallel jaw closing along +y and along -y is the same grasp, so
        # the sign is free. Rule 3 picks it from the eigenvector, which knows
        # nothing about the arm; folding it toward the wrist's rest
        # orientation instead caps the demanded wrist rotation at 90 degrees
        # and names the identical grasp.
        reference = self._wrist_reference
        alignment = y_axis[0] * reference[0] + y_axis[1] * reference[1]
        if alignment < 0.0:
            y_axis[0] = -y_axis[0]
            y_axis[1] = -y_axis[1]
        elif alignment == 0.0 and (y_axis[0] < 0.0
                                   or (y_axis[0] == 0.0 and y_axis[1] < 0.0)):
            y_axis[0] = -y_axis[0]
            y_axis[1] = -y_axis[1]
        x_axis = tcp[:3, 0]
        x_axis[:] = np.cross(y_axis, self._approach)
        # det([x y z]) is the triple product x . (y x z), and y x z is the
        # vector just written into x, so this is that determinant and not a
        # shortcut around it.
        determinant = float(x_axis @ x_axis)
        if abs(determinant - 1.0) > DETERMINANT_TOLERANCE:
            raise ValueError(f'grasp frame is not right-handed: det {determinant}')

        z_grasp = float(cluster[:, 2].max()) - self._depth
        if plane_found:
            plane_z = (-plane[3] - plane[0] * centroid[0]
                       - plane[1] * centroid[1]) / plane[2]
            floor = plane_z + self._min_height
            if z_grasp < floor:
                z_grasp = floor
        tcp[0, 3] = centroid[0]
        tcp[1, 3] = centroid[1]
        tcp[2, 3] = z_grasp
        return True
