"""S6: forward kinematics and damped-least-squares IK.

This is the stage the fairness rule cannot rescue. The iteration is a fixed
point: every step needs the pose the previous step produced, so there is no
axis to vectorise over and Python pays interpreter overhead per NumPy call on
matrices of order 6. What can be batched is batched anyway: all seven
Rodrigues rotations in one expression, all seven joint frames in one batched
product, and the two damped solves the spec asks for share a single LU by
being posed as one right-hand side matrix.

The rest is written the way ALGORITHM.md writes it, including forming the
null-space projector explicitly. Rewriting `N @ (q_neutral - q)` as two
matrix-vector products is cheaper and mathematically identical, but it rounds
differently, and a different rounding can cross the convergence tolerance one
iteration earlier than the C++ does. `iterations` is part of the equivalence
gate, so the cheaper spelling is not worth its risk here.
"""
from __future__ import annotations

import math

import numpy as np


class IKSolver:
    __slots__ = ('_chain', '_dof', '_max_iterations', '_tol_p', '_tol_r',
                 '_damping', '_gain', '_max_step', '_skew', '_skew_squared',
                 '_sin', '_cos', '_rotation', '_scratch', '_local', '_frame',
                 '_transforms', '_axis_column', '_axes', '_error',
                 '_jacobian', '_normal', '_rhs', '_identity7',
                 '_damped', '_step', '_pull', '_projector', 'q', 'tcp',
                 'iterations', 'converged')

    def __init__(self, chain, ik: dict):
        self._chain = chain
        n = chain.dof
        self._dof = n
        self._max_iterations = ik['max_iterations']
        self._tol_p = ik['position_tolerance_m']
        self._tol_r = ik['orientation_tolerance_rad']
        self._damping = ik['damping']
        self._gain = ik['nullspace_gain']
        self._max_step = ik['max_step_rad']

        axis = chain.axis
        skew = np.zeros((n, 3, 3), dtype=np.float64)
        skew[:, 0, 1] = -axis[:, 2]
        skew[:, 0, 2] = axis[:, 1]
        skew[:, 1, 0] = axis[:, 2]
        skew[:, 1, 2] = -axis[:, 0]
        skew[:, 2, 0] = -axis[:, 1]
        skew[:, 2, 1] = axis[:, 0]
        self._skew = skew
        self._skew_squared = skew @ skew

        self._sin = np.empty(n, dtype=np.float64)
        self._cos = np.empty(n, dtype=np.float64)
        self._rotation = np.empty((n, 3, 3), dtype=np.float64)
        self._scratch = np.empty((n, 3, 3), dtype=np.float64)

        # Joint-local transform: the fixed parent-to-joint transform composed
        # with the joint rotation. Only the rotation block changes with q.
        self._local = np.zeros((n, 4, 4), dtype=np.float64)
        self._local[:] = chain.fixed
        self._frame = np.empty((n, 4, 4), dtype=np.float64)
        self._transforms = np.empty((n + 1, 4, 4), dtype=np.float64)
        self._transforms[0] = np.eye(4)
        self._axis_column = np.ascontiguousarray(axis[:, :, None])
        self._axes = np.empty((n, 3), dtype=np.float64)

        self._error = np.empty(6, dtype=np.float64)
        self._jacobian = np.empty((6, n), dtype=np.float64)
        self._normal = np.empty((6, 6), dtype=np.float64)
        self._damped = np.eye(6, dtype=np.float64) * (self._damping ** 2)
        self._rhs = np.empty((6, 1 + n), dtype=np.float64)
        self._identity7 = np.eye(n, dtype=np.float64)
        self._projector = np.empty((n, n), dtype=np.float64)
        self._step = np.empty(n, dtype=np.float64)
        self._pull = np.empty(n, dtype=np.float64)

        self.q = np.empty(n, dtype=np.float64)
        self.tcp = np.empty((4, 4), dtype=np.float64)
        self.iterations = 0
        self.converged = False

    def forward(self, q: np.ndarray) -> np.ndarray:
        """FK for the whole chain, leaving the joint frames in place for J."""
        rotation = self._rotation
        scratch = self._scratch
        np.sin(q, out=self._sin)
        np.cos(q, out=self._cos)
        # R = I + sin(t) K + (1 - cos(t)) K^2, all seven joints at once.
        np.multiply(self._skew, self._sin[:, None, None], out=rotation)
        np.subtract(1.0, self._cos, out=self._cos)
        np.multiply(self._skew_squared, self._cos[:, None, None], out=scratch)
        np.add(rotation, scratch, out=rotation)
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 2] += 1.0

        local = self._local
        np.matmul(self._chain.fixed[:, :3, :3], rotation, out=scratch)
        local[:, :3, :3] = scratch

        transforms = self._transforms
        for i in range(self._dof):
            np.matmul(transforms[i], local[i], out=transforms[i + 1])
        np.matmul(transforms[self._dof], self._chain.flange_to_tcp, out=self.tcp)

        # Joint frames, for the Jacobian: the frame a joint rotates in is its
        # parent's frame composed with the fixed offset, before its own
        # rotation is applied.
        frame = self._frame
        np.matmul(transforms[:self._dof], self._chain.fixed, out=frame)
        np.matmul(frame[:, :3, :3], self._axis_column,
                  out=self._axes[:, :, None])
        return self.tcp

    def solve(self, target: np.ndarray) -> bool:
        """Damped least squares from q_neutral, per ALGORITHM.md S6."""
        chain = self._chain
        n = self._dof
        q = self.q
        np.copyto(q, chain.q_neutral)

        position = target[:3, 3]
        rotation_desired = target[:3, :3]
        error = self._error
        error_position = error[:3]
        error_rotation = error[3:]
        jacobian = self._jacobian
        normal = self._normal
        rhs = self._rhs
        projector = self._projector
        step = self._step
        pull = self._pull
        tol_p = self._tol_p
        tol_r = self._tol_r
        gain = self._gain
        max_step = self._max_step
        lower = chain.lower
        upper = chain.upper
        neutral = chain.q_neutral

        for iteration in range(self._max_iterations):
            tcp = self.forward(q)
            np.subtract(position, tcp[:3, 3], out=error_position)
            # 0.5 sum_i cross(R_cur[:, i], R_des[:, i]), the standard
            # small-angle orientation error, columns as rows for np.cross.
            np.multiply(np.cross(tcp[:3, :3].T, rotation_desired.T).sum(axis=0),
                        0.5, out=error_rotation)
            if (math.sqrt(error_position @ error_position) < tol_p
                    and math.sqrt(error_rotation @ error_rotation) < tol_r):
                self.iterations = iteration
                self.converged = True
                return True

            axes = self._axes            # joint axes in base, from forward()
            jacobian[3:] = axes.T
            jacobian[:3] = np.cross(axes, tcp[:3, 3] - self._frame[:, :3, 3]).T

            np.matmul(jacobian, jacobian.T, out=normal)
            np.add(normal, self._damped, out=normal)
            rhs[:, 0] = error
            rhs[:, 1:] = jacobian
            # One LU, two solves: solving [e | J] together is bit-identical to
            # solving them apart, because LAPACK factorises once either way.
            solved = np.linalg.solve(normal, rhs)
            np.matmul(jacobian.T, solved[:, 0], out=step)
            np.matmul(jacobian.T, solved[:, 1:], out=projector)
            np.subtract(self._identity7, projector, out=projector)
            np.subtract(neutral, q, out=pull)
            np.matmul(projector, pull, out=pull)
            np.multiply(pull, gain, out=pull)
            np.add(step, pull, out=step)

            np.clip(step, -max_step, max_step, out=step)
            np.add(q, step, out=q)
            np.clip(q, lower, upper, out=q)

        self.iterations = self._max_iterations
        self.converged = False
        return False
