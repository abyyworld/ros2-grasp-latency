"""S6: forward kinematics and damped-least-squares IK.

This is the stage the fairness rule cannot rescue. The iteration is a fixed
point: every step needs the pose the previous step produced, so there is no
axis to vectorise over, and NumPy is left paying interpreter and dispatch
overhead on matrices of order 6. Everything below is written to issue as few
array operations as possible rather than to move as few bytes as possible,
because at this size the call is the cost. Measured on the machine in
docs/TOOLCHAIN.md, one NumPy call on a small array is around 1.5 us and one
iteration of this loop is around 40 calls.

Three consequences of that, none of which change what is computed:

* `np.cross` costs 20 us on a 7 x 3 array, because it is a Python function that
  slices its way to an answer. The two cross products in the loop are written
  out instead, into the rows of the Jacobian they belong to.
* The orientation error `0.5 sum_i cross(R_cur[:, i], R_des[:, i])` is the
  antisymmetric part of `R_cur R_des^T`, which is one 3 x 3 product and three
  subtractions instead of a batched cross and a reduction.
* The two damped solves the spec asks for share one LU by being posed as a
  single right-hand side matrix. LAPACK factorises once either way, so the
  columns come back bit-identical to solving them apart.

The null-space projector is formed in full, as ALGORITHM.md writes it, even
though `N @ (q_neutral - q)` can be had for two matrix-vector products.
`iterations` is part of the equivalence gate, the cheaper spelling rounds
differently, and a different rounding can cross the convergence tolerance an
iteration earlier than the C++ does.
"""
from __future__ import annotations

import math

import numpy as np


class IKSolver:
    __slots__ = ('_chain', '_dof', '_max_iterations', '_tol_p', '_tol_r',
                 '_damping', '_gain', '_max_step', '_skew', '_skew_squared',
                 '_identity3', '_sin', '_cos', '_rotation', '_scratch',
                 '_local', '_frame', '_transforms', '_axis_column', '_axes',
                 '_axes_out', '_delta', '_product', '_error', '_jacobian',
                 '_cross_term', '_normal', '_rhs', '_identity7', '_damped',
                 '_projector', '_step', '_pull', 'q', 'tcp', 'iterations',
                 'converged')

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
        self._identity3 = np.eye(3, dtype=np.float64)

        self._sin = np.empty(n, dtype=np.float64)
        self._cos = np.empty(n, dtype=np.float64)
        self._rotation = np.empty((n, 3, 3), dtype=np.float64)
        self._scratch = np.empty((n, 3, 3), dtype=np.float64)

        # Joint-local transform: the fixed parent-to-joint transform composed
        # with the joint's own rotation. Only the rotation block moves with q,
        # so the translation and the bottom row are written once, here.
        self._local = np.zeros((n, 4, 4), dtype=np.float64)
        self._local[:] = chain.fixed
        self._frame = np.empty((n, 4, 4), dtype=np.float64)
        self._transforms = np.empty((n + 1, 4, 4), dtype=np.float64)
        self._transforms[0] = np.eye(4)
        self._axis_column = np.ascontiguousarray(axis[:, :, None])

        self._error = np.empty(6, dtype=np.float64)
        self._jacobian = np.empty((6, n), dtype=np.float64)
        # J_w is the joint axes in base, so forward() writes them straight into
        # the bottom half of the Jacobian instead of into a buffer that would
        # then have to be transposed into it.
        self._axes = self._jacobian[3:]
        self._axes_out = self._axes.T[:, :, None]
        self._delta = np.empty((3, n), dtype=np.float64)
        self._cross_term = np.empty(n, dtype=np.float64)
        self._product = np.empty((3, 3), dtype=np.float64)
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
        # Rodrigues: R = I + sin(t) K + (1 - cos(t)) K^2, all seven at once.
        np.multiply(self._skew, self._sin[:, None, None], out=rotation)
        np.subtract(1.0, self._cos, out=self._cos)
        np.multiply(self._skew_squared, self._cos[:, None, None], out=scratch)
        np.add(rotation, scratch, out=rotation)
        np.add(rotation, self._identity3, out=rotation)

        local = self._local
        np.matmul(self._chain.fixed[:, :3, :3], rotation, out=scratch)
        local[:, :3, :3] = scratch

        transforms = self._transforms
        for i in range(self._dof):
            np.matmul(transforms[i], local[i], out=transforms[i + 1])
        np.matmul(transforms[self._dof], self._chain.flange_to_tcp, out=self.tcp)

        # The frame a joint rotates in is its parent's, composed with the fixed
        # offset and taken before its own rotation.
        np.matmul(transforms[:self._dof], self._chain.fixed, out=self._frame)
        np.matmul(self._frame[:, :3, :3], self._axis_column, out=self._axes_out)
        return self.tcp

    def solve(self, target: np.ndarray) -> bool:
        """Damped least squares from q_neutral, per ALGORITHM.md S6."""
        chain = self._chain
        q = self.q
        np.copyto(q, chain.q_neutral)

        position = target[:3, 3]
        rotation_desired = target[:3, :3]
        error = self._error
        error_position = error[:3]
        error_rotation = error[3:]
        jacobian = self._jacobian
        axes = self._axes
        delta = self._delta
        cross_term = self._cross_term
        product = self._product
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
        origins = self._frame[:, :3, 3].T

        for iteration in range(self._max_iterations):
            tcp = self.forward(q)
            reached = tcp[:3, 3]
            np.subtract(position, reached, out=error_position)
            # 0.5 sum_i cross(R_cur[:, i], R_des[:, i]) is the vee of the
            # antisymmetric part of R_cur R_des^T: the sums the cross products
            # would take are exactly that product's entries.
            np.matmul(tcp[:3, :3], rotation_desired.T, out=product)
            error_rotation[0] = 0.5 * (product[1, 2] - product[2, 1])
            error_rotation[1] = 0.5 * (product[2, 0] - product[0, 2])
            error_rotation[2] = 0.5 * (product[0, 1] - product[1, 0])
            if (math.sqrt(error_position @ error_position) < tol_p
                    and math.sqrt(error_rotation @ error_rotation) < tol_r):
                self.iterations = iteration
                self.converged = True
                return True

            # J_v[:, i] = cross(z_i, p_tcp - o_i), written out per component
            # because np.cross on an array this small costs more than the
            # fifteen operations it hides.
            np.subtract(reached[:, None], origins, out=delta)
            np.multiply(axes[1], delta[2], out=jacobian[0])
            np.multiply(axes[2], delta[1], out=cross_term)
            np.subtract(jacobian[0], cross_term, out=jacobian[0])
            np.multiply(axes[2], delta[0], out=jacobian[1])
            np.multiply(axes[0], delta[2], out=cross_term)
            np.subtract(jacobian[1], cross_term, out=jacobian[1])
            np.multiply(axes[0], delta[1], out=jacobian[2])
            np.multiply(axes[1], delta[0], out=cross_term)
            np.subtract(jacobian[2], cross_term, out=jacobian[2])

            np.matmul(jacobian, jacobian.T, out=normal)
            np.add(normal, self._damped, out=normal)
            rhs[:, 0] = error
            rhs[:, 1:] = jacobian
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
