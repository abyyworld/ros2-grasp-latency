"""S1 deprojection and S2 transform plus crop.

Both stages are pure array arithmetic over every pixel, which is where NumPy is
supposed to be at its best, so this is the part of the pipeline the fairness
rule protects: there is no Python-level loop over pixels anywhere below, only
whole-image operations into buffers allocated once per resolution.

Storage is float32 and arithmetic is float64, which is determinism rule 1 and
is also what a C++ implementation that keeps a `struct Point { float x, y, z; }`
does when its intrinsics are doubles. Getting that split wrong is not a
rounding curiosity: a point that lands 1e-7 m on the other side of the RANSAC
inlier threshold changes a score, and a changed score can change which
candidate plane wins.
"""
from __future__ import annotations

import numpy as np


class Deprojector:
    """S1: depth image to `points_cam`, in row-major pixel order."""

    __slots__ = ('_stride', '_z_min', '_z_max', '_depth_scale', '_ref_height',
                 '_f0', '_shape', 'fx', 'fy', 'cx', 'cy',
                 '_u_minus_cx', '_v_minus_cy', '_z', '_lo', '_hi', '_scratch',
                 '_work', '_points')

    def __init__(self, camera: dict, deproject: dict):
        # Square pixels, so one focal length, and it is the vertical one
        # because that is the axis the sensor mode does not crop. See S1.
        self._ref_height = camera['reference_height']
        self._f0 = camera['fy']
        self._depth_scale = camera['depth_scale_m']
        self._stride = deproject['stride']
        self._z_min = deproject['z_min_m']
        self._z_max = deproject['z_max_m']
        self._shape = None

    def resize(self, height: int, width: int) -> None:
        """Rebind every buffer to a new resolution. Never called per frame."""
        stride = self._stride
        rows = (height + stride - 1) // stride
        cols = (width + stride - 1) // stride
        # ALGORITHM.md S1: the focal length in pixels tracks the vertical
        # resolution and the principal point is the centre of the image that
        # actually arrives. At the reference resolution this is the identity.
        self.fx = self.fy = self._f0 * height / self._ref_height
        self.cx = (width - 1) / 2
        self.cy = (height - 1) / 2

        self._u_minus_cx = np.arange(0, width, stride, dtype=np.float64) - self.cx
        self._v_minus_cy = np.arange(0, height, stride, dtype=np.float64) - self.cy
        self._z = np.empty((rows, cols), dtype=np.float64)
        self._lo = np.empty((rows, cols), dtype=np.bool_)
        self._hi = np.empty((rows, cols), dtype=np.bool_)
        self._scratch = np.empty(rows * cols, dtype=np.float64)
        self._work = np.empty(rows * cols, dtype=np.float64)
        self._points = np.empty((rows * cols, 3), dtype=np.float32)
        self._shape = (height, width)

    def run(self, depth: np.ndarray) -> np.ndarray:
        z = self._z
        lo = self._lo
        hi = self._hi
        stride = self._stride

        sampled = depth if stride == 1 else depth[::stride, ::stride]
        np.multiply(sampled, self._depth_scale, out=z)
        # A zero return is an invalid pixel and falls out of the same comparison
        # as an out-of-range one, so there is no separate validity test.
        np.greater_equal(z, self._z_min, out=lo)
        np.less_equal(z, self._z_max, out=hi)
        np.logical_and(lo, hi, out=lo)

        rows, cols = np.nonzero(lo)
        n = rows.size
        points = self._points[:n]
        held = self._scratch[:n]
        np.compress(lo.reshape(-1), z.reshape(-1), out=held)
        points[:, 2] = held

        # (u - cx) * z / fx, in the order ALGORITHM.md writes it. Folding the
        # division into a precomputed 1/fx would be one pass cheaper and would
        # round differently from the C++, which is not a trade worth making
        # against a 1e-6 equivalence gate.
        work = self._work[:n]
        np.take(self._u_minus_cx, cols, out=work)
        np.multiply(work, held, out=work)
        np.divide(work, self.fx, out=work)
        points[:, 0] = work
        np.take(self._v_minus_cy, rows, out=work)
        np.multiply(work, held, out=work)
        np.divide(work, self.fy, out=work)
        points[:, 1] = work
        return points


class Cropper:
    """S2: camera frame to base frame, then the workspace AABB."""

    __slots__ = ('_r_transposed', '_t', '_bounds', '_shape', '_cam', '_base',
                 '_keep', '_other', '_selected', '_points')

    def __init__(self, camera: dict, workspace: dict):
        transform = np.array(camera['T_base_cam'], dtype=np.float64).reshape(4, 4)
        self._r_transposed = np.ascontiguousarray(transform[:3, :3].T)
        self._t = transform[:3, 3].copy()
        self._bounds = np.array(
            [[workspace['x_min'], workspace['y_min'], workspace['z_min']],
             [workspace['x_max'], workspace['y_max'], workspace['z_max']]],
            dtype=np.float64)
        self._shape = None

    def resize(self, capacity: int) -> None:
        self._cam = np.empty((capacity, 3), dtype=np.float64)
        self._base = np.empty((capacity, 3), dtype=np.float64)
        self._keep = np.empty(capacity, dtype=np.bool_)
        self._other = np.empty(capacity, dtype=np.bool_)
        self._selected = np.empty((capacity, 3), dtype=np.float64)
        self._points = np.empty((capacity, 3), dtype=np.float32)
        self._shape = capacity

    def run(self, points_cam: np.ndarray) -> np.ndarray:
        n = points_cam.shape[0]
        cam = self._cam[:n]
        base = self._base[:n]
        keep = self._keep[:n]
        other = self._other[:n]

        # Widening float32 to float64 is exact, and it lets the rotation run as
        # one dgemm rather than nine broadcast passes. Rule 1 wants the
        # three-term dot accumulated in float64 in any case.
        np.copyto(cam, points_cam)
        np.matmul(cam, self._r_transposed, out=base)
        np.add(base, self._t, out=base)

        lower, upper = self._bounds
        np.greater_equal(base[:, 0], lower[0], out=keep)
        np.less_equal(base[:, 0], upper[0], out=other)
        np.logical_and(keep, other, out=keep)
        np.greater_equal(base[:, 1], lower[1], out=other)
        np.logical_and(keep, other, out=keep)
        np.less_equal(base[:, 1], upper[1], out=other)
        np.logical_and(keep, other, out=keep)
        np.greater_equal(base[:, 2], lower[2], out=other)
        np.logical_and(keep, other, out=keep)
        np.less_equal(base[:, 2], upper[2], out=other)
        np.logical_and(keep, other, out=keep)

        m = int(np.count_nonzero(keep))
        selected = self._selected[:m]
        np.compress(keep, base, axis=0, out=selected)
        points = self._points[:m]
        points[...] = selected
        return points
