"""S3: RANSAC table removal.

This stage is the pipeline's arithmetic floor. Scoring `iterations` candidate
planes against `M` points is `3 * M * iterations` multiply-adds that no
implementation can avoid, so the only question is how the work is issued. The
scalar reading of the spec is a doubly nested loop; the array reading is one
matrix product per block of points, which is the same arithmetic handed to a
BLAS kernel instead of to the C compiler. Both are honest readings of the same
spec, and the fairness rule says Python is allowed to take the second one.

Points are blocked so that the `block x iterations` distance matrix stays in
cache: the whole point of the reformulation is that each point is read from
memory once and scored against every candidate while it is still hot.
"""
from __future__ import annotations

import numpy as np

from .config import canonicalise_columns

# ALGORITHM.md S3: a candidate whose edge cross product is shorter than this is
# three collinear points, not a plane. Not a tuning constant, so not in the
# config: any value in the neighbourhood picks out exactly the degenerate ones.
DEGENERATE_NORM = 1e-12

# A normal that scores nothing, used to keep the candidate matrix rectangular:
# an offset this large puts every point far outside the inlier band, so a
# rejected candidate contributes a column of zeroes instead of a reshape.
REJECTED_OFFSET = 1e30


class PlaneRemover:
    __slots__ = ('_deviates', '_iterations', '_threshold', '_min_cos',
                 '_min_inliers', '_removal_offset', '_block', '_capacity',
                 '_p64', '_dist', '_hit', '_counts', '_partial', '_signed',
                 '_mask', '_inliers', '_centred', '_normals', '_offsets',
                 '_normals_t', '_valid', 'plane')

    def __init__(self, plane: dict, deviates: np.ndarray, block_points: int):
        self._iterations = plane['iterations']
        self._threshold = plane['inlier_threshold_m']
        self._min_cos = plane['min_horizontal_cos']
        self._min_inliers = plane['min_inliers']
        self._removal_offset = plane['inlier_threshold_m'] + plane['clearance_m']
        self._block = block_points
        self._deviates = deviates.reshape(self._iterations, 3)
        self._capacity = 0

        k = self._iterations
        self._dist = np.empty((block_points, k), dtype=np.float64)
        self._hit = np.empty((block_points, k), dtype=np.bool_)
        self._counts = np.empty(k, dtype=np.int64)
        self._partial = np.empty(k, dtype=np.int64)
        self._normals = np.empty((k, 3), dtype=np.float64)
        self._normals_t = np.empty((3, k), dtype=np.float64)
        self._offsets = np.empty(k, dtype=np.float64)
        self._valid = np.empty(k, dtype=np.bool_)
        self.plane = np.zeros(4, dtype=np.float64)

    def resize(self, capacity: int) -> None:
        self._capacity = capacity
        self._p64 = np.empty((capacity, 3), dtype=np.float64)
        self._signed = np.empty(capacity, dtype=np.float64)
        self._mask = np.empty(capacity, dtype=np.bool_)
        self._inliers = np.empty((capacity, 3), dtype=np.float64)
        self._centred = np.empty((capacity, 3), dtype=np.float64)

    def run(self, points_base: np.ndarray):
        """Return (points_object as float64, plane_found).

        The returned array holds float32 values widened to float64. Widening is
        exact, so the values are the ones ALGORITHM.md specifies; carrying them
        as float64 saves every later stage a conversion it would otherwise
        repeat, and rule 1 makes every later reduction float64 regardless.
        """
        m = points_base.shape[0]
        points = self._p64[:m]
        np.copyto(points, points_base)
        if m < 3:
            self.plane[:] = 0.0
            return points, False

        index = np.minimum(m - 1, (self._deviates * m).astype(np.int64))
        i0 = index[:, 0]
        i1 = index[:, 1]
        i2 = index[:, 2]
        p0 = points[i0]
        edge1 = points[i1] - p0
        edge2 = points[i2] - p0

        normals = self._normals
        np.cross(edge1, edge2, out=normals)
        length = np.linalg.norm(normals, axis=1)

        valid = self._valid
        np.not_equal(i0, i1, out=valid)
        np.logical_and(valid, i1 != i2, out=valid)
        np.logical_and(valid, i0 != i2, out=valid)
        np.logical_and(valid, length >= DEGENERATE_NORM, out=valid)
        np.divide(normals, np.where(valid, length, 1.0)[:, None], out=normals)
        # Orient upward first, then reject what is still not close to level:
        # the spec's test is on the upward-oriented normal.
        np.multiply(normals, np.where(normals[:, 2] < 0.0, -1.0, 1.0)[:, None],
                    out=normals)
        np.logical_and(valid, normals[:, 2] >= self._min_cos, out=valid)

        offsets = self._offsets
        np.negative(np.einsum('ij,ij->i', normals, p0), out=offsets)
        np.copyto(normals, 0.0, where=~valid[:, None])
        np.copyto(offsets, REJECTED_OFFSET, where=~valid)
        np.copyto(self._normals_t, normals.T)

        counts = self._counts
        counts[:] = 0
        threshold = self._threshold
        dist = self._dist
        hit = self._hit
        partial = self._partial
        block = self._block
        normals_t = self._normals_t
        for start in range(0, m, block):
            stop = min(start + block, m)
            rows = stop - start
            chunk = dist[:rows]
            np.matmul(points[start:stop], normals_t, out=chunk)
            np.add(chunk, offsets, out=chunk)
            np.abs(chunk, out=chunk)
            np.less(chunk, threshold, out=hit[:rows])
            np.count_nonzero(hit[:rows], axis=0, out=partial)
            np.add(counts, partial, out=counts)

        # argmax returns the lowest index among equal maxima, which is the
        # spec's "strictly greatest score, so the earliest iteration wins".
        best = int(np.argmax(counts))
        if counts[best] < self._min_inliers:
            self.plane[:] = 0.0
            return points, False

        normal = normals[best]
        signed = self._signed[:m]
        mask = self._mask[:m]
        np.matmul(points, normal, out=signed)
        np.add(signed, offsets[best], out=signed)
        np.less(np.abs(signed, out=signed), threshold, out=mask)

        count = int(np.count_nonzero(mask))
        inliers = self._inliers[:count]
        np.compress(mask, points, axis=0, out=inliers)
        centroid = inliers.mean(axis=0)
        centred = self._centred[:count]
        np.subtract(inliers, centroid, out=centred)
        covariance = (centred.T @ centred) / count
        _, vectors = np.linalg.eigh(covariance)
        canonicalise_columns(vectors)
        normal = vectors[:, 0]
        if normal[2] < 0.0:
            normal = -normal
        offset = -float(normal @ centroid)

        np.matmul(points, normal, out=signed)
        np.add(signed, offset, out=signed)
        # Everything within the band, and everything below the plane, goes in
        # one comparison. What is left is what stands on the table.
        np.greater_equal(signed, self._removal_offset, out=mask)
        kept = int(np.count_nonzero(mask))
        objects = self._inliers[:kept]
        np.compress(mask, points, axis=0, out=objects)

        self.plane[:3] = normal
        self.plane[3] = offset
        return objects, True
