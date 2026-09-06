"""S3: RANSAC table removal.

This stage is the pipeline's arithmetic floor. Scoring `iterations` candidate
planes against `M` points is work no implementation can avoid; the only
question is how it is issued. The scalar reading of the spec is a doubly nested
loop, the array reading is one matrix product per block of points, and both are
honest readings of the same spec. The fairness rule says Python is allowed to
take the second one.

Two things make the array form worth the trouble:

* Points are blocked so the `block x candidates` distance matrix stays inside
  L2. Each point is read from memory once and scored against every candidate
  while it is still hot. Block size is `implementation.ransac_block_points`.
* Points are carried in homogeneous form, so `n . p + d` is one matrix product
  rather than a product and a broadcast add. Measured on the 640x480 corpus the
  broadcast add cost more than the extra column of arithmetic does.

Candidates the spec scores as zero (degenerate triples, normals that tilt too
far from vertical) are compacted out before the product rather than multiplied
by and thrown away, which is what "score = 0; continue" says to do.
"""
from __future__ import annotations

import numpy as np

from .config import canonicalise_columns

# ALGORITHM.md S3: an edge cross product shorter than this is three collinear
# points, not a plane. Not a tuning constant, so not in the config; any value
# in this neighbourhood picks out exactly the degenerate triples.
DEGENERATE_NORM = 1e-12


class PlaneRemover:
    __slots__ = ('_deviates', '_iterations', '_threshold', '_min_cos',
                 '_min_inliers', '_removal_offset', '_block', '_points',
                 '_dist_flat', '_hit_flat', '_plane_flat', '_counts',
                 '_partial', '_signed', '_mask', '_inliers', '_centred',
                 '_objects', '_normals', '_offsets', '_valid', 'plane')

    def __init__(self, plane: dict, deviates: np.ndarray, block_points: int):
        self._iterations = plane['iterations']
        self._threshold = plane['inlier_threshold_m']
        self._min_cos = plane['min_horizontal_cos']
        self._min_inliers = plane['min_inliers']
        self._removal_offset = plane['inlier_threshold_m'] + plane['clearance_m']
        self._block = block_points
        self._deviates = deviates.reshape(self._iterations, 3)

        k = self._iterations
        self._dist_flat = np.empty(block_points * k, dtype=np.float64)
        self._hit_flat = np.empty(block_points * k, dtype=np.bool_)
        self._plane_flat = np.empty(4 * k, dtype=np.float64)
        self._counts = np.empty(k, dtype=np.int64)
        self._partial = np.empty(k, dtype=np.int32)
        self._normals = np.empty((k, 3), dtype=np.float64)
        self._offsets = np.empty(k, dtype=np.float64)
        self._valid = np.empty(k, dtype=np.bool_)
        self.plane = np.zeros(4, dtype=np.float64)

    def resize(self, capacity: int) -> None:
        # Column 3 is the homogeneous one and is written once, here.
        self._points = np.empty((capacity, 4), dtype=np.float64)
        self._points[:, 3] = 1.0
        self._signed = np.empty(capacity, dtype=np.float64)
        self._mask = np.empty(capacity, dtype=np.bool_)
        self._inliers = np.empty((capacity, 3), dtype=np.float64)
        self._centred = np.empty((capacity, 3), dtype=np.float64)
        self._objects = np.empty((capacity, 3), dtype=np.float64)

    def run(self, points_base: np.ndarray):
        """Return (points_object, plane_found).

        `points_object` is float64 holding exactly the float32 values
        ALGORITHM.md specifies: widening is lossless, and carrying the wide copy
        saves S4 and S5 a conversion each while rule 1 makes their reductions
        float64 anyway.
        """
        m = points_base.shape[0]
        homogeneous = self._points[:m]
        np.copyto(homogeneous[:, :3], points_base)
        if m < 3:
            self.plane[:] = 0.0
            objects = self._objects[:m]
            np.copyto(objects, homogeneous[:, :3])
            return objects, False

        index = np.minimum(m - 1, (self._deviates * m).astype(np.int64))
        i0 = index[:, 0]
        i1 = index[:, 1]
        i2 = index[:, 2]
        p0 = homogeneous[i0, :3]
        normals = self._normals
        normals[...] = np.cross(homogeneous[i1, :3] - p0,
                                homogeneous[i2, :3] - p0)
        length = np.linalg.norm(normals, axis=1)

        valid = self._valid
        np.not_equal(i0, i1, out=valid)
        np.logical_and(valid, i1 != i2, out=valid)
        np.logical_and(valid, i0 != i2, out=valid)
        np.logical_and(valid, length >= DEGENERATE_NORM, out=valid)
        np.divide(normals, np.where(valid, length, 1.0)[:, None], out=normals)
        # Orient upward first: the spec tests the upward-oriented normal.
        np.multiply(normals, np.where(normals[:, 2] < 0.0, -1.0, 1.0)[:, None],
                    out=normals)
        np.logical_and(valid, normals[:, 2] >= self._min_cos, out=valid)
        np.negative(np.einsum('ij,ij->i', normals, p0), out=self._offsets)

        candidates = np.nonzero(valid)[0]
        wide = candidates.size
        if wide == 0:
            self.plane[:] = 0.0
            objects = self._objects[:m]
            np.copyto(objects, homogeneous[:, :3])
            return objects, False

        model = self._plane_flat[:4 * wide].reshape(4, wide)
        model[:3] = normals[candidates].T
        model[3] = self._offsets[candidates]

        counts = self._counts[:wide]
        counts[:] = 0
        partial = self._partial[:wide]
        dist_flat = self._dist_flat
        hit_flat = self._hit_flat
        threshold = self._threshold
        block = self._block
        for start in range(0, m, block):
            stop = start + block
            if stop > m:
                stop = m
            span = (stop - start) * wide
            chunk = dist_flat[:span].reshape(stop - start, wide)
            inside = hit_flat[:span].reshape(stop - start, wide)
            np.matmul(homogeneous[start:stop], model, out=chunk)
            np.abs(chunk, out=chunk)
            np.less(chunk, threshold, out=inside)
            # A bool view summed in a narrow integer type is markedly cheaper
            # than the int64 reduction numpy picks by default. int32, not
            # int16: the sum reaches the block's row count, so int16 wraps
            # silently for any ransac_block_points above 32767 and RANSAC then
            # picks a different candidate. That is not hypothetical, it was
            # caught by the equivalence gate at a 131072-point block.
            np.add.reduce(inside.view(np.uint8), axis=0, dtype=np.int32,
                          out=partial)
            np.add(counts, partial, out=counts)

        # argmax returns the lowest index among equal maxima, and compaction
        # preserved candidate order, so this is the spec's "strictly greatest
        # score, earliest iteration wins a tie".
        winner = int(np.argmax(counts))
        if counts[winner] < self._min_inliers:
            self.plane[:] = 0.0
            objects = self._objects[:m]
            np.copyto(objects, homogeneous[:, :3])
            return objects, False

        signed = self._signed[:m]
        mask = self._mask[:m]
        np.matmul(homogeneous, model[:, winner], out=signed)
        np.less(np.abs(signed, out=signed), threshold, out=mask)

        count = int(np.count_nonzero(mask))
        # Gather the inliers into their own three columns rather than carrying
        # the homogeneous one through the covariance: the fourth column is a
        # third more traffic in every pass that follows, for a constant 1.
        inliers = self._inliers[:count]
        np.compress(mask, homogeneous[:, :3], axis=0, out=inliers)
        centroid = inliers.mean(axis=0)
        centred = self._centred[:count]
        np.subtract(inliers, centroid, out=centred)
        _, vectors = np.linalg.eigh((centred.T @ centred) / count)
        canonicalise_columns(vectors)
        normal = vectors[:, 0]
        if normal[2] < 0.0:
            normal = -normal
        offset = -float(normal @ centroid)

        refit = self.plane
        refit[:3] = normal
        refit[3] = offset
        np.matmul(homogeneous, refit, out=signed)
        # The band and everything below it go in one comparison. What survives
        # is what stands on the table.
        np.greater_equal(signed, self._removal_offset, out=mask)
        kept = int(np.count_nonzero(mask))
        objects = self._objects[:kept]
        np.compress(mask, homogeneous[:, :3], axis=0, out=objects)
        return objects, True
