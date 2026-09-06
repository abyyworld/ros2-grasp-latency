"""S4: voxel connected components.

The spec allows either a hash map with a BFS or a dense grid with a labeller,
as long as the index set comes out the same. On a grid this small it is not a
close call: `scipy.ndimage.label` is a compiled union-find raster scan, and a
Python-level BFS over a dict would be tens of thousands of interpreted
iterations doing the same thing.

The grid is the object cloud's own bounding box rather than the whole
workspace, so the labeller walks a few thousand voxels instead of a quarter of
a million. Both grids are flat buffers reshaped per frame, so a change of
bounding box costs no allocation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# scipy's structuring element rank for each connectivity the config can ask
# for: rank 1 is face neighbours only, which is the spec's "keys differ by 1 in
# exactly one axis".
_RANK_FOR_CONNECTIVITY = {6: 1, 18: 2, 26: 3}


class Clusterer:
    __slots__ = ('_voxel', '_min_points', '_structure', '_grid', '_labels',
                 '_keys', '_scaled', '_flat', '_capacity')

    def __init__(self, cluster: dict, capacity_voxels: int):
        self._voxel = cluster['voxel_size_m']
        self._min_points = cluster['min_points']
        self._structure = ndimage.generate_binary_structure(
            3, _RANK_FOR_CONNECTIVITY[cluster['connectivity']])
        self._grid = np.empty(capacity_voxels, dtype=np.bool_)
        self._labels = np.empty(capacity_voxels, dtype=np.int32)
        self._capacity = 0

    def resize(self, capacity: int) -> None:
        self._capacity = capacity
        self._keys = np.empty((capacity, 3), dtype=np.int64)
        self._scaled = np.empty((capacity, 3), dtype=np.float64)
        self._flat = np.empty(capacity, dtype=np.int64)

    def run(self, points: np.ndarray) -> np.ndarray:
        """Return the indices into `points` of the winning cluster."""
        n = points.shape[0]
        if n < self._min_points:
            return self._flat[:0]

        # floor(x / v) spelled out, rather than floor_divide: numpy's
        # floor_divide carries a remainder correction that C++ floor(x / v)
        # does not, and the two disagree on values that sit on a voxel face.
        scaled = self._scaled[:n]
        keys = self._keys[:n]
        np.divide(points, self._voxel, out=scaled)
        np.floor(scaled, out=scaled)
        np.copyto(keys, scaled, casting='unsafe')

        low = keys.min(axis=0)
        np.subtract(keys, low, out=keys)
        dims = keys.max(axis=0) + 1
        size = int(dims[0] * dims[1] * dims[2])
        if size > self._grid.size:
            raise ValueError(f'voxel grid of {size} exceeds the reserved space')

        flat = self._flat[:n]
        np.multiply(keys[:, 0], dims[1], out=flat)
        np.add(flat, keys[:, 1], out=flat)
        np.multiply(flat, dims[2], out=flat)
        np.add(flat, keys[:, 2], out=flat)

        grid = self._grid[:size]
        grid[:] = False
        grid[flat] = True
        labels = self._labels[:size]
        found = ndimage.label(grid.reshape(dims), self._structure,
                              labels.reshape(dims))
        if found == 0:
            return self._flat[:0]

        per_point = np.take(labels, flat)
        counts = np.bincount(per_point, minlength=found + 1)
        counts[0] = 0
        best = int(np.argmax(counts))
        if counts[best] < self._min_points:
            return self._flat[:0]

        tied = np.nonzero(counts == counts[best])[0]
        if tied.size > 1:
            # C order over (kx, ky, kz) is lexicographic order over the keys,
            # and the offset subtracted above is a translation, so the smallest
            # flat index in a component holds its smallest key.
            first = np.full(found + 1, size, dtype=np.int64)
            np.minimum.at(first, per_point, flat)
            best = int(tied[np.argmin(first[tied])])

        return np.nonzero(per_point == best)[0]
