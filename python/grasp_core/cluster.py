"""S4: voxel connected components.

The spec allows either a hash map with a BFS or a dense grid with a labeller,
as long as the index set comes out the same. On a grid this small the dense
form is not a close call: `scipy.ndimage.label` is a compiled union-find raster
scan, and a Python-level BFS over a dict would be tens of thousands of
interpreted iterations doing the same thing.

The grid is the cluster's own bounding box, not the workspace, so the labeller
walks a few thousand voxels instead of a quarter of a million. Both buffers are
flat and reshaped per frame, so a change of bounding box costs no allocation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import canonicalise_columns  # noqa: F401  (kept for symmetry)


class Clusterer:
    __slots__ = ('_voxel', '_min_points', '_structure', '_grid', '_labels',
                 '_keys', '_capacity', '_flat')

    def __init__(self, cluster: dict, capacity_voxels: int):
        self._voxel = cluster['voxel_size_m']
        self._min_points = cluster['min_points']
        # connectivity 6 in a 3-D grid is scipy's rank-1 structuring element:
        # face neighbours only, which is "keys differ by 1 in exactly one axis".
        connectivity = cluster['connectivity']
        rank = {6: 1, 18: 2, 26: 3}[connectivity]
        self._structure = ndimage.generate_binary_structure(3, rank)
        self._grid = np.empty(capacity_voxels, dtype=np.bool_)
        self._labels = np.empty(capacity_voxels, dtype=np.int32)
        self._capacity = 0

    def resize(self, capacity: int) -> None:
        self._capacity = capacity
        self._keys = np.empty((capacity, 3), dtype=np.int64)
        self._flat = np.empty(capacity, dtype=np.int64)

    def run(self, points: np.ndarray) -> np.ndarray:
        """Return the indices into `points` of the winning cluster."""
        n = points.shape[0]
        empty = self._flat[:0]
        if n < self._min_points:
            return empty

        keys = self._keys[:n]
        np.floor_divide(points, self._voxel, out=keys, casting='unsafe')
        low = keys.min(axis=0)
        np.subtract(keys, low, out=keys)
        dims = keys.max(axis=0) + 1
        size = int(dims.prod())
        if size > self._grid.size:
            # A cluster spanning more voxels than the workspace can hold means
            # the crop or the voxel size changed under us.
            raise ValueError(f'voxel grid of {size} exceeds the reserved space')

        grid = self._grid[:size]
        grid[:] = False
        flat = self._flat[:n]
        np.ravel_multi_index((keys[:, 0], keys[:, 1], keys[:, 2]), dims, out=flat)
        grid[flat] = True

        labels = self._labels[:size].reshape(dims)
        found = ndimage.label(grid.reshape(dims), self._structure, labels)
        if found == 0:
            return empty

        per_point = np.take(self._labels[:size], flat)
        counts = np.bincount(per_point, minlength=found + 1)
        counts[0] = 0
        best = int(np.argmax(counts))
        if counts[best] < self._min_points:
            return empty

        tied = np.nonzero(counts == counts[best])[0]
        if tied.size > 1:
            # C order over (kx, ky, kz) is lexicographic order over the keys,
            # and the offset subtracted above is a translation, so the smallest
            # flat index in a component is its lexicographically smallest key.
            first = np.full(found + 1, size, dtype=np.int64)
            np.minimum.at(first, per_point, flat)
            best = int(tied[np.argmin(first[tied])])

        return np.nonzero(per_point == best)[0]
