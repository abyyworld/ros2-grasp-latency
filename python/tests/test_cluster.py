"""Connected components on blobs whose membership is known by construction.

The interesting cases are the ones where the answer is decided by a rule rather
than by the data: most points rather than most voxels, and the lexicographic
tie-break. Both are places where a hash-map implementation and a dense-grid one
can silently disagree.
"""
import numpy as np
import pytest


@pytest.fixture(scope='module')
def clusterer(config):
    from grasp_core.cluster import Clusterer
    voxel = config['cluster']['voxel_size_m']
    workspace = config['workspace']
    span = 1
    for axis in ('x', 'y', 'z'):
        span *= int(np.ceil((workspace[f'{axis}_max']
                             - workspace[f'{axis}_min']) / voxel)) + 2
    clusterer = Clusterer(config['cluster'], span)
    clusterer.resize(20000)
    return clusterer


def blob(centre, count, radius, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(centre) + rng.uniform(-radius, radius, size=(count, 3))


def test_picks_the_blob_with_the_most_points(clusterer, config):
    small = blob((0.30, -0.10, 0.05), config['cluster']['min_points'] * 2,
                 0.015, 1)
    large = blob((0.55, 0.12, 0.05), config['cluster']['min_points'] * 5,
                 0.015, 2)
    points = np.vstack([small, large])
    indices = clusterer.run(points)
    assert indices.size == large.shape[0]
    assert np.array_equal(np.sort(indices),
                          np.arange(small.shape[0], points.shape[0]))


def test_most_points_beats_most_voxels(clusterer, config):
    """A dense clump of few voxels must beat a sparse shell of many."""
    voxel = config['cluster']['voxel_size_m']
    minimum = config['cluster']['min_points']
    rng = np.random.default_rng(3)
    dense = np.array([0.30, 0.0, 0.05]) + rng.uniform(
        0.0, voxel * 0.9, size=(minimum * 4, 3))
    # A connected line of voxels, one point each: many voxels, few points.
    sparse = np.stack([
        np.array([0.55, 0.0, 0.05]) + np.array([i * voxel, 0.0, 0.0])
        for i in range(minimum * 2)])
    points = np.vstack([dense, sparse])
    indices = clusterer.run(points)
    assert indices.size == dense.shape[0]
    assert indices.max() < dense.shape[0]


def test_tie_breaks_toward_the_lowest_key(clusterer, config):
    """Two components with identical point counts: the one holding the
    lexicographically smallest (kx, ky, kz) wins."""
    voxel = config['cluster']['voxel_size_m']
    minimum = config['cluster']['min_points']
    rng = np.random.default_rng(5)
    offsets = rng.uniform(0.0, voxel * 0.5, size=(minimum * 2, 3))
    low = np.array([0.20, -0.20, 0.05]) + offsets
    high = np.array([0.60, 0.20, 0.05]) + offsets
    # Whichever order the points arrive in, the same component must win.
    for order in (np.vstack([low, high]), np.vstack([high, low])):
        indices = clusterer.run(order)
        assert indices.size == minimum * 2
        centre = order[indices].mean(axis=0)
        assert centre[0] < 0.4 and centre[1] < 0.0


def test_returns_empty_below_the_minimum(clusterer, config):
    points = blob((0.4, 0.0, 0.05), config['cluster']['min_points'] - 1,
                  0.01, 7)
    assert clusterer.run(points).size == 0


def test_six_connectivity_does_not_join_diagonal_voxels(clusterer, config):
    """Face neighbours only: two clumps touching at a corner stay apart."""
    voxel = config['cluster']['voxel_size_m']
    minimum = config['cluster']['min_points']
    rng = np.random.default_rng(9)
    base = np.array([0.40, 0.0, 0.05])
    first = base + rng.uniform(0.0, voxel * 0.9, size=(minimum * 3, 3))
    diagonal = base + np.array([voxel, voxel, 0.0]) \
        + rng.uniform(0.0, voxel * 0.9, size=(minimum * 2, 3))
    indices = clusterer.run(np.vstack([first, diagonal]))
    assert indices.size == first.shape[0]
