"""RANSAC on a plane whose answer is known in closed form.

The deviate table is the real one, so this also checks that the table is read
and indexed the way ALGORITHM.md S3 says: a mis-scaled index would still find a
plane here, but a mis-read table would not survive the tilted case.
"""
import numpy as np
import pytest


@pytest.fixture(scope='module')
def remover(config, deviates):
    from grasp_core.plane import PlaneRemover
    remover = PlaneRemover(config['plane'], deviates,
                           config['implementation']['ransac_block_points'])
    remover.resize(60000)
    return remover


def synthetic_scene(normal, offset, box_centre=(0.45, 0.0), box_half=0.03,
                    box_height=0.05, side=140, seed=4):
    """A tilted plane of `side * side` points with a box standing on it."""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    x = np.linspace(0.1, 0.7, side)
    y = np.linspace(-0.3, 0.3, side)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
    plane_z = (-offset - normal[0] * grid_x - normal[1] * grid_y) / normal[2]
    table = np.stack([grid_x, grid_y, plane_z], axis=1)
    table[:, 2] += rng.normal(scale=0.001, size=table.shape[0])

    inside = ((np.abs(grid_x - box_centre[0]) <= box_half)
              & (np.abs(grid_y - box_centre[1]) <= box_half))
    points = table.copy()
    points[inside, 2] += box_height
    return points.astype(np.float32), inside


def test_finds_a_level_plane(remover):
    points, inside = synthetic_scene([0.0, 0.0, 1.0], 0.0)
    objects, found = remover.run(points)
    assert found
    assert np.abs(remover.plane[:3] - [0.0, 0.0, 1.0]).max() < 1e-3
    assert abs(remover.plane[3]) < 1e-3
    assert objects.shape[0] == pytest.approx(int(inside.sum()), rel=0.02)


def test_finds_a_tilted_plane_and_keeps_only_what_stands_on_it(remover, config):
    normal = np.array([0.03, -0.02, 1.0])
    normal /= np.linalg.norm(normal)
    offset = -0.004
    points, inside = synthetic_scene(normal, offset)
    objects, found = remover.run(points)
    assert found
    assert np.abs(remover.plane[:3] - normal).max() < 1e-3
    assert remover.plane[3] == pytest.approx(offset, abs=1e-3)
    assert remover.plane[2] > 0.0, 'the normal must be oriented upward'

    # Everything kept must stand clear of the plane by the removal margin.
    margin = config['plane']['inlier_threshold_m'] + config['plane']['clearance_m']
    signed = objects @ remover.plane[:3] + remover.plane[3]
    assert signed.min() >= margin
    assert objects.shape[0] == pytest.approx(int(inside.sum()), rel=0.02)


def test_reports_no_plane_when_there_is_none(remover, config):
    """A sphere of points has no dominant plane with min_inliers support."""
    rng = np.random.default_rng(1)
    points = rng.normal(size=(config['plane']['min_inliers'] * 3, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    objects, found = remover.run(points.astype(np.float32) * 0.3)
    assert not found
    assert objects.shape[0] == points.shape[0], 'points pass through unchanged'


def test_too_few_points_passes_through(remover):
    points = np.zeros((2, 3), dtype=np.float32)
    objects, found = remover.run(points)
    assert not found
    assert objects.shape[0] == 2


def test_is_deterministic(remover):
    points, _ = synthetic_scene([0.01, 0.01, 1.0], -0.002)
    first, _ = remover.run(points)
    first = first.copy()
    plane = remover.plane.copy()
    second, _ = remover.run(points)
    assert np.array_equal(first, second)
    assert np.array_equal(plane, remover.plane)
