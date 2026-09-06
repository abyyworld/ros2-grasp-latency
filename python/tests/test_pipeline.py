"""End to end on a frame whose geometry is known, plus the properties the
rclpy node and the benchmark depend on.

The scene is rendered from the camera model in the config rather than read from
`data/`, so this runs in a checkout that has never generated a corpus. The
corpus tests live in `tests/test_dataset.py`.
"""
import numpy as np
import pytest


def run_scene(pipeline, scene):
    return pipeline.run(scene['depth'], scene['rgb'],
                        scene['width'], scene['height'])


def test_finds_the_box_that_was_rendered(pipeline, top_down_scene, config):
    scene = top_down_scene
    result = run_scene(pipeline, scene)

    assert result.plane_found
    assert np.abs(result.plane[:3] - [0.0, 0.0, 1.0]).max() < 1e-6
    assert abs(result.plane[3]) < 1e-6
    assert result.graspable

    # The fingers close along the narrow side, so the width is the short
    # extent plus the clearance.
    expected = 2.0 * min(scene['half']) + config['grasp']['finger_clearance_m']
    assert result.width == pytest.approx(expected, abs=0.01)

    assert result.tcp[0, 3] == pytest.approx(scene['centre'][0], abs=0.01)
    assert result.tcp[1, 3] == pytest.approx(scene['centre'][1], abs=0.01)
    # Grasp depth below the box top, floored by the plane clearance.
    assert result.tcp[2, 3] == pytest.approx(
        scene['height_m'] - config['grasp']['grasp_depth_m'], abs=0.005)
    assert np.abs(result.tcp[:3, 2] - config['grasp']['approach_axis_base']).max() == 0.0
    assert np.linalg.det(result.tcp[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_closing_axis_is_the_narrow_one(pipeline, config):
    from conftest import _render_box_on_table
    scene = _render_box_on_table(config, half=(0.015, 0.04))
    result = run_scene(pipeline, scene)
    # half[0] is the short side and lies along base x, so the fingers close
    # along x and the TCP y axis points that way.
    assert abs(result.tcp[0, 1]) > abs(result.tcp[1, 1])


def test_ik_answer_reaches_the_grasp_pose(pipeline, top_down_scene, config):
    result = run_scene(pipeline, top_down_scene)
    if not result.converged:
        pytest.skip('this pose did not converge; nothing to check the pose of')
    reached = pipeline._ik.forward(result.q)
    target = result.tcp
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) \
        < config['ik']['position_tolerance_m']
    error = 0.5 * np.cross(reached[:3, :3].T, target[:3, :3].T).sum(axis=0)
    assert np.linalg.norm(error) < config['ik']['orientation_tolerance_rad']


def test_bytes_and_arrays_give_the_same_answer(pipeline, top_down_scene):
    """The node hands over NumPy views, the benchmark hands over bytes. S0 is
    the only stage that can tell, and it must not change the answer."""
    scene = top_down_scene
    first = run_scene(pipeline, scene)
    reference = (first.q.copy(), first.tcp.copy(), first.plane.copy())

    depth = np.frombuffer(scene['depth'], dtype='<u2').reshape(
        scene['height'], scene['width'])
    colour = np.frombuffer(scene['rgb'], dtype=np.uint8).reshape(
        scene['height'], scene['width'], 3)
    second = pipeline.run(depth, colour)
    assert np.array_equal(second.q, reference[0])
    assert np.array_equal(second.tcp, reference[1])
    assert np.array_equal(second.plane, reference[2])


def test_repeated_runs_are_bit_identical(pipeline, top_down_scene):
    first = run_scene(pipeline, top_down_scene)
    snapshot = (first.q.copy(), first.tcp.copy(), first.plane.copy(),
                first.width, first.iterations, first.duration_s,
                first.positions.copy())
    for _ in range(3):
        again = run_scene(pipeline, top_down_scene)
        assert np.array_equal(again.q, snapshot[0])
        assert np.array_equal(again.tcp, snapshot[1])
        assert np.array_equal(again.plane, snapshot[2])
        assert again.width == snapshot[3]
        assert again.iterations == snapshot[4]
        assert again.duration_s == snapshot[5]
        assert np.array_equal(again.positions, snapshot[6])


def test_result_carries_every_stage_the_wire_format_names(pipeline,
                                                          top_down_scene):
    from grasp_core import STAGE_KEYS
    result = run_scene(pipeline, top_down_scene)
    assert tuple(result.stage_ns) == STAGE_KEYS
    assert all(value > 0 for value in result.stage_ns.values())
    assert result.total_ns == sum(result.stage_ns.values()) \
        or result.total_ns >= sum(result.stage_ns.values())


def test_trajectory_is_the_shape_the_node_expects(pipeline, top_down_scene,
                                                  config):
    result = run_scene(pipeline, top_down_scene)
    assert len(result.trajectory) == config['trajectory']['waypoints']
    for waypoint in result.trajectory:
        assert waypoint.position.shape == (pipeline.chain.dof,)
        assert waypoint.velocity.shape == (pipeline.chain.dof,)
        assert waypoint.acceleration.shape == (pipeline.chain.dof,)
        assert isinstance(waypoint.time_from_start, float)
    assert result.trajectory[0].time_from_start == 0.0
    assert result.trajectory[-1].time_from_start == pytest.approx(
        result.duration_s)


def test_a_second_resolution_rebinds_the_buffers(pipeline, config):
    """Buffers are bound to a resolution on the first frame that uses it. A
    stream that changes size must rebind rather than read the old shape, and
    the intrinsics must rescale with the width."""
    from conftest import _render_box_on_table
    camera = config['camera']
    first = _render_box_on_table(config)
    assert run_scene(pipeline, first).graspable

    wide = dict(first)
    scale = 2
    doubled = _render_box_on_table(config)
    # Re-render at twice the width by asking the helper for the reference size.
    doubled['width'] = first['width'] * scale
    doubled['height'] = first['height'] * scale
    grid = np.frombuffer(first['depth'], dtype='<u2').reshape(
        first['height'], first['width'])
    doubled['depth'] = np.repeat(np.repeat(grid, scale, axis=0), scale,
                                 axis=1).tobytes()
    doubled['rgb'] = bytes(doubled['width'] * doubled['height'] * 3)

    result = run_scene(pipeline, doubled)
    assert result.graspable
    assert result.tcp[0, 3] == pytest.approx(first['centre'][0], abs=0.02)
    assert pipeline._deprojector.fx == pytest.approx(
        camera['fx'] * doubled['width'] / camera['reference_width'])

    # And back again, so the rebinding is not one way.
    assert run_scene(pipeline, wide).graspable
