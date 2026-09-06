"""The quintic has to be a quintic: right endpoints, zero derivatives there,
and a duration that respects the joint velocity limits and the clamps.
"""
import numpy as np
import pytest


@pytest.fixture(scope='module')
def planner(config, chain):
    from grasp_core.trajectory import TrajectoryPlanner
    return TrajectoryPlanner(config['trajectory'], chain)


def test_endpoints_are_exact(planner, chain, config):
    goal = chain.q_neutral + np.linspace(0.1, 0.7, chain.dof)
    planner.run(chain.q_neutral, goal)
    assert np.allclose(planner.positions[0], chain.q_neutral, atol=0.0, rtol=0.0)
    assert np.allclose(planner.positions[-1], goal, rtol=0.0, atol=1e-15)
    for profile in (planner.velocities, planner.accelerations):
        assert np.all(profile[0] == 0.0)
        assert np.all(profile[-1] == 0.0)
    assert planner.times[0] == 0.0
    assert planner.times[-1] == pytest.approx(planner.duration, rel=0.0, abs=1e-15)


def test_profiles_match_the_closed_form(planner, chain, config):
    """The shape functions are built as coefficient times power so that they
    are provably each other's derivative. Check them against the polynomial
    ALGORITHM.md S7 writes out, term by term."""
    goal = chain.q_neutral + np.linspace(-0.5, 0.5, chain.dof)
    duration = planner.run(chain.q_neutral, goal)
    delta = goal - chain.q_neutral
    count = planner.positions.shape[0]
    for k in range(count):
        s = k / (count - 1)
        h = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        hd = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration
        hdd = (60 * s - 180 * s ** 2 + 120 * s ** 3) / duration ** 2
        # Relative, because the planner sums the polynomial as coefficient
        # times power while the line above sums it as written, and numpy
        # evaluates an integral power by repeated multiplication where Python
        # calls pow. Both are the same polynomial to 1e-13 relative, which the
        # nine-decimal rounding in the trajectory checksum absorbs.
        assert np.allclose(planner.positions[k], chain.q_neutral + h * delta,
                           rtol=1e-13, atol=1e-15)
        assert np.allclose(planner.velocities[k], hd * delta,
                           rtol=1e-13, atol=1e-15)
        assert np.allclose(planner.accelerations[k], hdd * delta,
                           rtol=1e-13, atol=1e-14)
        assert planner.times[k] == pytest.approx(s * duration, abs=1e-15)


def test_velocity_integrates_to_the_displacement(planner, chain):
    """A coarse but independent check that the scaling by 1/duration is right:
    the trapezoid rule over 20 waypoints recovers the travel to within 1%."""
    goal = chain.q_neutral + np.linspace(-0.5, 0.5, chain.dof)
    planner.run(chain.q_neutral, goal)
    travelled = np.trapezoid(planner.velocities, planner.times, axis=0)
    assert np.abs(travelled - (goal - chain.q_neutral)).max() < 0.01


def test_duration_follows_the_slowest_joint(planner, chain, config):
    trajectory = config['trajectory']
    limits = chain.velocity * trajectory['velocity_fraction']
    goal = chain.q_neutral.copy()
    goal[3] += 1.2
    duration = planner.run(chain.q_neutral, goal)
    assert duration == pytest.approx(1.2 / limits[3])
    assert trajectory['min_duration_s'] <= duration <= trajectory['max_duration_s']


def test_duration_is_clamped_both_ways(planner, chain, config):
    trajectory = config['trajectory']
    assert planner.run(chain.q_neutral, chain.q_neutral) \
        == pytest.approx(trajectory['min_duration_s'])
    far = np.clip(chain.q_neutral + 6.0, chain.lower, chain.upper)
    assert planner.run(chain.q_neutral, far) \
        <= trajectory['max_duration_s'] + 1e-15


def test_waypoints_view_the_planner_buffers(planner, chain):
    """The ROS node writes straight out of these, so they must be views."""
    planner.run(chain.q_neutral, chain.q_neutral + 0.3)
    for k, waypoint in enumerate(planner.waypoints):
        assert waypoint.position.base is not None
        assert np.shares_memory(waypoint.position, planner.positions)
        assert waypoint.time_from_start == planner.times[k]
