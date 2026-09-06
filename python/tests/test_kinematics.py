"""FK against an independent implementation, J against finite differences, and
IK checked on the pose it was asked for rather than on the joint angles.

The round trip is deliberately not `q_in == q_out`. The Panda has a null space,
so a seven-joint solution is not unique and demanding the seed back would be
testing the wrong thing. What has to hold is that FK of the answer is the pose
that was asked for, to the tolerance the solver claims.
"""
import json
import math

import numpy as np
import pytest

from conftest import CHAIN_PATH  # noqa: E402

RAW = json.loads(CHAIN_PATH.read_text())


def axis_rotation(axis, angle):
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def reference_fk(q):
    """The textbook loop, written out. Slow, obvious, and independent."""
    pose = np.eye(4)
    for joint, angle in zip(RAW['joints'], q):
        pose = pose @ np.array(joint['fixed']).reshape(4, 4)
        rotation = np.eye(4)
        rotation[:3, :3] = axis_rotation(joint['axis'], float(angle))
        pose = pose @ rotation
    return pose @ np.array(RAW['flange_to_tcp']).reshape(4, 4)


@pytest.fixture(scope='module')
def solver(chain, config):
    from grasp_core.kinematics import IKSolver
    return IKSolver(chain, config['ik'])


def sample_configurations(chain, count, seed):
    rng = np.random.default_rng(seed)
    span = chain.upper - chain.lower
    return [chain.lower + span * rng.random(chain.dof) for _ in range(count)]


def test_forward_kinematics_match_the_reference(solver, chain):
    for q in [chain.q_neutral] + sample_configurations(chain, 16, 20240906):
        assert np.abs(solver.forward(q) - reference_fk(q)).max() < 1e-12


def test_jacobian_matches_central_differences(solver, chain):
    """The Jacobian is written out per component, so it gets checked per
    component, against a derivative that shares no code with it."""
    step = 1e-6
    for q in sample_configurations(chain, 4, 7):
        pose = solver.forward(q).copy()
        axes = solver._axes
        origins = solver._frame[:, :3, 3]
        analytic = np.empty((6, chain.dof))
        analytic[3:] = axes
        offsets = pose[:3, 3] - origins
        for i in range(chain.dof):
            analytic[:3, i] = np.cross(axes[:, i], offsets[i])

        numeric = np.zeros((6, chain.dof))
        for i in range(chain.dof):
            forward, backward = q.copy(), q.copy()
            forward[i] += step
            backward[i] -= step
            plus, minus = reference_fk(forward), reference_fk(backward)
            numeric[:3, i] = (plus[:3, 3] - minus[:3, 3]) / (2.0 * step)
            spin = ((plus[:3, :3] - minus[:3, :3]) / (2.0 * step)) @ pose[:3, :3].T
            numeric[3:, i] = [spin[2, 1], spin[0, 2], spin[1, 0]]
        assert np.abs(analytic - numeric).max() < 1e-7


def test_ik_round_trip_on_the_pose_not_the_angles(solver, chain, config):
    """Every pose FK can produce near the neutral posture must come back."""
    rng = np.random.default_rng(11)
    tol_p = config['ik']['position_tolerance_m']
    tol_r = config['ik']['orientation_tolerance_rad']
    solved = 0
    for _ in range(24):
        q = np.clip(chain.q_neutral + rng.normal(scale=0.25, size=chain.dof),
                    chain.lower, chain.upper)
        target = reference_fk(q).copy()
        if not solver.solve(target):
            continue
        solved += 1
        reached = reference_fk(solver.q)
        assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < tol_p
        error = 0.5 * np.cross(reached[:3, :3].T, target[:3, :3].T).sum(axis=0)
        assert np.linalg.norm(error) < tol_r
    assert solved > 20, f'only {solved} of 24 nearby poses converged'


def test_ik_respects_joint_limits(solver, chain):
    rng = np.random.default_rng(2)
    for _ in range(8):
        q = np.clip(chain.q_neutral + rng.normal(scale=0.4, size=chain.dof),
                    chain.lower, chain.upper)
        solver.solve(reference_fk(q).copy())
        assert np.all(solver.q >= chain.lower - 1e-12)
        assert np.all(solver.q <= chain.upper + 1e-12)


def test_ik_is_seeded_from_neutral_every_call(solver, chain):
    """Two calls with the same target give the same answer, whatever ran in
    between. Per-frame cost must not depend on tracking history."""
    first_target = reference_fk(chain.q_neutral + 0.2).copy()
    second_target = reference_fk(chain.q_neutral - 0.3).copy()
    solver.solve(first_target)
    once = solver.q.copy()
    solver.solve(second_target)
    solver.solve(first_target)
    assert np.array_equal(once, solver.q)
