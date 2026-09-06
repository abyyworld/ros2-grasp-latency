"""The URDF and the MJCF are supposed to be the same robot. Check, don't assume.

This repository claims both formats, and the claim is only worth something if
the two files actually agree. So: forward kinematics from the flattened URDF
chain, compared against MuJoCo's own kinematics on the vendored MJCF, over
random configurations drawn from the joint limits.

They agree to 6 picometres in position. They do *not* agree exactly in
orientation, and the reason is worth recording: Menagerie writes each body's
quaternion to seven decimal places, so the hand's 45-degree offset is stored as
0.785398128478 rad against the URDF's 0.785398163397. Those truncations
accumulate down the chain to 8.4e-8 rad at the hand -- about 84 nanometres of
tip displacement at full reach. That is a property of the vendored files, not
of the solver, so the tolerance below is set to admit it and nothing larger.
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = json.loads((ROOT / "assets/franka/panda_chain.json").read_text())
MJCF = ROOT / "assets/franka/mjcf/panda.xml"
HAND_TO_TCP_Z = 0.1034

N_SAMPLES = 500
POSITION_TOL_M = 1e-9        # measured worst case: 6.1e-12
ORIENTATION_TOL_RAD = 5e-7   # measured worst case: 8.4e-8, from quat truncation


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation -- the same formulation both pipelines use."""
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)


def urdf_fk(q: np.ndarray, upto: int) -> np.ndarray:
    t = np.eye(4)
    for joint, angle in list(zip(CHAIN["joints"], q))[:upto]:
        t = t @ np.array(joint["fixed"]).reshape(4, 4)
        rot = np.eye(4)
        rot[:3, :3] = axis_rotation(np.array(joint["axis"]), float(angle))
        t = t @ rot
    return t


def urdf_hand(q: np.ndarray) -> np.ndarray:
    """Base -> panda_hand. flange_to_tcp includes the TCP offset, which MuJoCo's
    `hand` body does not have, so it is backed out for a like-for-like frame."""
    t = urdf_fk(q, 7) @ np.array(CHAIN["flange_to_tcp"]).reshape(4, 4)
    back = np.eye(4)
    back[2, 3] = -HAND_TO_TCP_Z
    return t @ back


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.acos(float(np.clip(cos, -1.0, 1.0)))


def joint_ranges() -> np.ndarray:
    return np.array([[j["lower"], j["upper"]] for j in CHAIN["joints"]])


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    return model, mujoco.MjData(model)


@pytest.fixture(scope="module")
def sampled_configurations():
    lo, hi = joint_ranges().T
    rng = np.random.default_rng(20240906)
    return rng.uniform(lo, hi, size=(N_SAMPLES, 7))


@pytest.mark.parametrize("body,upto", [(f"link{i}", i) for i in range(1, 8)] + [("hand", 7)])
def test_forward_kinematics_agree(mj, sampled_configurations, body, upto):
    """Every frame in the arm chain must match MuJoCo's, not just the tip."""
    model, data = mj
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, f"vendored MJCF has no `{body}` body"

    worst_pos = worst_rot = 0.0
    for q in sampled_configurations:
        data.qpos[:7] = q
        data.qpos[7:] = 0.0
        mujoco.mj_kinematics(model, data)

        ours = urdf_hand(q) if body == "hand" else urdf_fk(q, upto)
        worst_pos = max(worst_pos, float(np.abs(ours[:3, 3] - data.xpos[bid]).max()))
        worst_rot = max(worst_rot, angle_between(np.array(data.xmat[bid]).reshape(3, 3),
                                                 ours[:3, :3]))

    assert worst_pos < POSITION_TOL_M, f"{body}: position off by {worst_pos:.3e} m"
    assert worst_rot < ORIENTATION_TOL_RAD, f"{body}: orientation off by {worst_rot:.3e} rad"


def test_orientation_error_is_upstream_quaternion_truncation():
    """Pin the *cause*, so a real kinematic regression can't hide behind it.

    If the residual were a modelling error it would not be reproducible from
    the vendored quaternion's own rounding.
    """
    stored = np.array([0.9238795, 0.0, 0.0, -0.3826834])   # Menagerie `hand` quat
    exact = np.array([math.cos(math.pi / 8), 0.0, 0.0, -math.sin(math.pi / 8)])

    def half_angle(quat):
        return 2.0 * math.atan2(float(np.linalg.norm(quat[1:])), float(quat[0]))

    truncation = abs(half_angle(stored) - half_angle(exact))
    assert 3e-8 < truncation < 4e-8, f"unexpected quat truncation {truncation:.3e}"
    # The whole-chain residual is the same order, i.e. explained by this effect.
    assert truncation < ORIENTATION_TOL_RAD


def test_joint_limits_differ_as_documented(mj):
    """The two files do NOT agree on joint limits. Pin the difference.

    moveit_resources carries the Panda's wider physical limits; Menagerie
    carries Franka's tighter recommended limits. The pipeline solves against the
    URDF's, so an asset bump that silently moves either set should fail here.
    """
    model, _ = mj
    urdf, mjcf = joint_ranges(), np.array(model.jnt_range[:7])

    assert not np.allclose(urdf, mjcf), "expected the documented limit mismatch"
    assert np.all(mjcf[:, 0] >= urdf[:, 0] - 1e-12), "MJCF lower limit wider than URDF"
    assert np.all(mjcf[:, 1] <= urdf[:, 1] + 1e-12), "MJCF upper limit wider than URDF"
