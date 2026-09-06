"""Construction-time loading of every constant the pipeline uses.

Determinism rule 5 forbids config parsing inside the measured path, and a dict
lookup is a hash and a pointer chase that no stage should be paying per frame.
So everything is read once here, unpacked into plain attributes on the stage
objects, and hoisted into locals by each `run`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEPTH_DTYPE = np.dtype('<u2')
COLOUR_DTYPE = np.dtype(np.uint8)
DEVIATE_DTYPE = np.dtype('<f8')


def load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_deviates(path, expected_count: int) -> np.ndarray:
    """The RANSAC sample table, read whole (determinism rule 4).

    Its length is checked against the config rather than trusted: a truncated
    table would silently turn later iterations into repeats of index zero,
    which reads as a plausible result and is not one.
    """
    u = np.fromfile(str(path), dtype=DEVIATE_DTYPE)
    if u.size != expected_count:
        raise ValueError(
            f'{path} holds {u.size} deviates, config declares {expected_count}')
    if u.min() < 0.0 or u.max() >= 1.0:
        raise ValueError(f'{path} holds a deviate outside [0, 1)')
    return u


class Chain:
    """The flattened 7-DOF chain, as arrays rather than dicts of lists.

    `fixed` is the parent-to-joint-frame transform, `axis` the joint axis in
    that frame. Nothing assumes the axis is z even though every Panda arm joint
    happens to be, because the moment it is assumed the chain file stops being
    the source of truth.
    """

    __slots__ = ('dof', 'fixed', 'axis', 'lower', 'upper', 'velocity',
                 'q_neutral', 'flange_to_tcp', 'joint_names')

    def __init__(self, chain: dict):
        joints = chain['joints']
        self.dof = len(joints)
        self.joint_names = [j['name'] for j in joints]
        self.fixed = np.array([j['fixed'] for j in joints],
                              dtype=np.float64).reshape(self.dof, 4, 4)
        self.axis = np.array([j['axis'] for j in joints], dtype=np.float64)
        self.lower = np.array([j['lower'] for j in joints], dtype=np.float64)
        self.upper = np.array([j['upper'] for j in joints], dtype=np.float64)
        self.velocity = np.array([j['velocity'] for j in joints], dtype=np.float64)
        self.q_neutral = np.array(chain['q_neutral'], dtype=np.float64)
        self.flange_to_tcp = np.array(chain['flange_to_tcp'],
                                      dtype=np.float64).reshape(4, 4)
        if self.q_neutral.size != self.dof:
            raise ValueError('q_neutral does not match the joint count')


def canonicalise_columns(vectors: np.ndarray) -> np.ndarray:
    """Determinism rule 3, in place: fix the sign of each eigenvector.

    An eigensolver is free to return either sign, and the two implementations
    use different ones, so the sign has to be pinned by the spec instead of by
    LAPACK. Flip each column so that its largest-magnitude component is
    positive; `argmax` returns the lowest index, which is the tie-break the
    spec asks for.
    """
    lead = np.argmax(np.abs(vectors), axis=0)
    picked = vectors[lead, np.arange(vectors.shape[1])]
    vectors *= np.where(picked < 0.0, -1.0, 1.0)
    return vectors
