#!/usr/bin/env python3
"""Flatten the Panda URDF into a language-neutral kinematic chain.

Both the C++ and the Python pipeline load the JSON this emits, so neither one
carries its own copy of the robot's numbers and neither can drift from the
URDF. Parsing XML at benchmark time would also put an XML parser inside the
measured path, which is not what this project is measuring.

Each actuated joint is reduced to a constant parent->joint transform plus a
rotation axis, so forward kinematics in either language is the same loop:

    T = T_world_base
    for i in range(7):
        T = T @ fixed[i] @ axis_rotation(axis[i], q[i])
    T_tcp = T @ fixed_tip

Usage: extract_chain.py <panda.urdf> <out.json>
"""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Franka's published tool-centre point for the standard hand: 103.4 mm along
# +z from the `panda_hand` frame, midway between the fingertips. The URDF has
# no EE link, so the offset is stated here rather than silently baked into the
# solver.
HAND_TO_TCP_Z = 0.1034
GRIPPER_MAX_WIDTH = 0.08  # two fingers x 40 mm stroke


def rpy_to_matrix(r: float, p: float, y: float) -> list[list[float]]:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def homogeneous(xyz, rpy) -> list[float]:
    """Row-major 4x4 as a flat list of 16 -- the wire format for both languages."""
    rot = rpy_to_matrix(*rpy)
    return [rot[0][0], rot[0][1], rot[0][2], xyz[0],
            rot[1][0], rot[1][1], rot[1][2], xyz[1],
            rot[2][0], rot[2][1], rot[2][2], xyz[2],
            0.0, 0.0, 0.0, 1.0]


def matmul4(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * 16
    for i in range(4):
        for j in range(4):
            out[4 * i + j] = sum(a[4 * i + k] * b[4 * k + j] for k in range(4))
    return out


def floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def origin_of(joint: ET.Element) -> tuple[list[float], list[float]]:
    node = joint.find("origin")
    if node is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return floats(node.get("xyz", "0 0 0")), floats(node.get("rpy", "0 0 0"))


def main(urdf_path: Path, out_path: Path) -> None:
    root = ET.parse(urdf_path).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}

    joints = []
    for i in range(1, 8):
        j = by_name[f"panda_joint{i}"]
        xyz, rpy = origin_of(j)
        limit = j.find("limit")
        joints.append({
            "name": j.get("name"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "fixed": homogeneous(xyz, rpy),
            "axis": floats(j.find("axis").get("xyz")),
            "lower": float(limit.get("lower")),
            "upper": float(limit.get("upper")),
            "velocity": float(limit.get("velocity")),
            "effort": float(limit.get("effort")),
        })

    # panda_link7 -> panda_link8 -> panda_hand -> TCP, collapsed to one constant.
    tip = homogeneous(*origin_of(by_name["panda_joint8"]))
    tip = matmul4(tip, homogeneous(*origin_of(by_name["panda_hand_joint"])))
    tip = matmul4(tip, homogeneous([0.0, 0.0, HAND_TO_TCP_Z], [0.0, 0.0, 0.0]))

    chain = {
        "name": root.get("name"),
        "generated_by": "tools/extract_chain.py",
        "source_urdf": "assets/franka/urdf/panda.urdf",
        "base_link": "panda_link0",
        "tip_frame": "panda_tcp",
        "tip_note": f"panda_link7 -> panda_hand -> TCP (+{HAND_TO_TCP_Z} m along hand z)",
        "dof": 7,
        "joints": joints,
        "flange_to_tcp": tip,
        # Franka's documented ready pose. Used as the IK seed and as the
        # null-space attractor so redundancy resolves the same way every frame.
        "q_neutral": [0.0, -0.785398163397, 0.0, -2.35619449019, 0.0,
                      1.57079632679, 0.785398163397],
        "gripper_max_width": GRIPPER_MAX_WIDTH,
    }
    out_path.write_text(json.dumps(chain, indent=2) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, dof={chain['dof']})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
