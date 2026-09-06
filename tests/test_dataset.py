"""The corpus is a measurement instrument, so it gets the same scrutiny as one.

These tests do not re-derive the renderer's arithmetic; they check the four
properties the benchmark actually leans on: the frame store has the byte layout
docs/FORMATS.md promises, a seed reproduces it exactly, the ground truth in the
manifest is really the geometry the depth image encodes, and essentially every
frame gives the pipeline something to grasp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "harness"))

import make_dataset  # noqa: E402

FRAMES = 12
WIDTH, HEIGHT = 320, 240


@pytest.fixture(scope="module")
def cfg() -> dict:
    return make_dataset.load_config()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, cfg) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("corpus")
    manifest = make_dataset.generate(
        "table_test", WIDTH, HEIGHT, FRAMES, cfg["ransac_table"]["seed"], out, cfg=cfg
    )
    return out / "table_test", manifest


def deproject(store: Path, manifest: dict, frame: dict, cfg: dict) -> np.ndarray:
    """S1 + S2, written out independently of the pipeline implementations."""
    w, h = manifest["width"], manifest["height"]
    depth = np.fromfile(store / frame["depth"], dtype="<u2").reshape(h, w)
    z = depth.astype(np.float64) * manifest["depth_scale_m"]
    v, u = np.mgrid[0:h, 0:w]
    keep = (z >= cfg["deproject"]["z_min_m"]) & (z <= cfg["deproject"]["z_max_m"])
    z = z[keep]
    points_cam = np.stack(
        [
            (u[keep] - manifest["cx"]) * z / manifest["fx"],
            (v[keep] - manifest["cy"]) * z / manifest["fy"],
            z,
        ],
        axis=1,
    )
    T = np.asarray(cfg["camera"]["T_base_cam"], dtype=np.float64).reshape(4, 4)
    return points_cam @ T[:3, :3].T + T[:3, 3]


def test_manifest_shape(corpus, cfg):
    store, manifest = corpus
    for key in ("name", "width", "height", "fx", "fy", "cx", "cy", "depth_scale_m",
                "seed", "frame_count", "frames"):
        assert key in manifest, f"manifest is missing {key}"
    assert manifest["frame_count"] == FRAMES == len(manifest["frames"])

    scale = WIDTH / cfg["camera"]["reference_width"]
    assert manifest["fx"] == pytest.approx(cfg["camera"]["fx"] * scale)
    assert manifest["cy"] == pytest.approx(cfg["camera"]["cy"] * scale)
    assert json.loads((store / "manifest.json").read_text()) == manifest

    for index, frame in enumerate(manifest["frames"]):
        assert frame["id"] == index
        assert frame["depth"] == f"{index:06d}.depth.bin"
        assert frame["rgb"] == f"{index:06d}.rgb.bin"


def test_frame_byte_sizes(corpus):
    store, manifest = corpus
    w, h = manifest["width"], manifest["height"]
    for frame in manifest["frames"]:
        assert (store / frame["depth"]).stat().st_size == h * w * 2
        assert (store / frame["rgb"]).stat().st_size == h * w * 3


def test_same_seed_is_byte_identical(corpus, cfg, tmp_path):
    store, manifest = corpus
    again = make_dataset.generate(
        "table_test", WIDTH, HEIGHT, FRAMES, manifest["seed"], tmp_path, cfg=cfg
    )
    assert again == manifest
    for frame in manifest["frames"]:
        for key in ("depth", "rgb"):
            expected = (store / frame[key]).read_bytes()
            assert (tmp_path / "table_test" / frame[key]).read_bytes() == expected


def test_a_different_seed_is_a_different_corpus(corpus, cfg, tmp_path):
    store, manifest = corpus
    other = make_dataset.generate(
        "table_test", WIDTH, HEIGHT, 2, manifest["seed"] + 1, tmp_path, cfg=cfg
    )
    assert other["frames"][0]["truth"] != manifest["frames"][0]["truth"]


def test_scene_does_not_depend_on_resolution(corpus, cfg, tmp_path):
    """Only the sampling of the scene must be resolution-free -- that is what
    lets the four corpora be the same experiment at four pixel counts."""
    _, manifest = corpus
    wide = make_dataset.generate(
        "table_test_640", 640, 480, FRAMES, manifest["seed"], tmp_path, cfg=cfg
    )
    for small, large in zip(manifest["frames"], wide["frames"]):
        assert small["truth"] == large["truth"]


def test_depth_is_in_range_or_the_invalid_sentinel(corpus, cfg):
    store, manifest = corpus
    scale = manifest["depth_scale_m"]
    for frame in manifest["frames"]:
        depth = np.fromfile(store / frame["depth"], dtype="<u2")
        valid = depth[depth != make_dataset.INVALID_DEPTH].astype(np.float64) * scale
        assert valid.min() >= cfg["deproject"]["z_min_m"]
        assert valid.max() <= cfg["deproject"]["z_max_m"]
        # The table is finite and the sensor drops grazing hits, so a frame
        # with no invalid returns at all would mean the sensor model is off.
        invalid_fraction = float(np.mean(depth == make_dataset.INVALID_DEPTH))
        assert 0.0 < invalid_fraction < 0.5


def test_rgb_is_not_degenerate(corpus):
    store, manifest = corpus
    rgb = np.fromfile(store / manifest["frames"][0]["rgb"], dtype=np.uint8)
    assert len(np.unique(rgb)) > 8


def test_truth_matches_the_rendered_geometry(corpus, cfg):
    """Deproject the depth image and check the object is where the manifest
    says it is: the column of points over the truth centre must sit on the
    object's top face, not on the table."""
    store, manifest = corpus
    ws = cfg["workspace"]
    for frame in manifest["frames"]:
        truth = frame["truth"]
        cx, cy, _ = truth["object_center"]
        points = deproject(store, manifest, frame, cfg)
        inside = (
            (points[:, 0] >= ws["x_min"]) & (points[:, 0] <= ws["x_max"])
            & (points[:, 1] >= ws["y_min"]) & (points[:, 1] <= ws["y_max"])
            & (points[:, 2] >= ws["z_min"]) & (points[:, 2] <= ws["z_max"])
        )
        points = points[inside]

        near = points[np.hypot(points[:, 0] - cx, points[:, 1] - cy) < 0.006]
        assert len(near) >= 5, f"frame {frame['id']}: no depth over the truth centre"
        # 1.1 mm of axial noise plus 0.5 mm of quantisation, so 4 mm is loose
        # enough to never flake and tight enough to catch a real pose error.
        assert np.median(near[:, 2]) == pytest.approx(truth["z_top"], abs=0.004)

        plane = np.asarray(truth["plane"])
        table_z = (-plane[3] - plane[0] * cx - plane[1] * cy) / plane[2]
        assert truth["z_top"] - table_z == pytest.approx(truth["extent"][2], abs=1e-6)

        # Nothing stands anywhere near that far out, so a ring well clear of
        # every footprint must lie on the truth plane itself (which is tilted,
        # hence the point-to-plane residual rather than a height).
        far = points[np.hypot(points[:, 0] - cx, points[:, 1] - cy) > 0.25]
        assert np.median(np.abs(far @ plane[:3] + plane[3])) < 0.002


def test_every_frame_offers_a_grasp(corpus, cfg):
    _, manifest = corpus
    max_width = cfg["grasp"]["max_width_m"]
    graspable = [f["truth"]["width"] <= max_width for f in manifest["frames"]]
    assert np.mean(graspable) >= 0.95
    assert all(f["truth"]["object_count"] >= 1 for f in manifest["frames"])


def test_object_footprint_is_wide_enough_to_cluster(corpus, cfg):
    """A cluster below min_points is dropped by S4, which would make the frame
    measure a bail-out instead of the full pipeline."""
    store, manifest = corpus
    cut = cfg["plane"]["inlier_threshold_m"] + cfg["plane"]["clearance_m"]
    for frame in manifest["frames"]:
        truth = frame["truth"]
        points = deproject(store, manifest, frame, cfg)
        plane = np.asarray(truth["plane"])
        above = points[points @ plane[:3] + plane[3] >= cut]
        assert len(above) >= cfg["cluster"]["min_points"], f"frame {frame['id']}"


def test_bag_round_trips_depth_bit_exactly(cfg, tmp_path):
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    frames = 3
    manifest = make_dataset.generate(
        "table_bag", WIDTH, HEIGHT, frames, cfg["ransac_table"]["seed"], tmp_path,
        cfg=cfg, bag=True,
    )
    store = tmp_path / "table_bag"
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    period_ns = round(1e9 / cfg["benchmark"]["deadline_hz"])

    seen = {cfg["ros"]["depth_topic"]: 0, cfg["ros"]["color_topic"]: 0,
            cfg["ros"]["camera_info_topic"]: 0}
    with Reader(store / "bag") as reader:
        assert {c.topic for c in reader.connections} == set(seen)
        for connection, timestamp, raw in reader.messages():
            index = seen[connection.topic]
            assert timestamp == cfg["ros"]["bag_start_ns"] + index * period_ns
            msg = typestore.deserialize_cdr(raw, connection.msgtype)
            if connection.topic == cfg["ros"]["camera_info_topic"]:
                assert msg.k[0] == pytest.approx(manifest["fx"])
                assert msg.k[5] == pytest.approx(manifest["cy"])
            else:
                key = "depth" if connection.topic == cfg["ros"]["depth_topic"] else "rgb"
                encoding = "16UC1" if key == "depth" else "rgb8"
                assert msg.encoding == encoding
                assert (msg.height, msg.width) == (HEIGHT, WIDTH)
                assert msg.data.tobytes() == (store / f"{index:06d}.{key}.bin").read_bytes()
            seen[connection.topic] += 1
    assert all(count == frames for count in seen.values())
