#!/usr/bin/env python3
"""Deterministic RGB-D corpus generator: an analytic ray-caster, not a renderer.

Frames are not committed (docs/FORMATS.md); they are regenerated from a seed,
which only works if the renderer is bit-reproducible. That rules out a
rasteriser -- there is no GPU, no OpenGL and no EGL here or in CI, and a
software rasteriser would still owe us exact ground truth separately. Casting
one ray per pixel against a plane, boxes and cylinders gives both at once:
closed-form hit points, closed-form normals, and a truth block that *is* the
scene rather than a measurement of it.

The sensor model on top of the geometry is not decoration. RANSAC's inlier
count, the cluster's point count and the grasp width all move with depth noise
and dropout, so a noiseless corpus would make the benchmark easier than
reality in precisely the stages that dominate its latency.

Every frame is guaranteed to contain a graspable object. A frame that bails
out at S5 is a legitimate latency sample but a *shorter* pipeline, and a corpus
full of them would measure the wrong thing; the interesting variance lives in
point counts, cluster size and IK iterations instead.

Usage:
  make_dataset.py --name table_640x480 --width 640 --height 480 --frames 100
  make_dataset.py --name table_640x480_bag --frames 100 --bag
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "assets" / "pipeline_config.json"

# Rays that hit nothing, and hits the sensor rejects, are written as this.
INVALID_DEPTH = 0


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# Camera


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    origin: np.ndarray  # (3,) camera centre in base
    dirs: np.ndarray  # (N, 3) ray directions in base, z_cam-normalised
    dirs_unit: np.ndarray  # (N, 3) the same rays, unit length


def make_camera(cfg: dict, width: int, height: int) -> Camera:
    cam = cfg["camera"]
    scale = width / cam["reference_width"]
    fx, fy = cam["fx"] * scale, cam["fy"] * scale
    cx, cy = cam["cx"] * scale, cam["cy"] * scale

    T = np.asarray(cam["T_base_cam"], dtype=np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]

    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    # Third component is exactly 1, so the ray parameter is z_cam -- which is
    # what a depth image stores. No range-to-depth conversion anywhere.
    d_cam = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    dirs = d_cam.reshape(-1, 3) @ R.T
    dirs_unit = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    return Camera(width, height, fx, fy, cx, cy, t.copy(), dirs, dirs_unit)


# --------------------------------------------------------------------------
# Scene primitives. Each returns +inf where a ray misses.


@dataclass
class TablePlane:
    normal: np.ndarray  # (3,) unit, n_z > 0
    d: float  # plane is dot(n, p) + d = 0
    pivot: np.ndarray  # (2,)
    half_x: float
    half_y: float

    def height_at(self, x: float, y: float) -> float:
        n = self.normal
        return float((-self.d - n[0] * x - n[1] * y) / n[2])

    def intersect(self, o: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        denom = dirs @ self.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -(o @ self.normal + self.d) / denom
        p = o + t[:, None] * dirs
        inside = (np.abs(p[:, 0] - self.pivot[0]) <= self.half_x) & (
            np.abs(p[:, 1] - self.pivot[1]) <= self.half_y
        )
        return np.where(np.isfinite(t) & (t > 0.0) & inside, t, np.inf)

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.normal, p.shape)


@dataclass
class Box:
    center: np.ndarray  # (3,) geometric centre
    half: np.ndarray  # (3,) half extents in the yawed local frame
    yaw: float

    def _to_local(self, o: np.ndarray, dirs: np.ndarray):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return (o - self.center) @ Rz, dirs @ Rz

    def intersect(self, o: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        o_l, d_l = self._to_local(o, dirs)
        # Nudge exact zeros rather than dividing by them: a 0 * inf slab product
        # is NaN, and one NaN poisons the whole min/max reduction.
        d_l = np.where(np.abs(d_l) < 1e-12, 1e-12, d_l)
        inv = 1.0 / d_l
        t1 = (-self.half - o_l) * inv
        t2 = (self.half - o_l) * inv
        t_near = np.minimum(t1, t2).max(axis=1)
        t_far = np.maximum(t1, t2).min(axis=1)
        t = np.where(t_near > 0.0, t_near, t_far)
        hit = (t_far >= np.maximum(t_near, 0.0)) & (t > 0.0)
        return np.where(hit, t, np.inf)

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        p_l = (p - self.center) @ Rz
        axis = np.argmax(np.abs(p_l) / self.half, axis=1)
        n_l = np.zeros_like(p_l)
        rows = np.arange(p_l.shape[0])
        n_l[rows, axis] = np.sign(p_l[rows, axis])
        return n_l @ Rz.T


@dataclass
class Cylinder:
    center: np.ndarray  # (3,) centre of the axis segment
    radius: float
    half_height: float

    @property
    def z_top(self) -> float:
        return float(self.center[2] + self.half_height)

    @property
    def z_bottom(self) -> float:
        return float(self.center[2] - self.half_height)

    def intersect(self, o: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        ox, oy = o[0] - self.center[0], o[1] - self.center[1]
        dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - self.radius * self.radius
        disc = b * b - 4.0 * a * c

        with np.errstate(divide="ignore", invalid="ignore"):
            sq = np.sqrt(np.maximum(disc, 0.0))
            roots = ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a))
            caps = ((self.z_top - o[2]) / dz, (self.z_bottom - o[2]) / dz)

        best = np.full(dirs.shape[0], np.inf)
        for t in roots:
            z = o[2] + t * dz
            ok = (disc > 0.0) & (a > 0.0) & (t > 0.0) & (z >= self.z_bottom) & (z <= self.z_top)
            best = np.where(ok & (t < best), t, best)
        for t in caps:
            r2 = (ox + t * dx) ** 2 + (oy + t * dy) ** 2
            ok = np.isfinite(t) & (t > 0.0) & (r2 <= self.radius * self.radius)
            best = np.where(ok & (t < best), t, best)
        return best

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        on_cap = np.abs(p[:, 2] - self.z_top) < 1e-9
        radial = np.stack(
            [p[:, 0] - self.center[0], p[:, 1] - self.center[1], np.zeros(p.shape[0])], axis=1
        )
        norm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial = np.divide(radial, norm, out=np.zeros_like(radial), where=norm > 1e-12)
        cap = np.zeros_like(radial)
        cap[:, 2] = np.where(on_cap, 1.0, -1.0)
        return np.where(on_cap[:, None] | (norm < 1e-12), cap, radial)


# --------------------------------------------------------------------------
# Scene sampling


@dataclass
class Scene:
    table: TablePlane
    objects: list
    truth: dict


def _sample_size(rng: np.random.Generator, obj_cfg: dict) -> dict:
    """Footprint and height of one primitive, in metres."""
    height = rng.uniform(*obj_cfg["height_m"])
    if rng.random() < obj_cfg["box_probability"]:
        narrow = rng.uniform(*obj_cfg["narrow_half_m"])
        long = narrow * rng.uniform(*obj_cfg["long_ratio"])
        # Which of the two footprint axes is the narrow one is itself random,
        # so the minor principal axis is not always the box's local x.
        half = (narrow, long) if rng.random() < 0.5 else (long, narrow)
        return {
            "kind": "box",
            "half_xy": np.array(half),
            "height": height,
            "yaw": float(rng.uniform(0.0, np.pi)),
            "area": 4.0 * half[0] * half[1],
            "radius": float(np.hypot(*half)),
            "narrow_extent": 2.0 * min(half),
        }
    r = rng.uniform(*obj_cfg["cylinder_radius_m"])
    return {
        "kind": "cylinder",
        "half_xy": np.array([r, r]),
        "height": height,
        "yaw": 0.0,
        "area": float(np.pi * r * r),
        "radius": float(r),
        "narrow_extent": 2.0 * r,
    }


def sample_scene(rng: np.random.Generator, cfg: dict) -> Scene:
    scene_cfg = cfg["scene"]
    table_cfg, obj_cfg = scene_cfg["table"], scene_cfg["objects"]
    grasp = cfg["grasp"]

    tilt = rng.uniform(0.0, table_cfg["tilt_max_rad"])
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    normal = np.array(
        [np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)]
    )
    pivot = np.asarray(table_cfg["pivot_xy"], dtype=np.float64)
    pivot_z = table_cfg["height_m"] + rng.uniform(
        -table_cfg["height_jitter_m"], table_cfg["height_jitter_m"]
    )
    table = TablePlane(
        normal=normal,
        d=float(-(normal[0] * pivot[0] + normal[1] * pivot[1] + normal[2] * pivot_z)),
        pivot=pivot,
        half_x=table_cfg["half_x"],
        half_y=table_cfg["half_y"],
    )

    count = int(rng.integers(1, obj_cfg["count_max"] + 1))
    ratio = obj_cfg["primary_area_ratio"]
    for _ in range(64):
        sizes = sorted((_sample_size(rng, obj_cfg) for _ in range(count)),
                       key=lambda s: -s["area"])
        if count == 1 or sizes[0]["area"] >= ratio * sizes[1]["area"]:
            break
    else:  # pragma: no cover - the ratio is satisfiable for any drawn count
        sizes = sizes[:1]

    placed = []
    for size in sizes:
        for _ in range(256):
            x = rng.uniform(*obj_cfg["place_x_m"])
            y = rng.uniform(*obj_cfg["place_y_m"])
            gap = obj_cfg["min_gap_m"]
            if all(
                np.hypot(x - p["x"], y - p["y"]) >= size["radius"] + p["radius"] + gap
                for p in placed
            ):
                placed.append({**size, "x": x, "y": y})
                break

    objects, records = [], []
    for size in placed:
        base_z = table.height_at(size["x"], size["y"])
        center = np.array([size["x"], size["y"], base_z + 0.5 * size["height"]])
        if size["kind"] == "box":
            half = np.array([size["half_xy"][0], size["half_xy"][1], 0.5 * size["height"]])
            objects.append(Box(center=center, half=half, yaw=size["yaw"]))
        else:
            objects.append(
                Cylinder(center=center, radius=size["radius"], half_height=0.5 * size["height"])
            )
        records.append({**size, "center": center, "z_top": base_z + size["height"]})

    widths = [r["narrow_extent"] + grasp["finger_clearance_m"] for r in records]
    assert any(w <= grasp["max_width_m"] for w in widths), (
        f"no graspable object in scene: widths {widths} all exceed "
        f"{grasp['max_width_m']} m"
    )
    # sizes[] is area-sorted and placement never drops the first, so the
    # largest footprint -- the cluster S4 will pick -- is index 0.
    primary = records[0]
    assert widths[0] <= grasp["max_width_m"], "primary object is not graspable"

    truth = {
        "object_center": [round(float(v), 9) for v in primary["center"]],
        "yaw": round(float(primary["yaw"]), 9),
        "width": round(float(widths[0]), 9),
        "shape": primary["kind"],
        "z_top": round(float(primary["z_top"]), 9),
        "extent": [
            round(float(2.0 * primary["half_xy"][0]), 9),
            round(float(2.0 * primary["half_xy"][1]), 9),
            round(float(primary["height"]), 9),
        ],
        "object_count": len(objects),
        "plane": [round(float(v), 9) for v in table.normal] + [round(table.d, 9)],
    }
    return Scene(table=table, objects=objects, truth=truth)


# --------------------------------------------------------------------------
# Render


def render(cam: Camera, scene: Scene, rng: np.random.Generator, cfg: dict):
    """Return (depth uint16 millimetres, rgb uint8) for one frame."""
    surfaces = [scene.table] + scene.objects
    n_pix = cam.dirs.shape[0]

    best_t = np.full(n_pix, np.inf)
    surface_id = np.zeros(n_pix, dtype=np.int32)
    for sid, surface in enumerate(surfaces, start=1):
        t = surface.intersect(cam.origin, cam.dirs)
        closer = t < best_t
        best_t[closer] = t[closer]
        surface_id[closer] = sid

    normals = np.zeros((n_pix, 3))
    for sid, surface in enumerate(surfaces, start=1):
        mask = surface_id == sid
        if mask.any():
            hits = cam.origin + best_t[mask, None] * cam.dirs[mask]
            normals[mask] = surface.normal_at(hits)

    sensor = cfg["sensor"]
    depth_scale = cfg["camera"]["depth_scale_m"]
    z_min, z_max = cfg["deproject"]["z_min_m"], cfg["deproject"]["z_max_m"]

    # Draw both noise fields unconditionally: the stream then depends only on
    # the resolution, never on which pixels happened to hit something.
    z = np.where(surface_id > 0, best_t, 0.0)
    sigma = sensor["axial_noise_sigma_coeff"] * z * z
    z_noisy = z + sigma * rng.standard_normal(n_pix)

    cos_incidence = np.abs(np.einsum("ij,ij->i", normals, cam.dirs_unit))
    threshold = sensor["grazing_cos_threshold"]
    steepness = np.clip((threshold - cos_incidence) / threshold, 0.0, 1.0)
    p_drop = sensor["baseline_dropout_prob"] + steepness * (
        sensor["grazing_dropout_prob"] - sensor["baseline_dropout_prob"]
    )
    kept = rng.random(n_pix) >= p_drop

    millimetres = np.rint(z_noisy / depth_scale)
    valid = (
        (surface_id > 0)
        & kept
        & (z_noisy >= z_min)
        & (z_noisy <= z_max)
        & (millimetres >= 1.0)
        & (millimetres <= 65535.0)
    )
    depth = np.where(valid, millimetres, INVALID_DEPTH).astype("<u2")

    shading = cfg["scene"]["shading"]
    light = np.asarray(shading["light_dir_base"], dtype=np.float64)
    light /= np.linalg.norm(light)
    ambient = shading["ambient"]
    palette = np.asarray(shading["albedo"], dtype=np.float64)
    albedo = palette[(surface_id - 1) % len(palette)]
    lambert = np.clip(normals @ light, 0.0, 1.0)
    shade = ambient + (1.0 - ambient) * lambert
    rgb = np.clip(albedo * shade[:, None] * 255.0, 0.0, 255.0)
    rgb = np.where(surface_id[:, None] > 0, rgb, 0.0).astype(np.uint8)

    return depth.reshape(cam.height, cam.width), rgb.reshape(cam.height, cam.width, 3)


# --------------------------------------------------------------------------
# Frame store and bag


def _manifest(cam: Camera, name: str, seed: int, frames: list, cfg: dict) -> dict:
    return {
        "name": name,
        "width": cam.width,
        "height": cam.height,
        "fx": cam.fx,
        "fy": cam.fy,
        "cx": cam.cx,
        "cy": cam.cy,
        "depth_scale_m": cfg["camera"]["depth_scale_m"],
        "seed": seed,
        "frame_count": len(frames),
        "generated_by": "harness/make_dataset.py",
        "frames": frames,
    }


def write_bag(out_dir: Path, cam: Camera, cfg: dict, frame_count: int) -> Path:
    """Write a rosbag2 the ROS 2 nodes can replay, from the frames on disk."""
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    CameraInfo = typestore.types["sensor_msgs/msg/CameraInfo"]
    RegionOfInterest = typestore.types["sensor_msgs/msg/RegionOfInterest"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    ros = cfg["ros"]
    period_ns = round(1e9 / cfg["benchmark"]["deadline_hz"])
    bag_dir = out_dir / "bag"
    if bag_dir.exists():
        for leftover in sorted(bag_dir.iterdir()):
            leftover.unlink()
        bag_dir.rmdir()

    def header(stamp_ns: int, frame_id: str):
        return Header(stamp=Time(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9),
                      frame_id=frame_id)

    with Writer(bag_dir, version=8) as writer:
        conns = {
            "depth": writer.add_connection(
                ros["depth_topic"], "sensor_msgs/msg/Image", typestore=typestore
            ),
            "color": writer.add_connection(
                ros["color_topic"], "sensor_msgs/msg/Image", typestore=typestore
            ),
            "info": writer.add_connection(
                ros["camera_info_topic"], "sensor_msgs/msg/CameraInfo", typestore=typestore
            ),
        }
        k = np.array([cam.fx, 0.0, cam.cx, 0.0, cam.fy, cam.cy, 0.0, 0.0, 1.0])
        p = np.array([cam.fx, 0.0, cam.cx, 0.0, 0.0, cam.fy, cam.cy, 0.0, 0.0, 0.0, 1.0, 0.0])
        for index in range(frame_count):
            stamp = ros["bag_start_ns"] + index * period_ns
            depth = (out_dir / f"{index:06d}.depth.bin").read_bytes()
            rgb = (out_dir / f"{index:06d}.rgb.bin").read_bytes()

            depth_msg = Image(
                header=header(stamp, ros["depth_frame_id"]),
                height=cam.height, width=cam.width, encoding="16UC1",
                is_bigendian=0, step=cam.width * 2,
                data=np.frombuffer(depth, dtype=np.uint8),
            )
            color_msg = Image(
                header=header(stamp, ros["color_frame_id"]),
                height=cam.height, width=cam.width, encoding="rgb8",
                is_bigendian=0, step=cam.width * 3,
                data=np.frombuffer(rgb, dtype=np.uint8),
            )
            info_msg = CameraInfo(
                header=header(stamp, ros["depth_frame_id"]),
                height=cam.height, width=cam.width, distortion_model="plumb_bob",
                d=np.zeros(5), k=k, r=np.eye(3).reshape(-1), p=p,
                binning_x=0, binning_y=0,
                roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0,
                                     do_rectify=False),
            )
            writer.write(conns["depth"], stamp, typestore.serialize_cdr(depth_msg,
                                                                       "sensor_msgs/msg/Image"))
            writer.write(conns["color"], stamp, typestore.serialize_cdr(color_msg,
                                                                        "sensor_msgs/msg/Image"))
            writer.write(conns["info"], stamp, typestore.serialize_cdr(
                info_msg, "sensor_msgs/msg/CameraInfo"))
    return bag_dir


def verify_bag(bag_dir: Path, out_dir: Path, cfg: dict, frame_count: int) -> None:
    """Depth must survive the CDR round trip bit-for-bit, or the nodes are
    replaying something the frame store never contained."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    depth_topic = cfg["ros"]["depth_topic"]
    seen = 0
    with Reader(bag_dir) as reader:
        connections = [c for c in reader.connections if c.topic == depth_topic]
        assert connections, f"{depth_topic} missing from {bag_dir}"
        for _, _, raw in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(raw, "sensor_msgs/msg/Image")
            expected = (out_dir / f"{seen:06d}.depth.bin").read_bytes()
            assert msg.data.tobytes() == expected, f"depth mismatch on frame {seen}"
            seen += 1
    assert seen == frame_count, f"bag holds {seen} depth messages, expected {frame_count}"


def generate(
    name: str,
    width: int,
    height: int,
    frame_count: int,
    seed: int,
    out_root: Path,
    cfg: dict | None = None,
    bag: bool = False,
) -> dict:
    cfg = cfg or load_config()
    cam = make_camera(cfg, width, height)
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # One SeedSequence child per frame, split again into scene and sensor, so
    # the scene is identical across resolutions while the noise field is not.
    frame_seeds = np.random.SeedSequence(seed).spawn(frame_count)
    frames = []
    for index, frame_seed in enumerate(frame_seeds):
        scene_seed, sensor_seed = frame_seed.spawn(2)
        scene = sample_scene(np.random.default_rng(scene_seed), cfg)
        depth, rgb = render(cam, scene, np.random.default_rng(sensor_seed), cfg)

        depth_name, rgb_name = f"{index:06d}.depth.bin", f"{index:06d}.rgb.bin"
        (out_dir / depth_name).write_bytes(depth.tobytes())
        (out_dir / rgb_name).write_bytes(rgb.tobytes())
        frames.append({"id": index, "depth": depth_name, "rgb": rgb_name,
                       "truth": scene.truth})

    manifest = _manifest(cam, name, seed, frames, cfg)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if bag:
        bag_dir = write_bag(out_dir, cam, cfg, frame_count)
        verify_bag(bag_dir, out_dir, cfg, frame_count)

    return manifest


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--width", type=int, default=cfg["camera"]["reference_width"])
    parser.add_argument("--height", type=int, default=cfg["camera"]["reference_height"])
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=cfg["ransac_table"]["seed"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--bag", action="store_true",
                        help="also write a rosbag2 under <out>/<name>/bag")
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = generate(args.name, args.width, args.height, args.frames, args.seed,
                        args.out, cfg=cfg, bag=args.bag)
    elapsed = time.perf_counter() - started

    out_dir = Path(args.out) / args.name
    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(
        f"{manifest['name']}: {manifest['frame_count']} frames "
        f"{manifest['width']}x{manifest['height']} in {elapsed:.2f} s "
        f"({elapsed / max(manifest['frame_count'], 1) * 1e3:.1f} ms/frame, "
        f"{total_bytes / 1e6:.1f} MB) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
