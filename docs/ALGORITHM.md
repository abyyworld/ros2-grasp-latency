# The pipeline, specified once

Two implementations of the same pipeline only tell you something about the two
*languages* if they are the same pipeline. This document is the contract. It
specifies the output of each stage precisely enough that C++ and Python must
agree numerically, while deliberately **not** specifying loop structure. Each
implementation is free to be idiomatic, because that is the comparison worth
making.

> **The fairness rule.** The Python implementation must be the best reasonable
> NumPy/SciPy implementation of this spec, not a transliteration of the C++.
> Forcing scalar Python loops where a competent engineer would vectorise would
> manufacture the result. Where a stage is inherently sequential, both
> implementations pay for it; where it vectorises, Python is allowed to
> vectorise. See [METHOD.md](METHOD.md) for why this is the load-bearing
> methodological choice in the whole repository.

Every constant below lives in [`assets/pipeline_config.json`](../assets/pipeline_config.json).
Neither implementation may hard-code one.

## Frames

| Frame | Definition |
|---|---|
| `cam` | Camera optical frame: **x** right, **y** down, **z** forward. |
| `base` | `panda_link0`, **z** up. |
| `tcp` | Tool centre point: `panda_hand` translated +0.1034 m along its z. |

`T_base_cam` is fixed and given in the config. The camera looks straight down
from 0.85 m, 0.30 m in front of the base.

## Determinism rules

These apply to every stage. They exist so that "the two implementations
disagree" always means a real bug and never a coin flip.

1. **Reductions accumulate in float64**, even when the input is float32.
2. **Ties break toward the lowest index** (or lexicographically smallest key).
   Never toward "whichever the container happened to yield first".
3. **Eigenvectors are canonicalised**: flip each so that its largest-magnitude
   component is positive; if two components tie in magnitude, the lower index
   decides the sign.
4. **No PRNG runs inside the measured path.** RANSAC reads
   `assets/ransac_uniform.bin` (see `tools/make_ransac_table.py`).
5. **No I/O, allocation of the model, or config parsing inside the measured
   path.** All of that happens at construction.

Agreement is asserted to `1e-6` on the grasp pose (metres and radians) and
`1e-6` rad on joint angles. That is far tighter than any physical effect and
far looser than the ~1e-15 noise from differing summation orders.

---

## S0. Decode

**In:** `depth_raw`, `H*W` bytes of little-endian `uint16` in millimetres.
`rgb_raw`, `H*W*3` bytes of `uint8`.
**Out:** typed 2-D views of the same memory. No copy.

This stage exists because it is real: `rclpy` hands you a `bytes` object and
you pay to get an array out of it. It is measured, not skipped.

## S1. Deproject

**In:** depth view, intrinsics `fx, fy, cx, cy` (rescaled from the reference
resolution by `W / reference_width`), `stride`, `z_min_m`, `z_max_m`.
**Out:** `points_cam`, `N x 3` float32.

For every pixel `(u, v)` with `u % stride == 0` and `v % stride == 0`:

```
z = depth[v, u] * depth_scale_m
if z < z_min_m or z > z_max_m:  drop      # a zero depth is an invalid return
x = (u - cx) * z / fx
y = (v - cy) * z / fy
emit (x, y, z)
```

Surviving points keep **row-major pixel order**. Order matters: it is what
makes the RANSAC index table select the same points in both implementations.

## S2. Transform and crop

**In:** `points_cam`, `T_base_cam`, workspace AABB.
**Out:** `points_base`, `M x 3` float32, order preserved.

```
p_base = R_base_cam @ p_cam + t_base_cam
keep iff  x_min <= x <= x_max  and  y_min <= y <= y_max  and  z_min <= z <= z_max
```

## S3. Plane removal (RANSAC)

**In:** `points_base` (`M` points), the deviate table `u[0 .. 3*iterations)`.
**Out:** `points_object`, the plane `(n, d)` with `n` unit and `n_z > 0`, and
`plane_found`.

If `M < 3`, return `plane_found = false` and pass the points through.

For iteration `k` in `[0, iterations)`:

```
i0 = min(M-1, floor(u[3k+0] * M))
i1 = min(M-1, floor(u[3k+1] * M))
i2 = min(M-1, floor(u[3k+2] * M))
if i0 == i1 or i1 == i2 or i0 == i2:  score = 0; continue

n = cross(p[i1] - p[i0], p[i2] - p[i0])            # float64
if |n| < 1e-12:                       score = 0; continue
n = n / |n|
if n_z < 0:  n = -n                                # orient upward
if n_z < min_horizontal_cos:          score = 0; continue   # not a table
d = -dot(n, p[i0])
score = count of points with |dot(n, p) + d| < inlier_threshold_m
```

Keep the candidate with the **strictly greatest** score, so the earliest
iteration wins a tie. If the best score `< min_inliers`, return
`plane_found = false`.

**Refit.** Recompute the plane by least squares over the winning inlier set:
`n` is the eigenvector of the inliers' 3x3 covariance for the smallest
eigenvalue (canonicalised per rule 3, then oriented so `n_z > 0`), and
`d = -dot(n, centroid)`.

**Remove.** Drop every point with
`dot(n, p) + d < inlier_threshold_m + clearance_m`. This takes out the table
and everything under it in one comparison, leaving only what stands on it.

## S4. Cluster

**In:** `points_object`, `voxel_size_m`, `min_points`.
**Out:** the index set of one cluster, or empty.

Voxelise: `key(p) = (floor(x/v), floor(y/v), floor(z/v))`. Occupied voxels form
a set; two voxels are connected if their keys differ by 1 in exactly one axis
(**6-connectivity**). Take connected components over that set.

Return the component holding the **most points** (not the most voxels). Ties
break toward the lexicographically smallest `(kx, ky, kz)` in the component.
If the best component has fewer than `min_points` points, return empty.

*Implementations may differ freely here*: a dense grid with
`scipy.ndimage.label`, or a hash map with BFS, as long as the returned index
set matches.

## S5. Grasp synthesis

**In:** the cluster's points, the plane, gripper geometry.
**Out:** `T_base_tcp` (4x4), `width`, `graspable`.

1. `c` = centroid of the cluster (float64).
2. Eigendecompose the **2-D** covariance of the cluster's `(x, y)`. Canonicalise
   per rule 3. The **minor** axis `a2` (smaller eigenvalue) is the narrow
   direction, the one the fingers close along.
3. `width` = extent of the cluster projected onto `a2`
   (`max - min`) plus `finger_clearance_m`.
   If `width > max_width_m`, return `graspable = false`.
4. Build a right-handed TCP frame:
   ```
   z_tcp = approach_axis_base           # (0, 0, -1), straight down
   y_tcp = normalise((a2_x, a2_y, 0))   # fingers close along this
   ```

   **Resolve the closing-axis sign against the wrist, not against the
   eigenvector.** A parallel-jaw gripper is symmetric about its closing axis:
   closing along `+y` and along `-y` are the same physical grasp. Rule 3's
   canonicalisation picks a sign from the eigenvector's components, which knows
   nothing about the arm, and that confines the demanded yaw to two quadrants
   and can ask joint 7 for up to 180 degrees of travel from `q_neutral`. Joint
   7 only has 2.182 rad of headroom upward, so those grasps are orientation
   unreachable and IK burns all `max_iterations` failing to reach them.

   So fold the ambiguity toward the wrist's rest orientation:
   ```
   y_ref = horizontal component of the TCP y axis at q_neutral, normalised
           # computed once at construction, never in the measured path
   if dot(y_tcp, y_ref) < 0:      y_tcp = -y_tcp
   elif dot(y_tcp, y_ref) == 0:   flip so y_tcp_x > 0, or if y_tcp_x == 0, y_tcp_y > 0
   ```
   This caps the demanded wrist rotation at 90 degrees while naming the
   identical grasp. It is a change to which of two equivalent frames is
   reported, not to which object is grasped or where.

   ```
   x_tcp = cross(y_tcp, z_tcp)
   ```
   `det([x y z])` must be `+1` to within `1e-9`.
5. Position:
   ```
   z_top    = max z over the cluster
   z_grasp  = z_top - grasp_depth_m
   plane_z  = height of the plane under (c_x, c_y)
   z_grasp  = max(z_grasp, plane_z + min_height_above_plane_m)
   p_tcp    = (c_x, c_y, z_grasp)
   ```

## S6. Inverse kinematics

**In:** `T_base_tcp`, the chain from `assets/franka/panda_chain.json`.
**Out:** `q` (7), `converged`, `iterations`.

Damped least squares with a null-space pull toward `q_neutral`. Seeded from
`q_neutral` **every frame**, not from the previous solution, so per-frame cost
does not depend on tracking history and every frame is an independent sample.

```
q = q_neutral
for it in [0, max_iterations):
    T   = FK(q)                                  # 7 joints, then flange_to_tcp
    e_p = p_des - p_cur
    e_r = 0.5 * sum_i cross(R_cur[:, i], R_des[:, i])     # i = 0, 1, 2
    if |e_p| < position_tolerance_m and |e_r| < orientation_tolerance_rad:
        converged = true; break
    J   = geometric Jacobian, 6 x 7
          J_v[:, i] = cross(z_i, p_tcp - o_i)
          J_w[:, i] = z_i                        # z_i, o_i in base frame
    dq  = J^T @ solve(J @ J^T + damping^2 * I6, [e_p; e_r])
    N   = I7 - J^T @ solve(J @ J^T + damping^2 * I6, J)   # null-space projector
    dq += nullspace_gain * (N @ (q_neutral - q))
    dq  = clamp(dq, -max_step_rad, +max_step_rad)         # elementwise
    q   = clamp(q + dq, joint_lower, joint_upper)
```

FK uses Rodrigues' formula on each joint's axis; every Panda arm joint is
revolute about its child frame's z, but nothing may assume that.

## S7. Trajectory

**In:** `q_neutral` (start), `q` (goal), joint velocity limits.
**Out:** `waypoints` points, each with position, velocity and acceleration for
7 joints plus `time_from_start`.

```
duration = max_i |q_i - q_start_i| / (velocity_fraction * v_max_i)
duration = clamp(duration, min_duration_s, max_duration_s)

for k in [0, waypoints):
    s   = k / (waypoints - 1)
    h   =  10 s^3 - 15 s^4 +  6 s^5        # quintic, zero vel and acc at both ends
    hd  = (30 s^2 - 60 s^3 + 30 s^4) / duration
    hdd = (60 s   - 180 s^2 + 120 s^3) / duration^2
    position[k]     = q_start + h   * (q_goal - q_start)
    velocity[k]     =           hd  * (q_goal - q_start)
    acceleration[k] =           hdd * (q_goal - q_start)
    time[k]         = s * duration
```

---

## What each stage is measured against

`total = S0 + ... + S7`, measured with `CLOCK_MONOTONIC` in C++ and
`time.perf_counter_ns()` in Python, both of which read the same clock source.
Timer overhead is calibrated and reported so it can be subtracted; see
[METHOD.md](METHOD.md).

A frame where `plane_found`, `graspable` or `converged` is false still produces
a latency sample, because bailing out early is a legitimate outcome with a
legitimate cost, but it is excluded from the output-equivalence comparison
beyond the point where it bailed, and the bail-out rate is reported alongside
the percentiles.
