# Vendored robot models

Both files describe the **same** robot -- the Franka Emika Panda -- in the two
formats this project claims competence in. `tests/test_urdf_mjcf_agree.py`
checks that claim numerically rather than assuming it.

| File | Upstream | Commit | Licence |
|---|---|---|---|
| `urdf/panda.urdf` | [moveit/moveit_resources](https://github.com/moveit/moveit_resources) `panda_description/urdf/panda.urdf` | `c55b102711fc0aebe80c6952d2ce97c38110abba` | BSD-3-Clause |
| `mjcf/panda.xml` | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) `franka_emika_panda/panda.xml` | `8161bba264d7fa7c99ca301e91e7fb44737676ad` | Apache-2.0 |

## Modification to the MJCF

The upstream MJCF references ~33 MB of high-poly visual `.obj` meshes. Since
this repository uses the MJCF only as an independent statement of the Panda's
kinematics, `tools/slim_mjcf.py` strips the visual layer, leaving 232 KB. Body
frames, joints, joint limits and collision geometry are untouched: forward
kinematics over 200 random configurations agrees with the unmodified model to
`0.000e+00 m`.

To reproduce from upstream:

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie /tmp/mg
python3 tools/slim_mjcf.py /tmp/mg/franka_emika_panda/panda.xml \
    assets/franka/mjcf/panda.xml /tmp/mg/franka_emika_panda/assets \
    assets/franka/mjcf/assets
```

The URDF is vendored byte-for-byte.
