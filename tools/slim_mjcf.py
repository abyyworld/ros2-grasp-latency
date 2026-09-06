#!/usr/bin/env python3
"""Derive a mesh-slim MJCF from the MuJoCo Menagerie Panda model.

Upstream `panda.xml` references ~33 MB of high-poly visual `.obj` meshes. This
repository uses the MJCF only as a second, independent statement of the Panda's
kinematics (see tests/test_urdf_mjcf_agree.py), so the visual layer is dropped
to keep the vendored asset committable. Body frames, joints, joint limits and
collision geometry are left exactly as upstream wrote them.

Note that a few meshes serve double duty -- `finger_0.obj` is declared as a
visual asset but also referenced by a collision geom -- so assets are retained
by reachability from the surviving geoms rather than by naming convention.

Usage: slim_mjcf.py <src.xml> <dst.xml> <src_assets_dir> <dst_assets_dir>
"""
import re
import shutil
import sys
from pathlib import Path

BANNER = ("  <!-- Visual-only meshes stripped by tools/slim_mjcf.py. Kinematics,"
          " joint limits\n       and collision geometry are unmodified from upstream. -->")


def slim(src_xml: Path, dst_xml: Path, src_assets: Path, dst_assets: Path) -> list[str]:
    text = src_xml.read_text()

    # 1. Drop every geom tagged class="visual". Attribute order varies upstream,
    #    so key off the class rather than a fixed attribute sequence.
    text = re.sub(r'^[ \t]*<geom (?![^>]*<)[^>]*class="visual"[^>]*/>\n', "", text,
                  flags=re.MULTILINE)

    # 2. Whatever meshes the surviving geoms still reference must be retained.
    #    A <mesh> without an explicit name is addressed by its file stem.
    live = set(re.findall(r'<geom [^>]*mesh="([^"]+)"', text))

    kept_files: list[str] = []

    def keep(match: re.Match) -> str:
        decl = match.group(0)
        file_attr = re.search(r'file="([^"]+)"', decl).group(1)
        named = re.search(r'name="([^"]+)"', decl)
        name = named.group(1) if named else Path(file_attr).stem
        if name not in live:
            return ""
        kept_files.append(file_attr)
        return decl

    text = re.sub(r'^[ \t]*<mesh [^>]*/>\n', keep, text, flags=re.MULTILINE)
    text = re.sub(r"\n[ \t]*<!-- Visual meshes -->\n+", "\n", text)
    text = text.replace('<mujoco model="panda">', '<mujoco model="panda">\n' + BANNER, 1)

    dst_assets.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(kept_files)):
        shutil.copy2(src_assets / name, dst_assets / name)
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    dst_xml.write_text(text)
    return sorted(set(kept_files))


if __name__ == "__main__":
    src_xml, dst_xml, src_assets, dst_assets = (Path(a) for a in sys.argv[1:5])
    kept = slim(src_xml, dst_xml, src_assets, dst_assets)
    print(f"kept {len(kept)} meshes: {', '.join(kept)}")
