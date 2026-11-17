"""SIM-2 scene: the collection YCB mug and the target bin.

The scene mirrors the certified wall-pinch / force-aware construction
(2026-11-16) exactly where it matters: the mug is the checkpointed
collection asset `ycb-025-mug` (visual mesh + 32 CoACD hulls, cavity
open, mass 0.118 kg MEASURED by YCB), spawned with its origin under
the volumetric CoM (com_z from the package's own params.json), the
collision-floor supplement disc from the same params, and the bin is
the certified wall-pinch bin (centre (0.30, -0.02), interior half
extents 0.058, rim height 0.11). The table of the certified scenes
is replaced by the package harness's floor plane (z=0), as in SIM-1;
the mug/bin geometry and the containment definition are unchanged.

Friction: the drawn pair mu is set on every mug geom. The arm side
of the pair is set by the runner patching the staged arm copy's
gripper friction to the same value (the certified sweep did exactly
this — MuJoCo combines a contact pair by the maximum of the two
sides' priority classes, so both sides must carry the draw).
"""
import json
import math
from pathlib import Path

from . import assets

BIN = (0.30, -0.02, 0.058, 0.058, 0.11)   # cx, cy, hx, hy, rim height
FRIC_TAIL = "0.005 0.0001"


def mug_params() -> dict:
    p = assets.mug_dir() / "params.json"
    if not p.exists():
        raise FileNotFoundError(
            f"staged mug package missing: {p} — run dev/bootstrap.sh")
    return json.loads(p.read_text())


def mug_assets_xml() -> str:
    d = assets.mug_dir()
    hulls = sorted(d.glob("hull_*.obj"))
    if not hulls:
        raise FileNotFoundError(f"no mug hulls staged in {d}")
    out = ['<texture name="mug_tex" type="2d" file="texture_map.png"/>',
           '<material name="mug_mat" texture="mug_tex" texuniform="false"/>',
           '<mesh name="mug_vis" file="visual.obj"/>']
    for i, hp in enumerate(hulls):
        out.append(f'<mesh name="mug_h{i}" file="{hp.name}"/>')
    return "\n    ".join(out)


def mug_body_xml(mug_xy, yaw_rad: float, mu: float) -> str:
    par = mug_params()
    com_z = par["com_mm"][2] / 1000.0
    mass = par["mass_kg"]
    di = " ".join(f"{v:.3e}" for v in par["diaginertia_kgm2"])
    d = assets.mug_dir()
    n_hulls = len(sorted(d.glob("hull_*.obj")))
    qw, qz = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    geoms = []
    for i in range(n_hulls):
        geoms.append(
            f'<geom name="mug_h{i}" type="mesh" mesh="mug_h{i}" '
            f'pos="0 0 {-com_z:.5f}" friction="{mu} {FRIC_TAIL}" '
            f'rgba="0 0 0 0"/>')
    sup = par["collision_floor_supplement"]
    geoms.append(
        f'<geom name="mug_floor" type="cylinder" '
        f'size="{sup["radius_mm"] / 1000.0} 0.002" '
        f'pos="0 0 {0.001 - com_z:.5f}" friction="{mu} {FRIC_TAIL}" '
        f'solref="{sup["solref"]}" solimp="{sup["solimp"]}" '
        f'rgba="0 0 0 0"/>')
    geoms.append(
        f'<geom name="mug_vis" type="mesh" mesh="mug_vis" '
        f'pos="0 0 {-com_z:.5f}" contype="0" conaffinity="0" '
        f'material="mug_mat" rgba="1 1 1 1"/>')
    return f"""
    <body name="mug" pos="{mug_xy[0]} {mug_xy[1]} {com_z}" quat="{qw:.6f} 0 0 {qz:.6f}">
      <freejoint name="mug_free"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{di}"/>
      {"".join(geoms)}
    </body>"""


def contained(final_pos, settled_mm: float) -> dict:
    """Tier 1 containment for SIM-2, same definition as the certified
    scenes: centre inside the bin footprint, below the rim, settled."""
    cx, cy, hx, hy, rim = BIN
    x, y, z = (float(v) for v in final_pos)
    inside = abs(x - cx) < hx and abs(y - cy) < hy
    below = z < rim
    settled_ok = settled_mm <= 2.0
    return {"inside_xy": bool(inside), "below_rim": bool(below),
            "settled": bool(settled_ok),
            "pass": bool(inside and below and settled_ok)}
