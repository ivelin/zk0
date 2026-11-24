#!/usr/bin/env python3
"""Spec-only (Tier C) FULL gate-suite generator - twin-suite v2,
Amendment 3 (2026-11-23, founder ruling: spec-only is THE scene path).

DERIVATION: this file is generate_spec_only.py (the frozen Tier C
demonstration-set generator, SHA-256 c96fcec4... post-hygiene) copied
and extended for the full gate denominators. Every change vs the
frozen original is listed here and marked FULL: in the code:
  FULL-1 SCENES_PER_TASK 4 -> 20 (Amendment 3: 20 scenes per task).
  FULL-2 Master seed comes from specs/specs.json in THIS folder,
         whose master_seed is 20261127 (a new stream; the demo set's
         20261126 stream and outputs are untouched).
  FULL-3 Envelope-corner variants k=5..12 for the four precision
         tasks, per the FROZEN envelope corners
         (twin-suite-v2-docs/envelope-corners-v2.json, pre-reg s.3.3):
         pose +x/-x/+y/-y at the per-task sigma, friction_low /
         friction_high (absolute sliding values from the corner
         tables, incl. the global floor/mat/wall table; torsional and
         rolling components stay nominal per the corner file's note),
         mass_low / mass_high (corner mass ranges; block cube low is
         0.005 kg, the OI-2 foam corner, NOT the 0.010 sampled-range
         low). cable_clip has no frozen corners (its gate is the
         v1.1 suite) and gets k=0..4 only. The gripper friction corner
         is NOT expressible per-scene (the arm is a shared include);
         recorded as a limitation in the manifest.
  FULL-4 ARM_INCLUDE defaults to the canonical DGX twin-v2 arm path
         string (the same string the demo set carries), so generated
         files are machine-independent; a runner on another machine
         stages copies with the include rewritten locally. Override
         with env SPEC_ARM_INCLUDE for local testing.
  FULL-5 Manifest carries per-variant "kind" labels and the corner
         provenance; per-scene seed derivation is the F1 pattern with
         SCENES_PER_TASK=20 (scene_index 0..19).

Everything else - placement sampling, variant draw machinery, scene
construction, criteria targets - is the frozen generator's code,
unchanged. No episode, pose-file, split, or recording data is read:
the only inputs are specs/specs.json (in this folder), the audited
parameter values frozen into it, the frozen envelope-corner values
frozen into this file from envelope-corners-v2.json, and the padded
arm include path string.

Draw order (deterministic): tasks alphabetical; scene index 0..19;
per scene the placement draws, then sampled variants k=1..4 (jitter,
mass, friction multiplier). Corner variants are deterministic (no RNG
draws). v0 is nominal physics at the sampled placement.

Usage: python3 generate_spec_only_full.py [--out DIR]   (CPU-only)
"""
import argparse, hashlib, json, math, os, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
# FULL-4: canonical arm include string (machine-independent output).
CANONICAL_ARM = "/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/scenes-v2/arm/so101_new_calib_padded.xml"
ARM_INCLUDE = os.environ.get("SPEC_ARM_INCLUDE", CANONICAL_ARM)
SPECS = json.loads((HERE / "specs" / "specs.json").read_text())
MASTER_SEED = SPECS["master_seed"]
TASKS = ["block_stack", "cable_clip", "eye_drops_to_basket", "ring_insert", "tool_insert"]
SCENES_PER_TASK = 20  # FULL-1
K = 5  # sampled variants per scene: k=0 nominal + k=1..4 sampled (corners are extra, FULL-3)
SIGMA = {"block_stack": 0.025, "ring_insert": 0.025, "eye_drops_to_basket": 0.035,
         "tool_insert": 0.035, "cable_clip": 0.020}  # cable: declared ESTIMATE, gap G-C1

# FULL-3: frozen envelope corners (envelope-corners-v2.json). Absolute
# SLIDING friction values (nominal, low, high); torsional/rolling stay
# nominal. Global surfaces apply to common_xml geoms.
GLOBAL_FRIC = {"floor": (0.8, 0.5, 0.96), "mat": (0.9, 0.7, 1.08), "walls": (0.5, 0.4, 0.6)}
TASK_FRIC = {
    "block_stack": {"cube": (0.6, 0.48, 0.72)},
    "eye_drops_to_basket": {"carton": (0.5, 0.4, 0.6), "basket": (0.4, 0.32, 0.48),
                            "platform": (0.4, 0.32, 0.48)},
    "ring_insert": {"ring": (0.5, 0.4, 0.6), "board": (0.5, 0.4, 0.6), "peg": (0.4, 0.32, 0.48)},
    "tool_insert": {"tool": (0.4, 0.3, 0.5), "tray": (0.6, 0.48, 0.72)},
    "cable_clip": {},
}
# Corner masses (low, high) per manipulated body, from the corner file.
MASS_CORNERS = {
    "block_stack": {"cube_a": (0.005, 0.035), "cube_b": (0.005, 0.035)},
    "eye_drops_to_basket": {"carton": (0.02, 0.045)},
    "ring_insert": {"ring": (0.004, 0.015)},
    "tool_insert": {"tool": (0.05, 0.1)},
    "cable_clip": {},
}
FRIC_OVR = None  # set per friction-corner build: {class: absolute sliding value}

def fmt(v):
    if isinstance(v, (list, tuple)):
        return " ".join(fmt(x) for x in v)
    return f"{v:.6g}"

def yaw_quat(deg):
    r = math.radians(deg) / 2
    return [math.cos(r), 0.0, 0.0, math.sin(r)]

def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return [aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw]

def qinv(q):
    return [q[0], -q[1], -q[2], -q[3]]

def qrot(q, v):
    r = qmul(qmul(q, [0.0, *v]), qinv(q))
    return r[1:]

def quat_from_x_dir(d):
    import math as m
    x = [c / (m.sqrt(sum(c*c for c in d)) or 1.0) for c in d]
    up = [0.0, 0.0, 1.0]
    if abs(sum(a*b for a, b in zip(x, up))) > 0.99:
        up = [0.0, 1.0, 0.0]
    y = [up[1]*x[2]-up[2]*x[1], up[2]*x[0]-up[0]*x[2], up[0]*x[1]-up[1]*x[0]]
    ny = math.sqrt(sum(c*c for c in y)) or 1.0
    y = [c/ny for c in y]
    z = [x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]]
    tr = x[0]+y[1]+z[2]
    if tr > 0:
        s = math.sqrt(tr+1.0)*2
        return [0.25*s, (z[1]-y[2])/s, (x[2]-z[0])/s, (y[0]-x[1])/s]
    i = max(range(3), key=lambda i: [x[0], y[1], z[2]][i])
    if i == 0:
        s = math.sqrt(1.0+x[0]-y[1]-z[2])*2
        return [(z[1]-y[2])/s, 0.25*s, (x[1]+y[0])/s, (x[2]+z[0])/s]
    if i == 1:
        s = math.sqrt(1.0+y[1]-x[0]-z[2])*2
        return [(x[2]-z[0])/s, (x[1]+y[0])/s, 0.25*s, (y[2]+z[1])/s]
    s = math.sqrt(1.0+z[2]-x[0]-y[1])*2
    return [(y[0]-x[1])/s, (x[2]+z[0])/s, (y[2]+z[1])/s, 0.25*s]

W = SPECS["workspace"]
def _gf(cls):
    # FULL-3: global-surface sliding friction under a friction corner.
    return FRIC_OVR.get(cls) if FRIC_OVR else None

def common_xml():
    mx, my = W["mat_center_xy_m"]; sx, sy, st = W["mat_size_m"]
    _ff, _fm, _fw = _gf("floor"), _gf("mat"), _gf("walls")
    floor_f = f"{_ff} 0.005 0.0001" if _ff is not None else "0.8 0.005 0.0001"
    mat_f = f"{_fm} 0.005 0.0001" if _fm is not None else "0.9 0.005 0.0001"
    wall_f = f"{_fw} 0.005 0.0001" if _fw is not None else "0.5 0.005 0.0001"
    return f"""
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.001"
      friction="{floor_f}" rgba="0.35 0.30 0.25 1"/>
    <geom name="mat" type="box" pos="{mx} {my} {-st/2}" size="{sx/2} {sy/2} {st/2}"
      friction="{mat_f}" rgba="0.16 0.19 0.24 1"/>
    <geom name="wall_back" type="box" pos="{mx} {my+sy/2+0.005} 0.03" size="{sx/2} 0.005 0.03"
      friction="{wall_f}" rgba="0.76 0.66 0.50 1"/>
    <geom name="wall_front" type="box" pos="{mx} {my-sy/2-0.005} 0.03" size="{sx/2} 0.005 0.03"
      friction="{wall_f}" rgba="0.76 0.66 0.50 1"/>
    <geom name="wall_left" type="box" pos="{mx-sx/2-0.005} {my} 0.03" size="0.005 {sy/2} 0.03"
      friction="{wall_f}" rgba="0.76 0.66 0.50 1"/>
    <geom name="wall_right" type="box" pos="{mx+sx/2+0.005} {my} 0.03" size="0.005 {sy/2} 0.03"
      friction="{wall_f}" rgba="0.76 0.66 0.50 1"/>
    <camera name="topcam" pos="{mx} {my} 0.85" quat="1 0 0 0" fovy="50"/>
    <camera name="frontcam" pos="0.75 -0.55 0.35" xyaxes="0.66 -0.75 0 0.28 0.24 -0.92" fovy="55"/>
"""

def free_body(name, pos, quat, inner):
    return (f'<body name="{name}" pos="{fmt(pos)}" quat="{fmt(quat)}">'
            f'<joint name="{name}_free" type="free" damping="0.01"/>{inner}</body>')

def vfric(base, mult, cls=None):
    # FULL-3: under a friction corner, the class's absolute sliding value
    # from the frozen corner table replaces the multiplied nominal.
    if cls is not None and FRIC_OVR and cls in FRIC_OVR:
        return (FRIC_OVR[cls], base[1], base[2])
    return (round(base[0]*mult, 4), base[1], base[2])

def ffric(triple, cls):
    # FULL-3: fmt() for fixture friction triples under a friction corner.
    if FRIC_OVR and cls in FRIC_OVR:
        return fmt((FRIC_OVR[cls], triple[1], triple[2]))
    return fmt(triple)

# ---------------- cable helpers (v1 code default layout, declared) --------
CABLE_N, CABLE_SEG_LEN, CABLE_R = 24, 0.030, 0.0025
def default_cable_points():
    cx, cy, r = 0.27, -0.07, 0.055
    return [[cx + r*math.cos(2*math.pi*2.0*i/CABLE_N),
             cy + r*math.sin(2*math.pi*2.0*i/CABLE_N), CABLE_R] for i in range(CABLE_N+1)]

def cable_xml(pts, seg_mass, fric_mult, dx=0.0, dy=0.0):
    pts = [[p[0]+dx, p[1]+dy, p[2]] for p in pts]
    frames = [quat_from_x_dir([pts[i+1][j]-pts[i][j] for j in range(3)]) for i in range(CABLE_N)]
    usb_size = (0.025, 0.012, 0.008)
    child = ('<body name="usb" pos="0.03 0 0"><geom type="box" pos="0.0105 0 0" '
             f'size="{fmt([s/2 for s in usb_size])}" mass="0.006" '
             'friction="0.5 0.005 0.0001" rgba="0.2 0.2 0.2 1"/></body>')
    for i in reversed(range(CABLE_N)):
        if i == 0:
            pos, quat = pts[0], frames[0]
            # Gap G-C4 (specs/specs.json): seg_0 free-joint damping 0.05 is
            # [DEFAULT]/untraced in the audited table; traced 2026-11-23 to the
            # v1/v1.1 lineage: zk0-study001-twin-20261122/code/build_scenes.py,
            # cable_xml_build, hardcoded damping="0.05" for cable_seg_0_free,
            # while the ball joints use twin_params.CABLE_JOINT_DAMPING = 0.2
            # (twin_params.py). Discrepancy declared as gap G-C4; value NOT changed.
            joint = '<joint name="cable_seg_0_free" type="free" damping="0.05"/>'
        else:
            quat = qmul(qinv(frames[i-1]), frames[i])
            pos = qrot(qinv(frames[i-1]), [pts[i][j]-pts[i-1][j] for j in range(3)])
            joint = f'<joint name="cable_seg_{i}_ball" type="ball" damping="0.2"/>'
        fr = vfric((0.8, 0.005, 0.001), fric_mult)
        geom = (f'<geom type="capsule" fromto="0 0 0 {CABLE_SEG_LEN} 0 0" size="{CABLE_R}" '
                f'mass="{seg_mass:.6g}" friction="{fmt(fr)}" rgba="0.05 0.05 0.05 1"/>')
        child = (f'<body name="cable_seg_{i}" pos="{fmt(pos)}" quat="{fmt(quat)}">'
                 + joint + geom + child + '</body>')
    return child

# ---------------- per-task builders ---------------------------------------
def build_scene(task, place, var):
    """place: dict of sampled placements; var: {jitter,mass,friction_mult}."""
    S = SPECS["tasks"][task]
    bodies, welds, meta = [], [], {}
    j = var.get("jitter", {}); m = var.get("mass", {}); fm = var.get("friction_mult", {})
    def jp(name, pos):
        d = j.get(name)
        return [pos[0]+d[0], pos[1]+d[1], pos[2]] if d else list(pos)
    if task == "block_stack":
        e = S["objects"]["cube_a"]["edge_m"]
        for nm in ("cube_a", "cube_b"):
            o = S["objects"][nm]; p = jp(nm, place[nm]["pos"])
            q = yaw_quat(place[nm]["yaw_deg"] + (j.get(nm+"_yaw", 0.0)))
            inner = (f'<geom name="{nm}_geom" type="box" size="{e/2} {e/2} {e/2}" '
                     f'mass="{m.get(nm, o["mass_kg"])}" friction="{fmt(vfric(o["friction"], fm.get(nm,1.0), "cube"))}" rgba="{o["rgba"]}"/>')
            bodies.append(free_body(nm, p, q, inner)); welds.append(nm)
        meta["target"] = {"moved_cube": "cube_a", "base_cube": "cube_b",
                          "goal": "cube_a resting on cube_b per registered criterion"}
        meta["ground_truth_initial"] = {nm: {"pos": jp(nm, place[nm]["pos"]),
                                             "yaw_deg": place[nm]["yaw_deg"]} for nm in ("cube_a", "cube_b")}
    elif task == "eye_drops_to_basket":
        pf = S["fixtures"]["platform"]["pos_xy_m"]; bk = S["fixtures"]["basket"]["pos_xy_m"]
        tw, td, tt = S["fixtures"]["platform"]["top_size_m"]; h = S["fixtures"]["platform"]["height_m"]
        pfric = ffric(S["fixtures"]["platform"]["friction"], "platform"); bfric = ffric(S["fixtures"]["basket"]["friction"], "basket")
        bo = S["fixtures"]["basket"]; bw, bd = bo["outer_m"]; wh, wt = bo["wall_h_m"], bo["wall_t_m"]
        # Gap G-E5 (specs/specs.json): platform_web (half-extents 0.01 x td/4 x h/2)
        # and platform_base (half-extents tw/2 x td/2 x 0.002, centre z=0.002) below
        # are [DEFAULT] inherited v1.1 construction, traced to
        # zk0-study001-twin-20261122/code/build_scenes.py (platform_web/platform_base
        # geoms); not separately parameterized in the audited table (audit E4 note).
        bodies.append(f"""
    <body name="platform" pos="{pf[0]} {pf[1]} 0">
      <geom name="platform_top" type="box" pos="0 0 {h-tt/2}" size="{tw/2} {td/2} {tt/2}" friction="{pfric}" rgba="0.92 0.92 0.92 1"/>
      <geom name="platform_web" type="box" pos="0 0 {h/2}" size="0.01 {td/4} {h/2}" friction="{pfric}" rgba="0.92 0.92 0.92 1"/>
      <geom name="platform_base" type="box" pos="0 0 0.002" size="{tw/2} {td/2} 0.002" friction="{pfric}" rgba="0.92 0.92 0.92 1"/>
    </body>
    <body name="basket" pos="{bk[0]} {bk[1]} 0">
      <geom name="basket_floor" type="box" pos="0 0 {wt/2}" size="{bw/2} {bd/2} {wt/2}" friction="{bfric}" rgba="0.85 0.90 0.95 0.5"/>
      <geom name="basket_w1" type="box" pos="0 {bd/2-wt/2} {wh/2}" size="{bw/2} {wt/2} {wh/2}" friction="{bfric}" rgba="0.85 0.90 0.95 0.5"/>
      <geom name="basket_w2" type="box" pos="0 {-bd/2+wt/2} {wh/2}" size="{bw/2} {wt/2} {wh/2}" friction="{bfric}" rgba="0.85 0.90 0.95 0.5"/>
      <geom name="basket_w3" type="box" pos="{bw/2-wt/2} 0 {wh/2}" size="{wt/2} {bd/2} {wh/2}" friction="{bfric}" rgba="0.85 0.90 0.95 0.5"/>
      <geom name="basket_w4" type="box" pos="{-bw/2+wt/2} 0 {wh/2}" size="{wt/2} {bd/2} {wh/2}" friction="{bfric}" rgba="0.85 0.90 0.95 0.5"/>
    </body>""")
        o = S["objects"]["carton"]; sx, sy, sz = o["size_m"]
        p = jp("carton", place["carton"]["pos"]); q = yaw_quat(place["carton"]["yaw_deg"])
        inner = (f'<geom name="carton_geom" type="box" size="{sx/2} {sy/2} {sz/2}" '
                 f'mass="{m.get("carton", o["mass_kg"])}" friction="{fmt(vfric(o["friction"], fm.get("carton",1.0), "carton"))}" rgba="0.25 0.45 0.85 1"/>')
        bodies.append(free_body("carton", p, q, inner)); welds.append("carton")
        meta["target"] = {"goal": "carton final resting support is the basket",
                          "basket_centre_xy": bk, "basket_interior_m": [bw-2*wt, bd-2*wt]}
        meta["ground_truth_initial"] = {"carton": {"pos": p, "yaw_deg": place["carton"]["yaw_deg"]},
                                        "platform_pos_xy": pf, "basket_pos_xy": bk}
    elif task == "ring_insert":
        bp = S["fixtures"]["board_pos_xy_m"]; bt = S["objects"]["board"]["height_m"]
        bodies.append(f"""
    <body name="board" pos="{bp[0]} {bp[1]} 0">
      <geom name="board_geom" type="cylinder" pos="0 0 {bt/2}" size="{S['objects']['board']['radius_m']} {bt/2}" friction="{ffric(S['objects']['board']['friction'], "board")}" rgba="0.92 0.87 0.72 1"/>
    </body>
    <body name="peg" pos="{bp[0]} {bp[1]} 0">
      <geom name="peg_geom" type="cylinder" pos="0 0 {bt + S['objects']['peg']['height_above_board_m']/2}" size="{S['objects']['peg']['radius_m']} {S['objects']['peg']['height_above_board_m']/2}" friction="{ffric(S['objects']['peg']['friction'], "peg")}" rgba="0.80 0.65 0.42 1"/>
    </body>""")
        o = S["objects"]["ring"]; n = o["segments"]
        rmid = (o["r_out_m"] + o["r_in_m"]) / 2; tube = min((o["r_out_m"]-o["r_in_m"])/2, o["thick_m"]/2)
        per = m.get("ring", o["mass_kg"]) / n; geoms = []
        for k in range(n):
            a0 = 2*math.pi*k/n; a1 = 2*math.pi*(k+1)/n
            p0 = (rmid*math.cos(a0), rmid*math.sin(a0), 0); p1 = (rmid*math.cos(a1), rmid*math.sin(a1), 0)
            geoms.append(f'<geom type="capsule" fromto="{fmt(p0)} {fmt(p1)}" size="{tube}" mass="{per:.6g}" '
                         f'friction="{fmt(vfric(o["friction"], fm.get("ring",1.0), "ring"))}" rgba="{o["rgba"]}"/>')
        p = jp("ring", place["ring"]["pos"])
        bodies.append(free_body("ring", p, [1, 0, 0, 0], "".join(geoms))); welds.append("ring")
        meta["target"] = {"goal": "ring seated on peg", "peg_xy": bp, "board_top_z": bt}
        meta["ground_truth_initial"] = {"ring": {"pos": p, "quat": [1, 0, 0, 0]}, "board_pos_xy": bp, "peg_xy": bp}
    elif task == "tool_insert":
        tp = S["fixtures"]["tray_pos_xy_m"]; sp = S["fixtures"]["slot_pos_xy_m"]
        tw, td, th = S["objects"]["tray"]["size_m"]; sw, sd = S["objects"]["slot"]["size_m"]
        depth = S["objects"]["slot"]["depth_m"]; tfric = ffric(S["objects"]["tray"]["friction"], "tray")
        lx, ly = sp[0]-tp[0], sp[1]-tp[1]
        rx0, rx1 = max(lx-sw/2, -tw/2), min(lx+sw/2, tw/2); ry0, ry1 = max(ly-sd/2, -td/2), min(ly+sd/2, td/2)
        rims = []
        def _rim(name, cx, cy, hx, hy):
            if hx > 1e-9 and hy > 1e-9:
                rims.append(f'<geom name="{name}" type="box" pos="{cx} {cy} {th-depth/2}" size="{hx} {hy} {depth/2}" friction="{tfric}" rgba="0.10 0.10 0.10 1"/>')
        _rim("tray_rim_w", (-tw/2+rx0)/2, 0.0, (rx0+tw/2)/2, td/2)
        _rim("tray_rim_e", (rx1+tw/2)/2, 0.0, (tw/2-rx1)/2, td/2)
        _rim("tray_rim_s", (rx0+rx1)/2, (-td/2+ry0)/2, (rx1-rx0)/2, (ry0+td/2)/2)
        _rim("tray_rim_n", (rx0+rx1)/2, (ry1+td/2)/2, (rx1-rx0)/2, (td/2-ry1)/2)
        bodies.append(f"""
    <body name="tray" pos="{tp[0]} {tp[1]} 0">
      <geom name="tray_geom" type="box" pos="0 0 {(th-depth)/2}" size="{tw/2} {td/2} {(th-depth)/2}" friction="{tfric}" rgba="0.10 0.10 0.10 1"/>
      <geom name="slot_floor" type="box" pos="{(rx0+rx1)/2} {(ry0+ry1)/2} {th-depth+0.0005}" size="{(rx1-rx0)/2} {(ry1-ry0)/2} 0.0005" friction="{tfric}" rgba="0.45 0.08 0.08 1"/>
      {''.join(rims)}
    </body>""")
        o = S["objects"]["tool"]; p = jp("tool", place["tool"]["pos"]); q = yaw_quat(place["tool"]["yaw_deg"])
        inner = (f'<geom name="tool_geom" type="capsule" fromto="{-o["len_m"]/2} 0 0 {o["len_m"]/2} 0 0" '
                 f'size="{o["radius_m"]}" mass="{m.get("tool", o["mass_kg"])}" friction="{fmt(vfric(o["friction"], fm.get("tool",1.0), "tool"))}" rgba="0.75 0.75 0.78 1"/>')
        bodies.append(free_body("tool", p, q, inner)); welds.append("tool")
        meta["target"] = {"goal": "tool centre within slot footprint, axis within 20 deg of slot axis, resting in tray",
                          "slot_centre_xy": sp, "slot_size_m": [sw, sd], "slot_yaw_deg": 0.0,
                          "slot_floor_top_z": th-depth+0.001}
        meta["ground_truth_initial"] = {"tool": {"pos": p, "yaw_deg": place["tool"]["yaw_deg"]},
                                        "tray_pos_xy": tp, "slot_pos_xy": sp}
    elif task == "cable_clip":
        cb = S["fixtures"]["clip_block_pos_xy_m"]; bw, bd, bh = S["objects"]["clip_block"]["size_m"]
        cw = S["objects"]["clip_jaws"]["channel_width_m"]; jf = fmt(S["objects"]["clip_jaws"]["friction"])
        bodies.append(f"""
    <body name="clip_block" pos="{cb[0]} {cb[1]} 0" quat="1 0 0 0">
      <geom name="block_geom" type="box" pos="0 0 {bh/2}" size="{bw/2} {bd/2} {bh/2}" friction="{fmt(S['objects']['clip_block']['friction'])}" rgba="0.93 0.93 0.93 1"/>
      <geom name="clip_jaw_a" type="box" pos="0 {cw/2+0.004} {bh+0.006}" size="0.012 0.004 0.006" friction="{jf}" rgba="0.08 0.08 0.08 1"/>
      <geom name="clip_jaw_b" type="box" pos="0 {-cw/2-0.004} {bh+0.006}" size="0.012 0.004 0.006" friction="{jf}" rgba="0.08 0.08 0.08 1"/>
    </body>""")
        total = m.get("cable_total", S["objects"]["cable"]["total_mass_kg"])
        d = j.get("cable", [0.0, 0.0])
        bodies.append(cable_xml(default_cable_points(), total/CABLE_N, fm.get("cable", 1.0), d[0], d[1]))
        welds.extend([f"cable_seg_{i}" for i in range(CABLE_N)])
        meta["target"] = {"goal": "a cable segment seated in the clip channel",
                          "clip_block_pos_xy": cb, "clip_channel_centre": [cb[0], cb[1], bh+0.006],
                          "channel_width_m": cw}
        meta["ground_truth_initial"] = {"clip_block": {"pos": [cb[0], cb[1], 0.0], "yaw_deg": 0.0},
                                        "cable_coil_centre_xy": [0.27+d[0], -0.07+d[1]],
                                        "cable_points": [[round(x, 6) for x in p] for p in
                                                         [[q[0]+d[0], q[1]+d[1], q[2]] for q in default_cable_points()]]}
    Kw = 2 if task == "cable_clip" else 4
    weld_xml = "".join(f'<weld name="weld_{w}_{k}" body1="gripper" body2="{w}" active="false" '
                       f'solref="0.005 1" solimp="0.95 0.99 0.001"/>' for w in welds for k in range(Kw))
    xml = f"""<!-- twin-suite-v2 SPEC-ONLY (Tier C, Amendment 2 A2.1) task={task} master_seed={MASTER_SEED} - generated from task spec + audited parameter table; NO episode data -->
<mujoco model="spec_{task}">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option timestep="0.0005" gravity="0 0 -9.81"/>
  <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="145" elevation="-22"/></visual>
  <include file="{ARM_INCLUDE}"/>
  <worldbody>
    <light pos="0.3 -0.3 0.9" dir="-0.2 0.2 -1" diffuse="0.85 0.85 0.85"/>
    {common_xml()}
    {''.join(bodies)}
  </worldbody>
  <equality>{weld_xml}</equality>
</mujoco>
"""
    return xml, meta

# ---------------- sampling -------------------------------------------------
def sample_placement(rng, task):
    S = SPECS["tasks"][task]; P = S["placement"]
    if task == "block_stack":
        e = S["objects"]["cube_a"]["edge_m"]
        while True:
            pts = {}
            for nm in ("cube_a", "cube_b"):
                pts[nm] = {"pos": [round(rng.uniform(*P["region_x_m"]), 6),
                                   round(rng.uniform(*P["region_y_m"]), 6), e/2],
                           "yaw_deg": round(rng.uniform(*P["yaw_deg"]), 3)}
            dx = pts["cube_a"]["pos"][0]-pts["cube_b"]["pos"][0]
            dy = pts["cube_a"]["pos"][1]-pts["cube_b"]["pos"][1]
            if math.hypot(dx, dy) >= P["min_separation_m"]:
                return pts
    if task == "eye_drops_to_basket":
        pf = S["fixtures"]["platform"]["pos_xy_m"]; h = S["fixtures"]["platform"]["height_m"]
        sy = S["objects"]["carton"]["size_m"][1]
        return {"carton": {"pos": [round(pf[0]+rng.uniform(*P["offset_x_m"]), 6),
                                   round(pf[1]+rng.uniform(*P["offset_y_m"]), 6), h+sy/2],
                           "yaw_deg": round(rng.uniform(*P["yaw_deg"]), 3)}}
    if task == "ring_insert":
        bp = S["fixtures"]["board_pos_xy_m"]; bt = S["objects"]["board"]["height_m"]
        excl = S["objects"]["board"]["radius_m"] + 0.03
        while True:
            x = rng.uniform(*P["region_x_m"]); y = rng.uniform(*P["region_y_m"])
            if math.hypot(x-bp[0], y-bp[1]) >= excl:
                return {"ring": {"pos": [round(x, 6), round(y, 6), bt*0 + S["objects"]["ring"]["thick_m"]/2]}}
    if task == "tool_insert":
        tp = S["fixtures"]["tray_pos_xy_m"]; tw, td, _ = S["objects"]["tray"]["size_m"]
        while True:
            x = rng.uniform(*P["region_x_m"]); y = rng.uniform(*P["region_y_m"])
            if abs(x-tp[0]) < tw/2 and abs(y-tp[1]) < td/2:
                continue
            return {"tool": {"pos": [round(x, 6), round(y, 6), S["objects"]["tool"]["radius_m"]],
                             "yaw_deg": round(rng.uniform(*P["yaw_deg"]), 3)}}
    if task == "cable_clip":
        return {"cable": {"coil_centre_xy": [0.27, -0.07]}}
    raise ValueError(task)

MANIP = {"block_stack": ["cube_a", "cube_b"], "eye_drops_to_basket": ["carton"],
         "ring_insert": ["ring"], "tool_insert": ["tool"], "cable_clip": ["cable"]}

def make_variant(rng, task, k):
    if k == 0:
        return {"jitter": {}, "mass": {}, "friction_mult": {}}
    sig = SIGMA[task]; jit, mass, fm = {}, {}, {}
    mr = SPECS["variant_protocol"]["mass_ranges_kg"]
    for b in MANIP[task]:
        jit[b] = [round(rng.uniform(-sig, sig), 6), round(rng.uniform(-sig, sig), 6)]
        key = "cable_total" if b == "cable" else ("cube" if b.startswith("cube") else b)
        mass["cable_total" if b == "cable" else b] = round(rng.uniform(*mr[key]), 5)
        fm[b] = round(rng.uniform(0.8, 1.2), 4)
    return {"jitter": jit, "mass": mass, "friction_mult": fm}

def main():
    global FRIC_OVR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "scenes"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(MASTER_SEED)
    manifest = {"schema": "twin-suite-v2/spec-only-suite/1", "suite": "twin-suite-v2 spec-only FULL gate suite (Amendment 3, 2026-11-23)",
                "tier": "C (spec only, Amendment 2 A2.1; Amendment 3 scene path)", "master_seed": MASTER_SEED,
                "generator": "generate_spec_only_full.py (derived from the frozen generate_spec_only.py; changes FULL-1..FULL-5 in its docstring)",
                "specs": "specs/specs.json + specs/*.md",
                "episode_data_used": False,
                "arm_include": ARM_INCLUDE,
                "per_scene_seed_derivation": "per_scene_seed = master_seed + global_scene_index, where global_scene_index = task_order_index * 20 + scene_index (task order alphabetical as in TASKS; scene_index 0..19). Recorded identifier only; generation uses the single Random(master_seed) stream.",
                "variant_kinds": {"0": "nominal", "1-4": "sampled (jitter/mass/friction draws, frozen variant protocol)",
                                  "5-8": "envelope corner pose +x/-x/+y/-y at per-task sigma (frozen envelope-corners-v2.json)",
                                  "9": "envelope corner friction_low (absolute sliding values, corner tables; depends on [DEFAULT] parameters at their corners - pre-reg s.3.1 flag)",
                                  "10": "envelope corner friction_high (same flag)",
                                  "11": "envelope corner mass_low (corner mass ranges; block cube low 0.005 kg is the OI-2 foam case)",
                                  "12": "envelope corner mass_high"},
                "friction_corner_note": "Friction corners set the sliding component of every generator-emitted geom (objects, fixtures, floor/mat/walls) to the frozen corner tables' absolute low/high values; torsional/rolling stay nominal (corner file note). The gripper friction corner (global table 1.0 -> 0.8/1.2) is NOT expressible per-scene because the arm is a shared include; gripper friction stays nominal in all variants. Engine corner (Drake) is a rollout-time engine choice, not a scene file.",
                "note": "Solvability validation (A2.2 executed pass) is NOT part of generation: it belongs to the reference-controller workstream. Validated denominators are updated in the solvability register as executed passes land; scenes without an executed pass are excluded from denominators and logged.",
                "tasks": {}}
    for ti, task in enumerate(TASKS):
        entries = []
        for si in range(SCENES_PER_TASK):
            place = sample_placement(rng, task)
            # FULL-3: variant plan - k=0..4 sampled (frozen protocol),
            # then deterministic envelope corners for precision tasks.
            plan = []
            for k in range(K):
                plan.append((k, "nominal" if k == 0 else "sampled",
                             make_variant(rng, task, k), None))
            if MASS_CORNERS.get(task):
                sig = SIGMA[task]
                for i, (kind, dx, dy) in enumerate((("pose_+x", sig, 0.0), ("pose_-x", -sig, 0.0),
                                                    ("pose_+y", 0.0, sig), ("pose_-y", 0.0, -sig))):
                    jit = {b: [dx, dy] for b in MANIP[task]}
                    plan.append((5 + i, "corner:" + kind,
                                 {"jitter": jit, "mass": {}, "friction_mult": {}}, None))
                plan.append((9, "corner:friction_low",
                             {"jitter": {}, "mass": {}, "friction_mult": {}}, "low"))
                plan.append((10, "corner:friction_high",
                             {"jitter": {}, "mass": {}, "friction_mult": {}}, "high"))
                for i, side in enumerate(("low", "high")):
                    mv = {b: MASS_CORNERS[task][b][0 if side == "low" else 1]
                          for b in MASS_CORNERS[task]}
                    plan.append((11 + i, "corner:mass_" + side,
                                 {"jitter": {}, "mass": mv, "friction_mult": {}}, None))
            variants = []
            for k, kind, var, fric_side in plan:
                if fric_side is None:
                    FRIC_OVR = None
                else:
                    idx = 1 if fric_side == "low" else 2
                    FRIC_OVR = {c: v[idx] for c, v in GLOBAL_FRIC.items()}
                    FRIC_OVR.update({c: v[idx] for c, v in TASK_FRIC[task].items()})
                xml, meta = build_scene(task, place, var)
                FRIC_OVR = None
                name = f"{task}__spec{si:02d}_v{k}.xml"
                (out / name).write_text(xml)
                variants.append({"k": k, "kind": kind, "scene": name, "draws": var,
                                 "sha256": hashlib.sha256(xml.encode()).hexdigest()})
            entries.append({"scene_id": f"{task}__spec{si:02d}",
                            "seed": MASTER_SEED + ti * SCENES_PER_TASK + si,
                            "sampled_placement": place,
                            "parameter_labels": "see specs/specs.json tasks." + task,
                            "target_specification": variants and json.loads(
                                json.dumps(build_scene(task, place, {"jitter": {}, "mass": {}, "friction_mult": {}})[1]["target"])),
                            "ground_truth_initial_state": build_scene(task, place, {"jitter": {}, "mass": {}, "friction_mult": {}})[1]["ground_truth_initial"],
                            "variants": variants})
        manifest["tasks"][task] = {"denominator_nominal": SCENES_PER_TASK, "denominator_validated": 0,
                                   "scenes": entries}
    (HERE / "suite_manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(len(t["scenes"]) * len(t["scenes"][0]["variants"]) if t["scenes"] else 0
                for t in manifest["tasks"].values())
    total = sum(len(e["variants"]) for t in manifest["tasks"].values() for e in t["scenes"])
    print(f"generated {SCENES_PER_TASK*len(TASKS)} scenes, {total} XMLs total, seed {MASTER_SEED}")

if __name__ == "__main__":
    main()
