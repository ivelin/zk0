"""Study 001 twin-replay: per-episode MuJoCo scene builder (Builder C).

Generates scenes/<task>__ep<id>.xml = arm (vendored SO-101 model, included)
+ tabletop/mat/enclosure + task fixtures + dynamic objects placed at
Builder B's per-episode initial poses (scenes/<task>__ep<id>.json, written by
Builder B) with fixture poses from scenes/fixtures.json when present.

Pose-JSON adapter is schema-tolerant: it accepts object entries under
'initial'/'objects'/'poses' with pos under 'pos'/'position'/'xyz' and
orientation under 'quat'/'quaternion' (wxyz) or 'yaw_deg'/'yaw'/'theta'.
Confidence is read from 'confidence' at top level or per object. When
Builder B's exact schema is known this adapter is narrowed and the change
logged. Episodes without a pose file get a SMOKE scene only if --smoke.
"""
import hashlib, json, math, sys, zlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import twin_params as P

SCENES = P.TWIN / "scenes"

def param_table_hash():
    h = hashlib.sha256()
    h.update(Path(__file__).with_name("twin_params.py").read_bytes())
    h.update((P.TWIN / "docs" / "physics-parameters.md").read_bytes())
    return h.hexdigest()[:16]

# ---------------- FK frame correction (Builder C calibration finding) ----
# Builder B's calib_v2_mat_homography expresses poses in a world frame whose
# axes are rotated/reflected vs the arm FK frame (its mat near-edge was taken
# parallel to world x; in the top camera the near-far direction runs along
# image-vertical and image-right is -y). A robust affine px->FK-world map was
# fitted on 93 grasp/place dwell tie points (FK of recorded gt at gripper
# dwells vs Builder B detection pixels): median residual 1.4 cm, 83% < 3 cm.
# Every Builder B pose is corrected as AffineFK(H_B^-1(world)). Global, one
# map for all episodes/tasks. Changelog #1; report section "FK calibration".
_CAL = json.loads((SCENES / "calibration_v2.json").read_text())
_H = np.array(_CAL["H_pixel_to_matlocal"], float)
_X0 = float(_CAL["x0_near_edge_world_x"]); _V0 = float(_CAL["v0_arm_centreline"])
_M = np.array(json.loads((SCENES / "calibration_fk_affine.json").read_text())["M"], float)

# Builder B's detections_px.json is the primary segmentation output. For
# block_stack, Builder B's finalize JSON poses disagree with B's own
# detection pixels on 27/80 poses (e.g. ep411 purple initial inverts to px
# (490,330) vs the actual detection (447,482)); the detection pixels fit FK
# grasp dwells at 1.4 cm median (changelog #3), so block poses are taken
# from detection pixels directly. Ring JSON poses agree with detections
# (0 mismatches > 15 px) and tool/eye JSON poses are visual-grounding picks
# (tool validated at 1.9 cm vs dwells); those keep the composite path.
_DET = {}
try:
    _DET = json.loads((SCENES / "detections_px.json").read_text())
except Exception:
    _DET = {}

def _det_px(key, tag, name):
    d = _DET.get(key, {}).get(tag, {}).get(name)
    if isinstance(d, dict) and "px" in d: return np.array(d["px"], float)
    if isinstance(d, (list, tuple)) and len(d) == 2: return np.array(d, float)
    return None

def _affine_px(px):
    return _M.T @ np.array([px[0], px[1], 1.0])

# Eye-drops carton: the carton sits ~10.75 cm above the mat on the platform;
# Builder B's mat-plane homography displaces elevated points (parallax).
# Measured on all 20 episodes' pick dwells vs the composite-corrected JSON
# pose: median offset (-0.014, -0.093) m, residual scatter ~4.6 cm median
# after correction (changelog #3). Applied to the carton INITIAL pose only.
EYE_CARTON_OFFSET = (-0.014, -0.093)

# Parallax model (changelog #3b): the #1 affine was fitted on objects at
# pick height h_ref ~ 0.016 m, so it absorbs the parallax at that height.
# Builder B poses picked from pixels of points at other heights carry an
# extra displacement k*(h - h_ref), k = EYE_CARTON_OFFSET / (0.1075 - 0.016)
# (fitted on the carton, the one directly measured class). Applied to
# B-sourced eye platform (top-plate pick, h=0.09), basket (h=0.03),
# carton_final (h=0.03) and the block stack finals (top-cube detection,
# h=0.045). Mat-level and h~h_ref poses are unaffected by construction.
PARALLAX_K = (EYE_CARTON_OFFSET[0] / 0.0915, EYE_CARTON_OFFSET[1] / 0.0915)
PARALLAX_HREF = 0.016

def _parallax(h):
    d = h - PARALLAX_HREF
    return (PARALLAX_K[0] * d, PARALLAX_K[1] * d)

def _px_from_world(xy):
    q = np.linalg.solve(_H, np.array([xy[0] - _X0, xy[1] + _V0, 1.0]))
    return q[:2] / q[2]

def correct_xy(xy):
    px = _px_from_world(xy)
    return _M.T @ np.array([px[0], px[1], 1.0])

def _correct_yaw(xy, yaw_rad):
    eps = 0.005
    c = correct_xy(xy)
    jx = (correct_xy([xy[0] + eps, xy[1]]) - c) / eps
    jy = (correct_xy([xy[0], xy[1] + eps]) - c) / eps
    d = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    dp = jx * d[0] + jy * d[1]
    return math.atan2(dp[1], dp[0])

def make_draws(task, epi, poses, k):
    """Deterministic per-variant draws (seed 20261122). k=0 -> nominal."""
    if k == 0: return {"jitter": {}, "mass": {}, "friction_mult": {}}
    crc = zlib.crc32(f"{task}__ep{epi}".encode())
    rng = np.random.default_rng(np.random.SeedSequence([P.VARIANT_SEED, crc, k]))
    bodies = {"block_stack": ["cube_a", "cube_b"], "eye_drops_to_basket": ["carton"],
              "ring_insert": ["ring"], "tool_insert": ["tool"],
              "cable_clip": ["clip_block", "cable"]}[task]
    objs = (poses or {}).get("objects", {})
    draws = {"jitter": {}, "mass": {}, "friction_mult": {}}
    for b in bodies:
        # Amendment 1: jitter sampled from Builder B's per-pose sigma
        # estimates (Gaussian, 1 sigma = B's sigma_xy_cm / sigma_yaw_deg).
        if b == "cable":
            sxy, syaw = 0.08, 0.0  # B: cable layout descriptor-grade, sigma 8 cm
        else:
            o = objs.get(b) or {}
            sxy = (o.get("sigma_xy_cm") or 3.5) / 100.0
            syaw = o.get("sigma_yaw_deg") or 0.0
        draws["jitter"][b] = [round(float(rng.normal(0, sxy)), 5),
                              round(float(rng.normal(0, sxy)), 5),
                              round(float(rng.normal(0, syaw)), 3)]
        mr = P.MASS_RANGES.get(b)
        if mr: draws["mass"][b] = round(float(rng.uniform(*mr)), 5)
        draws["friction_mult"][b] = round(float(rng.uniform(*P.FRICTION_MULT_RANGE)), 4)
    if task == "cable_clip":
        lo, hi = P.MASS_RANGES["cable_total"]
        draws["mass"]["cable_total"] = round(float(rng.uniform(lo, hi)), 5)
    if task == "ring_insert":
        # The peg is the outcome reference and is itself a Builder B pose
        # (ring_final, sigma_xy 2.0 cm): sample it per variant within its
        # sigma (changelog #6). Board centre derives from the peg, so the
        # board moves rigidly with it.
        fin = (poses or {}).get("final", {}).get("ring", {})
        sig = (fin.get("sigma_xy_cm") or 2.0) / 100.0
        draws.setdefault("fixture_jitter", {})["peg"] = [
            round(float(rng.normal(0, sig)), 5), round(float(rng.normal(0, sig)), 5)]
    return draws


# ---------------- quaternion helpers (wxyz) ----------------
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])

def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qrot(q, v):
    return qmul(qmul(q, np.array([0, *v])), qinv(q))[1:]

def quat_from_x_dir(d):
    """Quaternion whose local +x axis aligns with unit direction d (wxyz).
    Twist chosen by propagating a reference up vector (Gram-Schmidt)."""
    x = np.asarray(d, float); x /= (np.linalg.norm(x) + 1e-12)
    up = np.array([0, 0, 1.0])
    if abs(np.dot(x, up)) > 0.9: up = np.array([0, 1.0, 0])
    y = np.cross(up, x); y /= (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)  # columns are local axes in world
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1) * 2; q = np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1+R[0,0]-R[1,1]-R[2,2]) * 2; q = np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1+R[1,1]-R[0,0]-R[2,2]) * 2; q = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = math.sqrt(1+R[2,2]-R[0,0]-R[1,1]) * 2; q = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
    return q / np.linalg.norm(q)

def yaw_quat(yaw_deg):
    h = math.radians(yaw_deg) / 2
    return np.array([math.cos(h), 0, 0, math.sin(h)])

def fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)

# ---------------- Builder B pose adapter ----------------
RANK = {"low": 0, "medium": 1, "high": 2}

def _b_pose(e):
    """Builder B entry: pose_xyz_yaw = [x, y, z, yaw_rad_or_null].
    xy (+ measured yaw) passed through the global FK frame correction;
    per-pose sigmas carried through for Amendment-1 variant sampling."""
    v = e.get("pose_xyz_yaw")
    if not v: return None
    yaw = v[3] if len(v) > 3 and v[3] is not None else None
    xy = correct_xy([float(v[0]), float(v[1])])
    yaw_c = _correct_yaw([float(v[0]), float(v[1])], float(yaw)) if yaw is not None else 0.0
    return {"pos": [float(xy[0]), float(xy[1]), float(v[2])],
            "quat": list(yaw_quat(math.degrees(yaw_c))),
            "confidence": e.get("confidence"),
            "sigma_xy_cm": e.get("sigma_xy_cm"), "sigma_yaw_deg": e.get("sigma_yaw_deg")}

def load_episode_poses(task, epi):
    """Builder B schema (landed 2026-11-22): objects keyed by colour/role
    names with pose_xyz_yaw + per-object confidence; finals are separate
    '<name>_final' entries. Returns the builder-facing dict:
    objects (body-name keyed), final (body-name keyed), confidence =
    minimum over initial objects (cable_clip: clip_block governs; the
    cable layout itself is declared low/coarse by Builder B for all 20
    episodes and is initialised as a standard coil at coil_centre_xy)."""
    f = SCENES / f"{task}__ep{epi}.json"
    if not f.exists(): return None
    raw = json.loads(f.read_text())
    objs_raw = raw.get("objects", {})
    out = {"objects": {}, "final": {}, "cable_points": None, "raw": raw,
           "confidence": None, "confidences": {}, "cable_coarse": False}
    def put(store, body, e):
        pp = _b_pose(e)
        if pp: store[body] = pp
    if task == "block_stack":
        man = next((k for k, v in objs_raw.items() if not k.endswith("_final") and v.get("role") == "manipulated"), None)
        base = next((k for k, v in objs_raw.items() if not k.endswith("_final") and v.get("role") == "base"), None)
        rest = [k for k in objs_raw if not k.endswith("_final") and k not in (man, base)]
        key = f"{task}__ep{epi}"
        def put_det(store, body, colour, z):
            """Pose from detection pixels (primary) with JSON fallback."""
            px = _det_px(key, "first", colour) if store is out["objects"] else None
            e = objs_raw.get(colour, {})
            if px is not None:
                xy = _affine_px(px)
                store[body] = {"pos": [float(xy[0]), float(xy[1]), z],
                               "quat": [1.0, 0.0, 0.0, 0.0],
                               "confidence": e.get("confidence"),
                               "sigma_xy_cm": e.get("sigma_xy_cm"),
                               "sigma_yaw_deg": e.get("sigma_yaw_deg")}
            elif e:
                put(store, body, e)
        if man: put_det(out["objects"], "cube_a", man, 0.015)
        if base: put_det(out["objects"], "cube_b", base, 0.015)
        for k in rest[:max(0, 2 - len(out["objects"]))]:
            put_det(out["objects"], "cube_a" if "cube_a" not in out["objects"] else "cube_b", k, 0.015)
        # finals: both cubes at the stack. Stack px = mean of the two final
        # detections when they agree (< 80 px apart); when they disagree one
        # detection is spurious and the manipulated (top) cube's detection
        # is the stack (it is the visible cube in the final frame).
        mf = next((k for k, v in objs_raw.items() if v.get("role") == "manipulated_final"), None)
        bf = next((k for k, v in objs_raw.items() if v.get("role") == "base_final"), None)
        stack_xy = None
        pxs = {}
        for nm, kk in (("man", man), ("base", base)):
            if kk: pxs[nm] = _det_px(key, "final", kk)
        # The stack forms on the base cube: use the final detection nearest
        # the base cube's initial px (the manipulated colour's own final
        # detection is unreliable -- in several episodes the colour mask
        # catches the mat's printed lettering instead of the stacked cube).
        base_init_px = _det_px(key, "first", base) if base else None
        cands = [pxs[nm] for nm in ("man", "base") if pxs.get(nm) is not None]
        if cands and base_init_px is not None:
            stack_px = min(cands, key=lambda q: np.linalg.norm(q - base_init_px))
            stack_xy = _affine_px(stack_px)
        elif pxs.get("man") is not None and pxs.get("base") is not None:
            if np.linalg.norm(pxs["man"] - pxs["base"]) < 80:
                stack_xy = _affine_px((pxs["man"] + pxs["base"]) / 2)
            else:
                stack_xy = _affine_px(pxs["man"])
        elif pxs.get("man") is not None:
            stack_xy = _affine_px(pxs["man"])
        if stack_xy is not None:
            _ps = _parallax(0.045)
            stack_xy = np.array([stack_xy[0] + _ps[0], stack_xy[1] + _ps[1]])
            for body, z, kk in (("cube_a", 0.045, mf), ("cube_b", 0.015, bf)):
                e = objs_raw.get(kk, {}) if kk else {}
                out["final"][body] = {"pos": [float(stack_xy[0]), float(stack_xy[1]), z],
                                      "quat": [1.0, 0.0, 0.0, 0.0],
                                      "confidence": e.get("confidence"),
                                      "sigma_xy_cm": e.get("sigma_xy_cm"),
                                      "sigma_yaw_deg": e.get("sigma_yaw_deg")}
        else:
            if mf: put(out["final"], "cube_a", objs_raw[mf])
            if bf: put(out["final"], "cube_b", objs_raw[bf])
        init_keys = [k for k in objs_raw if not k.endswith("_final")]
    elif task == "cable_clip":
        if "clip_block" in objs_raw: put(out["objects"], "clip_block", objs_raw["clip_block"])
        cab = objs_raw.get("cable", {})
        cc = cab.get("coil_centre_xy")
        if cc:
            ccc = correct_xy([float(cc[0]), float(cc[1])])
            out["cable_points"] = default_cable_points([ccc[0], ccc[1], 0])
            # recentre the default coil on Builder B's (corrected) coil centre
            cx, cy = 0.27, -0.07
            out["cable_points"] = [[x - cx + float(ccc[0]), y - cy + float(ccc[1]), z] for x, y, z in out["cable_points"]]
        out["cable_coarse"] = True
        init_keys = ["clip_block"]
    else:
        name = {"eye_drops_to_basket": "carton", "ring_insert": "ring", "tool_insert": "tool"}[task]
        if name in objs_raw: put(out["objects"], name, objs_raw[name])
        if task == "eye_drops_to_basket" and "carton" in out["objects"]:
            o = out["objects"]["carton"]
            o["pos"] = [o["pos"][0] + EYE_CARTON_OFFSET[0], o["pos"][1] + EYE_CARTON_OFFSET[1], o["pos"][2]]
        if name + "_final" in objs_raw: put(out["final"], name, objs_raw[name + "_final"])
        if task == "eye_drops_to_basket" and "carton" in out["final"]:
            _o = out["final"]["carton"]; _ps = _parallax(0.03)
            _o["pos"] = [_o["pos"][0] + _ps[0], _o["pos"][1] + _ps[1], _o["pos"][2]]
        init_keys = [name] if name in objs_raw else []
    confs = [objs_raw[k].get("confidence") for k in init_keys if k in objs_raw]
    out["confidences"] = {k: objs_raw[k].get("confidence") for k in init_keys if k in objs_raw}
    known = [c for c in confs if c in RANK]
    out["confidence"] = min(known, key=lambda c: RANK[c]) if known else None
    # episode-level fixtures (Builder B repeats the global set per episode)
    out["fixtures_b"] = raw.get("fixtures", {})
    return out

def _fx_entry(e):
    if not isinstance(e, dict): return {}
    pp = _b_pose(e)
    out = {}
    if pp: out["pos"] = pp["pos"]
    if "top_z" in e: out["top_z"] = e["top_z"]
    return out

def load_fixtures(task, poses=None):
    """Global fixtures.json (Builder B schema) merged with defaults;
    episode fixtures override when poses are passed."""
    fx = {k: dict(v) for k, v in P.DEFAULT_FIXTURES.get(task, {}).items()}
    f = SCENES / "fixtures.json"
    raw = {}
    if f.exists(): raw = json.loads(f.read_text()).get(task, {}) or {}
    if poses and poses.get("fixtures_b"): raw = {**raw, **poses["fixtures_b"]}
    for name, e in raw.items():
        if name == "recesses" or not isinstance(e, dict): continue
        ent = _fx_entry(e)
        if ent:
            if task == "eye_drops_to_basket" and "pos" in ent:
                _h = {"platform": 0.09, "basket": 0.03}.get(name)
                if _h is not None:
                    _ps = _parallax(_h)
                    ent["pos"] = [ent["pos"][0] + _ps[0], ent["pos"][1] + _ps[1], ent["pos"][2]]
            fx[name] = {**fx.get(name, {}), **ent}
    if "target_slot" in fx:  # Builder B name -> builder name
        fx["slot"] = fx.pop("target_slot")
        if "pos" in fx["slot"]: fx["slot"]["pos"] = [fx["slot"]["pos"][0], fx["slot"]["pos"][1], fx["slot"]["pos"][2]]
    return fx

# ---------------- XML fragments ----------------
def common_xml():
    mx, my = P.MAT_CENTER; sx, sy, st = P.MAT_SIZE
    return f"""
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.001"
      friction="{fmt(P.FLOOR_FRICTION)}" rgba="0.35 0.30 0.25 1"/>
    <geom name="mat" type="box" pos="{mx} {my} {-st/2}" size="{sx/2} {sy/2} {st/2}"
      friction="{fmt(P.MAT_FRICTION)}" rgba="0.16 0.19 0.24 1"/>
    <geom name="wall_back" type="box" pos="{mx} {my+sy/2+0.005} 0.03" size="{sx/2} 0.005 0.03"
      friction="{fmt(P.WALL_FRICTION)}" rgba="0.76 0.66 0.50 1"/>
    <geom name="wall_front" type="box" pos="{mx} {my-sy/2-0.005} 0.03" size="{sx/2} 0.005 0.03"
      friction="{fmt(P.WALL_FRICTION)}" rgba="0.76 0.66 0.50 1"/>
    <geom name="wall_left" type="box" pos="{mx-sx/2-0.005} {my} 0.03" size="0.005 {sy/2} 0.03"
      friction="{fmt(P.WALL_FRICTION)}" rgba="0.76 0.66 0.50 1"/>
    <geom name="wall_right" type="box" pos="{mx+sx/2+0.005} {my} 0.03" size="0.005 {sy/2} 0.03"
      friction="{fmt(P.WALL_FRICTION)}" rgba="0.76 0.66 0.50 1"/>
    <camera name="topcam" pos="{mx} {my} 0.85" quat="1 0 0 0" fovy="50"/>
    <camera name="frontcam" pos="0.75 -0.55 0.35" xyaxes="0.66 -0.75 0 0.28 0.24 -0.92" fovy="55"/>
"""

def free_body(name, pose, inner, rgba=None):
    return (f'<body name="{name}" pos="{fmt(pose["pos"])}" quat="{fmt(pose["quat"])}">'
            f'<joint name="{name}_free" type="free" damping="0.01"/>{inner}</body>')

def build(task, epi, poses, fixtures, var=None):
    var = var or {"jitter": {}, "mass": {}, "friction_mult": {}}
    def jpose(body, pose):
        j = var["jitter"].get(body)
        if not j or pose is None: return pose
        q = {"pos": [pose["pos"][0] + j[0], pose["pos"][1] + j[1], pose["pos"][2]],
             "quat": list(qmul(yaw_quat(j[2]), np.array(pose["quat"])))}
        return q
    def vmass(body, nominal):
        return var["mass"].get(body, nominal)
    def vfric(body, tup):
        m = var["friction_mult"].get(body, 1.0)
        return (round(tup[0] * m, 4), tup[1], tup[2])
    """Returns (xml_string, weldable_body_names, meta)."""
    welds, bodies, meta = [], [], {}
    objs = (poses or {}).get("objects", {})
    def obj_pose(*names, default=None):
        for n in names:
            if n in objs: return objs[n]
        return default
    if task == "block_stack":
        e = P.CUBE_EDGE
        cols = {"cube_a": "0.55 0.25 0.75 1", "cube_b": "0.25 0.65 0.30 1"}
        for i, nm in enumerate(["cube_a", "cube_b"]):
            dflt = {"pos": [0.26 + 0.07 * i, -0.05 + 0.09 * i, e / 2], "quat": [1, 0, 0, 0]}
            pose = obj_pose(nm, "cube1" if i == 0 else "cube2", "purple" if i == 0 else "green", default=dflt)
            inner = (f'<geom name="{nm}_geom" type="box" size="{e/2} {e/2} {e/2}" '
                     f'mass="{vmass(nm, P.CUBE_MASS)}" friction="{fmt(vfric(nm, P.CUBE_FRICTION))}" rgba="{cols[nm]}"/>')
            bodies.append(free_body(nm, jpose(nm, pose), inner)); welds.append(nm)
    elif task == "eye_drops_to_basket":
        pf = fixtures["platform"]["pos"]; bk = fixtures["basket"]["pos"]
        tw, td, tt = P.PLATFORM_TOP; h = P.PLATFORM_HEIGHT
        bodies.append(f"""
    <body name="platform" pos="{pf[0]} {pf[1]} 0">
      <geom name="platform_top" type="box" pos="0 0 {h-tt/2}" size="{tw/2} {td/2} {tt/2}"
        friction="{fmt(P.PLATFORM_FRICTION)}" rgba="0.92 0.92 0.92 1"/>
      <geom name="platform_web" type="box" pos="0 0 {h/2}" size="0.01 {td/4} {h/2}"
        friction="{fmt(P.PLATFORM_FRICTION)}" rgba="0.92 0.92 0.92 1"/>
      <geom name="platform_base" type="box" pos="0 0 0.002" size="{tw/2} {td/2} 0.002"
        friction="{fmt(P.PLATFORM_FRICTION)}" rgba="0.92 0.92 0.92 1"/>
    </body>
    <body name="basket" pos="{bk[0]} {bk[1]} 0">
      <geom name="basket_floor" type="box" pos="0 0 {P.BASKET_WALL_T/2}" size="{P.BASKET_OUTER[0]/2} {P.BASKET_OUTER[1]/2} {P.BASKET_WALL_T/2}"
        friction="{fmt(P.BASKET_FRICTION)}" rgba="0.85 0.90 0.95 0.5"/>
      <geom name="basket_w1" type="box" pos="0 {P.BASKET_OUTER[1]/2-P.BASKET_WALL_T/2} {P.BASKET_WALL_H/2}" size="{P.BASKET_OUTER[0]/2} {P.BASKET_WALL_T/2} {P.BASKET_WALL_H/2}" friction="{fmt(P.BASKET_FRICTION)}" rgba="0.85 0.90 0.95 0.5"/>
      <geom name="basket_w2" type="box" pos="0 {-P.BASKET_OUTER[1]/2+P.BASKET_WALL_T/2} {P.BASKET_WALL_H/2}" size="{P.BASKET_OUTER[0]/2} {P.BASKET_WALL_T/2} {P.BASKET_WALL_H/2}" friction="{fmt(P.BASKET_FRICTION)}" rgba="0.85 0.90 0.95 0.5"/>
      <geom name="basket_w3" type="box" pos="{P.BASKET_OUTER[0]/2-P.BASKET_WALL_T/2} 0 {P.BASKET_WALL_H/2}" size="{P.BASKET_WALL_T/2} {P.BASKET_OUTER[1]/2} {P.BASKET_WALL_H/2}" friction="{fmt(P.BASKET_FRICTION)}" rgba="0.85 0.90 0.95 0.5"/>
      <geom name="basket_w4" type="box" pos="{-P.BASKET_OUTER[0]/2+P.BASKET_WALL_T/2} 0 {P.BASKET_WALL_H/2}" size="{P.BASKET_WALL_T/2} {P.BASKET_OUTER[1]/2} {P.BASKET_WALL_H/2}" friction="{fmt(P.BASKET_FRICTION)}" rgba="0.85 0.90 0.95 0.5"/>
    </body>""")
        sx, sy, sz = P.CARTON_SIZE
        dflt = {"pos": [pf[0], pf[1], h + sy / 2], "quat": list(yaw_quat(0))}
        pose = obj_pose("carton", "box", "eye_drops", default=dflt)
        inner = (f'<geom name="carton_geom" type="box" size="{sx/2} {sy/2} {sz/2}" '
                 f'mass="{vmass("carton", P.CARTON_MASS)}" friction="{fmt(vfric("carton", P.CARTON_FRICTION))}" rgba="0.25 0.45 0.85 1"/>')
        bodies.append(free_body("carton", jpose("carton", pose), inner)); welds.append("carton")
    elif task == "ring_insert":
        bp = list(fixtures["board"]["pos"]); bt = P.BOARD_H
        pegp = list(fixtures.get("peg", {}).get("pos", bp))
        fj = (var.get("fixture_jitter") or {}).get("peg")
        if fj:
            bp = [bp[0] + fj[0], bp[1] + fj[1], bp[2]]
            pegp = [pegp[0] + fj[0], pegp[1] + fj[1], pegp[2]]
        bodies.append(f"""
    <body name="board" pos="{bp[0]} {bp[1]} 0">
      <geom name="board_geom" type="cylinder" pos="0 0 {bt/2}" size="{P.BOARD_R} {bt/2}"
        friction="{fmt(P.BOARD_FRICTION)}" rgba="0.92 0.87 0.72 1"/>
    </body>
    <body name="peg" pos="{pegp[0]} {pegp[1]} 0">
      <geom name="peg_geom" type="cylinder" pos="0 0 {bt + P.PEG_H/2}" size="{P.PEG_R} {P.PEG_H/2}"
        friction="{fmt(P.PEG_FRICTION)}" rgba="0.80 0.65 0.42 1"/>
    </body>""")
        meta["board_top_z"] = bt; meta["peg_xy"] = pegp[:2]
        # ring: one free body, N capsule geoms around a circle (local xy-plane)
        geoms = []
        n = P.RING_SEGMENTS; rmid = (P.RING_R_OUT + P.RING_R_IN) / 2
        tube = min((P.RING_R_OUT - P.RING_R_IN) / 2, P.RING_THICK / 2)
        for k in range(n):
            a0 = 2 * math.pi * k / n; a1 = 2 * math.pi * (k + 1) / n
            p0 = (rmid * math.cos(a0), rmid * math.sin(a0), 0)
            p1 = (rmid * math.cos(a1), rmid * math.sin(a1), 0)
            geoms.append(f'<geom type="capsule" fromto="{fmt(p0)} {fmt(p1)}" size="{tube}" '
                         f'friction="{fmt(vfric("ring", P.RING_FRICTION))}" rgba="0.95 0.80 0.15 1"/>')
        per = vmass("ring", P.RING_MASS) / n
        geoms = [g.replace('friction=', f'mass="{per:.6g}" friction=', 1) for g in geoms]
        dflt = {"pos": [bp[0] - 0.045, bp[1] - 0.045, bt + P.RING_THICK / 2], "quat": [1, 0, 0, 0]}
        pose = obj_pose("ring", default=dflt)
        bodies.append(free_body("ring", jpose("ring", pose), "".join(geoms))); welds.append("ring")
    elif task == "tool_insert":
        tp = fixtures["tray"]["pos"]; sp = fixtures["slot"]["pos"]
        tw, td, th = P.TRAY_SIZE
        bodies.append(f"""
    <body name="tray" pos="{tp[0]} {tp[1]} 0">
      <geom name="tray_geom" type="box" pos="0 0 {th/2}" size="{tw/2} {td/2} {th/2}"
        friction="{fmt(P.TRAY_FRICTION)}" rgba="0.10 0.10 0.10 1"/>
    </body>""")
        meta["slot_pos"] = sp; meta["slot_yaw_deg"] = fixtures["slot"].get("yaw_deg", 0.0)
        dflt = {"pos": [0.28, -0.06, P.TOOL_R], "quat": list(yaw_quat(20))}
        pose = obj_pose("tool", "loose_tool", default=dflt)
        inner = (f'<geom name="tool_geom" type="capsule" fromto="{-P.TOOL_LEN/2} 0 0 {P.TOOL_LEN/2} 0 0" '
                 f'size="{P.TOOL_R}" mass="{vmass("tool", P.TOOL_MASS)}" friction="{fmt(vfric("tool", P.TOOL_FRICTION))}" rgba="0.75 0.75 0.78 1"/>')
        bodies.append(free_body("tool", jpose("tool", pose), inner)); welds.append("tool")
    elif task == "cable_clip":
        cbp = obj_pose("clip_block", default=None)
        if cbp is not None:
            cbp = jpose("clip_block", cbp)
            cb = cbp["pos"]
            import math as _m
            yaw = _m.degrees(2 * _m.atan2(cbp["quat"][3], cbp["quat"][0]))
        else:
            cb = fixtures["clip_block"]["pos"]; yaw = fixtures["clip_block"].get("yaw_deg", 0.0)
        bw, bd, bh = P.BLOCK_SIZE; cw = P.CLIP_CHANNEL_W
        bodies.append(f"""
    <body name="clip_block" pos="{cb[0]} {cb[1]} 0" quat="{fmt(yaw_quat(yaw))}">
      <geom name="block_geom" type="box" pos="0 0 {bh/2}" size="{bw/2} {bd/2} {bh/2}"
        friction="{fmt(P.BLOCK_FRICTION)}" rgba="0.93 0.93 0.93 1"/>
      <geom name="clip_jaw_a" type="box" pos="0 {cw/2 + 0.004} {bh + 0.006}" size="0.012 0.004 0.006"
        friction="{fmt(P.CLIP_FRICTION)}" rgba="0.08 0.08 0.08 1"/>
      <geom name="clip_jaw_b" type="box" pos="0 {-cw/2 - 0.004} {bh + 0.006}" size="0.012 0.004 0.006"
        friction="{fmt(P.CLIP_FRICTION)}" rgba="0.08 0.08 0.08 1"/>
    </body>""")
        meta["clip_block_pos"] = cb; meta["clip_yaw_deg"] = yaw
        meta["clip_channel_z"] = bh + 0.006
        pts = (poses or {}).get("cable_points") or default_cable_points(cb)
        cj = var["jitter"].get("cable")
        if cj: pts = [[x + cj[0], y + cj[1], z] for x, y, z in pts]
        bodies.append(cable_xml(pts, seg_mass=var["mass"].get("cable_total", 0.045) / P.CABLE_N,
                                 fric_mult=var["friction_mult"].get("cable", 1.0)))
        welds.extend([f"cable_seg_{i}" for i in range(P.CABLE_N)])
    else:
        raise ValueError(task)
    # K welds per weldable body: MuJoCo 4.0 python bindings cannot set
    # eq_data at runtime, so replay.py precomputes each grasp crossing's
    # relpose into successive weld slots before the sim starts.
    K = 2 if task == "cable_clip" else 4
    weld_xml = "".join(
        f'<weld name="weld_{w}_{k}" body1="gripper" body2="{w}" active="false" solref="0.005 1" solimp="0.95 0.99 0.001"/>'
        for w in welds for k in range(K))
    xml = f"""<!-- {P.SUITE_VERSION} task={task} ep={epi} variant_seed={P.VARIANT_SEED} param_hash={param_table_hash()} frame_correction=fk-affine-v2 -->
<mujoco model="twin_{task}_ep{epi}">
  <compiler angle="radian" meshdir="{P.ARM_MESHDIR}" autolimits="true"/>
  <option timestep="{P.TIMESTEP}" gravity="0 0 -9.81"/>
  <visual><headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="145" elevation="-22"/></visual>
  <include file="{P.ARM_XML}"/>
  <worldbody>
    <light pos="0.3 -0.3 0.9" dir="-0.2 0.2 -1" diffuse="0.85 0.85 0.85"/>
    {common_xml()}
    {''.join(bodies)}
  </worldbody>
  <equality>{weld_xml}</equality>
</mujoco>
"""
    return xml, welds, meta

def default_cable_points(block_pos):
    """Default coil layout (used only when Builder B gives no polyline):
    ~2-turn coil of radius 55 mm centred left of the clip block, z=cable r."""
    cx, cy = 0.27, -0.07
    pts = []
    total = P.CABLE_N * P.CABLE_SEG_LEN
    turns = 2.0; r = 0.055
    for i in range(P.CABLE_N + 1):
        s = i / P.CABLE_N; a = 2 * math.pi * turns * s
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a), P.CABLE_R])
    return pts

def resample_polyline(pts, n, seg):
    pts = [np.array(p, float) for p in pts]
    if len(pts) < 2: return default_cable_points([0.3, 0.05, 0])
    # arc-length resample to n segments of ~seg length (uniform parameter walk)
    d = [0.0]
    for a, b in zip(pts[:-1], pts[1:]): d.append(d[-1] + float(np.linalg.norm(b - a)))
    d = np.array(d); total = d[-1]
    if total < 1e-6: return default_cable_points([0.3, 0.05, 0])
    out = []
    for i in range(n + 1):
        t = total * i / n
        j = int(np.searchsorted(d, t) - 1); j = max(0, min(j, len(pts) - 2))
        f = 0 if d[j + 1] - d[j] < 1e-9 else (t - d[j]) / (d[j + 1] - d[j])
        p = pts[j] * (1 - f) + pts[j + 1] * f
        p[2] = max(p[2], P.CABLE_R)
        out.append(p)
    return out

def cable_xml(pts, seg_mass=None, fric_mult=1.0):
    rp = resample_polyline(pts, P.CABLE_N, P.CABLE_SEG_LEN)
    frames = [quat_from_x_dir(rp[i + 1] - rp[i]) for i in range(P.CABLE_N)]
    return cable_xml_build(rp, frames, seg_mass=seg_mass, fric_mult=fric_mult)

def cable_xml_build(rp, frames, seg_mass=None, fric_mult=1.0):
    n = P.CABLE_N
    us = [x / 2 for x in P.USB_SIZE]
    usb = ('<body name="usb" pos="' + str(P.CABLE_SEG_LEN) + ' 0 0">'
           '<geom type="box" pos="' + fmt([P.USB_SIZE[0] / 2 - 0.002, 0, 0]) + '" '
           'size="' + fmt(us) + '" mass="' + str(P.USB_MASS) + '" '
           'friction="' + fmt(P.CLIP_FRICTION) + '" rgba="0.2 0.2 0.2 1"/></body>')
    child = usb
    for i in reversed(range(n)):
        if i == 0:
            pos, quat = rp[0], frames[0]
            joint = '<joint name="cable_seg_0_free" type="free" damping="0.05"/>'
        else:
            quat = qmul(qinv(frames[i - 1]), frames[i])
            pos = qrot(qinv(frames[i - 1]), rp[i] - rp[i - 1])
            joint = ('<joint name="cable_seg_' + str(i) + '_ball" type="ball" damping="'
                     + str(P.CABLE_JOINT_DAMPING) + '"/>')
        geom = ('<geom type="capsule" fromto="0 0 0 ' + str(P.CABLE_SEG_LEN) + ' 0 0" '
                'size="' + str(P.CABLE_R) + '" mass="' + format(seg_mass if seg_mass else P.CABLE_SEG_MASS, ".6g") + '" '
                'friction="' + fmt((round(P.CABLE_FRICTION[0] * fric_mult, 4), P.CABLE_FRICTION[1], P.CABLE_FRICTION[2])) + '" rgba="0.05 0.05 0.05 1"/>')
        child = ('<body name="cable_seg_' + str(i) + '" pos="' + fmt(pos) + '" quat="'
                 + fmt(quat) + '">' + joint + geom + child + '</body>')
    return child

def main():
    split = json.loads(P.SPLIT.read_text())
    only_smoke = "--smoke" in sys.argv
    made, skipped = [], []
    for task, v in split["tasks"].items():
        for rec in v["episodes"]:
            epi = rec["episode_index"]
            poses = load_episode_poses(task, epi)
            if poses is None and not only_smoke:
                skipped.append(f"{task}__ep{epi}"); continue
            fixtures = load_fixtures(task, poses)
            vlog = {"suite_version": P.SUITE_VERSION, "seed": P.VARIANT_SEED,
                    "param_hash": param_table_hash(), "frame_correction": "fk-affine-v2 (AffineFK o inv(H_B), changelog #1)", "task": task, "episode": epi,
                    "confidence": (poses or {}).get("confidence"), "variants": []}
            for k in range(P.VARIANT_K):
                draws = make_draws(task, epi, poses, k)
                xml, welds, meta = build(task, epi, poses, fixtures, var=draws)
                out = SCENES / f"{task}__ep{epi}_v{k}.xml"
                out.write_text(xml)
                if k == 0: (SCENES / f"{task}__ep{epi}.xml").write_text(xml)
                vlog["variants"].append({"k": k, "draws": draws, "scene": out.name})
                made.append(out.name)
            (SCENES / f"{task}__ep{epi}_variants.json").write_text(json.dumps(vlog, indent=1))
    print(f"built {len(made)} variant scenes; skipped (no Builder B poses): {len(skipped)}")
    if skipped: print("skipped:", skipped[:10], "...")

if __name__ == "__main__":
    main()
