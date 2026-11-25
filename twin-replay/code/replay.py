"""Study 001 twin-replay harness (Builder C, Phase 4).

Executes the RECORDED gt trajectory kinematically in a per-episode twin
scene: arm qpos set from gt (degrees->radians, identical joint names) with
linear interpolation at physics substeps (2000 Hz), objects fully dynamic,
grasp via the declared weld device (docs/physics-parameters.md section 7 +
normalisation in twin_params.py). After the trajectory, the scene settles
until the physics engine reports rest (every dynamic object body's speed
below |g| * dt read from the model; founder ruling 2026-11-25, replacing
the former fixed 1.0 s settle wait).
No checkpoint inference of any kind is performed here.

Outcome classes implement docs/twin-replay-preregistration-20261122.md
EXACTLY as written; geometric sub-checks are recorded so borderline calls
are auditable.

Usage: replay.py <task> <episode_id> [--out DIR] [--render]
Writes <task>__ep<id>.json result + optional top-view PNG.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).parent))
import twin_params as P
import build_scenes as B

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]

def closedness_trace(grip_deg):
    """Closedness in [0,1] from the recorded gripper channel.

    DIRECTION (calibration finding, changelog #4): the recorded channel is
    INVERTED vs the naive reading -- LOW values are jaws-closed (holding),
    HIGH values are jaws-open. Proof: in block_stack ep411 the arm site
    transits from the green cube to the stack point during frames ~124-225,
    exactly the trace's low phase (117-227, 2-4 deg); under the high=holding
    reading the hold would end at frame ~115, before the transit, and the
    cube could never reach the stack (Builder B's finals + the final frame
    show it stacked). ring_insert ep7 confirms: the ring's recess->peg
    transit is the low phase 152-223, and at frame 177 (trace 1 deg) the
    ring is visibly pinched in the jaws mid-air. Normalisation remains
    per-episode (p5/p99), direction reversed: closedness = (hi - v)/(hi - lo).
    """
    g = np.asarray(grip_deg, float)
    lo, hi = np.percentile(g, 5), np.percentile(g, 99)
    if hi - lo < P.GRIP_MIN_EXCURSION_DEG:
        return np.zeros_like(g), float(lo), float(hi), False
    return np.clip((hi - g) / (hi - lo), 0, 1), float(lo), float(hi), True

def quat_angle_deg(q):
    return 0.0  # placeholder (axis checks done via vectors)

def run_episode(task, epi, render=False, out_dir=None, variant=0):
    import mujoco
    scene = P.TWIN / "scenes" / f"{task}__ep{epi}_v{variant}.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = P.TIMESTEP
    data = mujoco.MjData(model)
    # arm geoms: no contacts (grasp is via weld only; physics-params s.7)
    arm_body_ids = set()
    for nm in ARM_BODIES:
        try: arm_body_ids.add(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm))
        except Exception: pass
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) in arm_body_ids:
            model.geom_contype[gid] = 0; model.geom_conaffinity[gid] = 0
    qadr = {j: int(model.joint(j).qposadr[0]) for j in JOINTS}
    vadr = {j: int(model.joint(j).dofadr[0]) for j in JOINTS}
    grip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    grip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    gt = np.load(P.RECORDS / f"base__{task}__ep{epi}.npz")["gt"].astype(float)
    T = len(gt)
    gt_rad = np.radians(gt[:, :5])
    closed, g_lo, g_hi, has_excursion = closedness_trace(gt[:, 5])
    # model gripper joint visual: closedness 0 -> 1.45 rad (open), 1 -> 0.10 (closed)
    grip_joint_rad = 1.45 + closed * (0.10 - 1.45)

    weld_slots = {}   # object name -> [eq ids in slot order]
    for eq in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
        if nm and nm.startswith("weld_"):
            obj = nm[5:].rsplit("_", 1)[0]
            weld_slots.setdefault(obj, []).append(eq)
    for v in weld_slots.values(): v.sort()
    body_id = {nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in weld_slots}

    def body_pos(nm):
        return np.array(data.xpos[body_id[nm]])

    def set_arm(frame_float):
        i0 = int(math.floor(frame_float)); i1 = min(i0 + 1, T - 1)
        f = frame_float - i0
        for k, j in enumerate(JOINTS[:5]):
            data.qpos[qadr[j]] = gt_rad[i0, k] * (1 - f) + gt_rad[i1, k] * f
            data.qvel[vadr[j]] = 0.0
        data.qpos[qadr["gripper"]] = grip_joint_rad[i0] * (1 - f) + grip_joint_rad[i1] * f
        data.qvel[vadr["gripper"]] = 0.0

    mujoco.mj_resetData(model, data)
    set_arm(0.0)
    mujoco.mj_forward(model, data)
    init_pos = {nm: np.array(data.xpos[bid]) for nm, bid in body_id.items()}
    # brief pre-settle with arm held at frame 0 (0.25 s) so objects start at rest
    for _ in range(int(0.25 / P.TIMESTEP)):
        set_arm(0.0); mujoco.mj_step(model, data)

    # build-defect check (Amendment 1 A2): objects must still be at their
    # spawn poses after the pre-settle; large displacement / sink / NaN =
    # build defect (variant excluded from statistics, defect logged).
    build_defect = None
    mujoco.mj_forward(model, data)
    for nm, bid in body_id.items():
        if task == "cable_clip" and nm.startswith("cable_seg"):
            continue
        dpos = np.array(data.xpos[bid])
        if not np.all(np.isfinite(dpos)):
            build_defect = f"{nm}: NaN position after pre-settle"; break
        disp = float(np.linalg.norm(dpos - init_pos[nm]))
        if disp > 0.03 or dpos[2] < -0.02:
            build_defect = f"{nm}: displaced {disp:.3f} m / z={dpos[2]:.3f} after pre-settle"
            break

    # ---- runtime grasp state machine -------------------------------------
    # Attach semantics (documented interpretations of Builder A's spec,
    # changelog #2 + #5):
    #  * Builder A's crossing-instant centre-distance reading never fires
    #    (at the 50% crossing the gripper is still approaching; the 3D
    #    site-to-centre distance is dominated by the inherent vertical gap
    #    of a top grasp; and for the carton/tool the physical grasp point
    #    sits at the object's half-extent, ~4.5 / ~5+ cm from its centre).
    #    The gate is therefore evaluated on the PLANAR distance from the
    #    gripperframe site to the object's MATERIAL (point-to-shape: box
    #    footprint / annulus / capsule axis), not its centre. Since
    #    2026-11-25 (founder ruling) the gate is a CONTACT read, not the
    #    former 3.0 cm proximity band: attach requires the site to reach
    #    the object's material (see contact_dist below).
    #  * Attach happens at the excursion's dwell (arg-min of the shape
    #    distance), using a SNAPSHOT relpose from that frame, once the
    #    distance turns up >= 3 mm, plateaus, or the excursion ends. The
    #    weld preserves the dwell relative transform (jaw geometry), so
    #    re-grasps after a drop use the dropped pose.
    #  * An excursion is a grasp candidate only if it shows an APPROACH:
    #    the shape distance must fall >= 3 mm from its excursion-start
    #    value before the minimum. This rejects the demos' final EMPTY
    #    close (jaws shut next to the just-placed object while the arm
    #    retreats -- distance only grows), which otherwise re-attaches
    #    and drags the placed object home (11/20 ring episodes in run 1).
    #  * Release: first frame closedness <= 30% while attached (Builder
    #    A's value, instantaneous).
    # MuJoCo 4.0.1 bindings cannot set eq_data on MjData, so an attach
    # writes the relpose into model.eq_data and recreates MjData, carrying
    # the full integration state across via mj_getState/mj_setState.
    def shape_dist(nm, site_xy):
        """Planar distance from site to the object's material (0 if over it)."""
        bp = body_pos(nm)
        q = np.array(data.xquat[body_id[nm]])
        yaw = 2 * math.atan2(q[3], q[0])
        off = np.array(site_xy) - bp[:2]
        c, sn = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * off[0] - sn * off[1], sn * off[0] + c * off[1]
        if nm.startswith("cube"):
            dx = max(abs(lx) - P.CUBE_EDGE / 2, 0.0); dy = max(abs(ly) - P.CUBE_EDGE / 2, 0.0)
            return math.hypot(dx, dy)
        if nm == "carton":
            dx = max(abs(lx) - P.CARTON_SIZE[0] / 2, 0.0); dy = max(abs(ly) - P.CARTON_SIZE[1] / 2, 0.0)
            return math.hypot(dx, dy)
        if nm == "ring":
            cd = math.hypot(off[0], off[1])
            rmid = (P.RING_R_OUT + P.RING_R_IN) / 2; tube = (P.RING_R_OUT - P.RING_R_IN) / 2
            return max(abs(cd - rmid) - tube, 0.0)
        if nm == "tool":
            dx = max(abs(lx) - P.TOOL_LEN / 2, 0.0)
            return math.hypot(dx, abs(ly))
        if nm.startswith("cable_seg"):
            dx = min(max(lx, 0.0), P.CABLE_SEG_LEN) - lx
            return math.hypot(max(dx, 0.0) if lx < 0 or lx > P.CABLE_SEG_LEN else 0.0, ly)
        return float(np.linalg.norm(off))

    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_buf = np.empty(mujoco.mj_stateSize(model, state_spec))
    used = {nm: 0 for nm in weld_slots}
    attached, attached_eq, events = None, None, []
    last_attach_frame = -10
    seeking = False
    trackers = {}   # obj -> tracker dict (per-object, no mid-excursion switching)
    # Release-recency ineligibility (changelog #5): an object just released
    # cannot be re-attached until the gripper has genuinely left it (site
    # >= 3 cm from its centre at some point after the release). This blocks
    # the demos' final EMPTY close -- jaws shutting next to the just-placed
    # object during the retreat -- from re-attaching and dragging the
    # placed object home, while genuine re-grasps (which leave and return,
    # e.g. ring ep7: site reaches 3.8 cm between release and re-grasp)
    # stay eligible.
    ineligible = set()
    # Pad-object CONTACT read (founder ruling 2026-11-25, replacing the
    # former GRASP_DIST_M 3.0 cm proximity gate): the planar distance at
    # which the gripper site touches the object's material. shape_dist()
    # measures to the material itself for the footprint objects (cube,
    # carton, ring), so contact is 0; for the axis-measured objects
    # (tool, cable segments) it measures to the centreline, so contact
    # is the object's own radius. No chosen tolerance anywhere: contact
    # is the object's geometry, read at the dwell.
    def contact_dist(nm):
        if nm == "tool": return P.TOOL_R
        if nm.startswith("cable_seg"): return P.CABLE_R
        return 0.0
    # Per-object centre-distance bound at the grasp snapshot: compact
    # objects (cubes, ring) are bounded by their own footprint extent --
    # a site in contact with the material is necessarily within it --
    # computed from the built geometry (converted 2026-11-25 from the
    # former flat 3.0 cm gate value). Elongated objects are grasped
    # off-centre, so their bound is their physical grasp extent (carton
    # half-length + margin, tool half-length + margin, cable segment
    # half-length + gate; pre-existing values, unchanged).
    def centre_bound(nm):
        if nm == "carton": return 0.060
        if nm == "tool": return 0.075
        if nm.startswith("cable_seg"): return 0.045
        if nm == "ring": return P.RING_R_OUT
        return P.CUBE_EDGE * math.sqrt(2.0) / 2   # cube footprint half-diagonal
    steps_per_frame = max(1, int(round((1.0 / P.FPS) / P.TIMESTEP)))
    prev_c = float(closed[0])

    def do_attach(fr, c, snap):
        nonlocal data, attached, attached_eq, seeking, trackers, last_attach_frame
        obj = snap["obj"]
        eq = weld_slots[obj][used[obj]]; used[obj] += 1
        row = model.eq_data[eq]
        row[0:3] = 0.0
        row[3:6] = B.qrot(B.qinv(snap["gq"]), snap["op"] - snap["gp"])
        row[6:10] = B.qmul(B.qinv(snap["gq"]), snap["oq"])
        mujoco.mj_forward(model, data)
        snap_gap = float(np.linalg.norm(np.array(data.xpos[grip_body]) - snap["gp"]))
        mujoco.mj_getState(model, data, state_buf, state_spec)
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, state_buf, state_spec)
        data.eq_active[eq] = 1
        attached, attached_eq = obj, eq
        last_attach_frame = fr
        seeking = False; trackers = {}
        events.append({"frame": fr, "event": "attach", "object": obj,
                       "planned_dist_m": round(float(snap["d"]), 4),
                       "live_dist_m": round(float(snap["d"]), 4),
                       "snap_gap_m": round(snap_gap, 4),
                       "gate": "planar_shape_dwell", "closedness": round(c, 3)})

    for fr in range(T):
        c = float(closed[fr])
        # eligibility maintenance: a released object becomes eligible again
        # once the site has been >= 3 cm from its centre
        if ineligible:
            mujoco.mj_forward(model, data)
            gpos0 = np.array(data.site_xpos[grip_site]) if grip_site >= 0 else np.array(data.xpos[grip_body])
            for nm in list(ineligible):
                if float(np.linalg.norm(body_pos(nm)[:2] - gpos0[:2])) >= 0.03:
                    ineligible.discard(nm)
                    trackers.pop(nm, None)
        if has_excursion and attached is None and not seeking and prev_c < P.ATTACH_CLOSEDNESS <= c:
            seeking = True; trackers = {}
        if seeking and attached is None:
            mujoco.mj_forward(model, data)
            gpos = np.array(data.site_xpos[grip_site]) if grip_site >= 0 else np.array(data.xpos[grip_body])
            ending = c <= P.RELEASE_CLOSEDNESS
            fired = False
            for nm in weld_slots:
                if nm in ineligible: continue
                d_shape = shape_dist(nm, gpos[:2])
                d_centre = float(np.linalg.norm(body_pos(nm)[:2] - gpos[:2]))
                tr = trackers.get(nm)
                if tr is None:
                    tr = trackers[nm] = {"min_shape": d_shape, "min_centre": d_centre, "snap": None, "fr0": fr}
                if d_shape <= tr["min_shape"]:
                    tr["min_shape"] = d_shape
                    tr["snap"] = {"obj": nm, "d": d_shape,
                                  "gp": np.array(data.xpos[grip_body]), "gq": np.array(data.xquat[grip_body]),
                                  "op": body_pos(nm), "oq": np.array(data.xquat[body_id[nm]])}
                tr["min_centre"] = min(tr["min_centre"], d_centre)
                turned = d_shape >= tr["min_shape"] + 0.003
                plateau = d_shape <= tr["min_shape"] + 0.001 and fr - tr["fr0"] >= 2
                if (tr["min_shape"] <= contact_dist(nm) and tr["min_centre"] <= centre_bound(nm)
                        and used[nm] < len(weld_slots[nm]) and (turned or plateau or ending)):
                    do_attach(fr, c, tr["snap"])
                    fired = True
                    break
            if not fired and ending:
                seeking = False
                elig = {n: t for n, t in trackers.items()}
                best_nm = min(elig, key=lambda n: elig[n]["min_shape"]) if elig else None
                events.append({"frame": fr, "event": "no_attach_in_excursion",
                               "min_dist_m": round(float(elig[best_nm]["min_shape"]), 4) if best_nm else None})
                trackers = {}
        if attached is not None and c <= P.RELEASE_CLOSEDNESS and fr > last_attach_frame:
            data.eq_active[attached_eq] = 0
            events.append({"frame": fr, "event": "release", "object": attached,
                           "closedness": round(c, 3)})
            ineligible.add(attached)
            attached, attached_eq = None, None
        prev_c = c
        for ssub in range(steps_per_frame):
            frac = (ssub + 1) / steps_per_frame
            set_arm(min(fr + frac, T - 1))
            mujoco.mj_step(model, data)
    return _finish(model, data, task, epi, events, g_lo, g_hi, has_excursion,
                   attached, render, out_dir, mujoco, variant=variant,
                   build_defect=build_defect,
                   max_settle_steps=T * steps_per_frame)

def _body_rbound(model, bid):
    """Bounding radius of a body from the model's own geometry: the
    farthest geom surface point from the body origin (geom centre offset
    plus the geom's own bounding radius)."""
    r = 0.0
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == bid:
            r = max(r, float(np.linalg.norm(model.geom_pos[gid])) + float(model.geom_rbound[gid]))
    return r

def _body_at_rest(model, data, mujoco, bid):
    """One body's rest read, at the engine's own time resolution: linear
    speed and rim speed (angular speed x the body's bounding radius,
    from the model's own geometry) both below one integration step of
    gravitational velocity gain, |g| * dt, with g and dt read from the
    model. A body in free fall for one step reaches exactly |g| * dt; a
    settled body sits orders of magnitude below it. (The ring seat
    test's SETTLED clause, generalised to any body.) Returns
    (at_rest, lin_speed, rim_speed)."""
    vel6 = np.array(data.cvel[bid])
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
    lin_speed = float(np.linalg.norm(vel6[3:]))
    rim_speed = float(np.linalg.norm(vel6[:3])) * _body_rbound(model, bid)
    rest_bound = float(np.linalg.norm(model.opt.gravity) * model.opt.timestep)
    return (lin_speed < rest_bound and rim_speed < rest_bound), lin_speed, rim_speed

def _dynamic_object_bodies(model, mujoco):
    """Bodies whose rest state the settle waits on: every jointed body
    that is not part of the arm (fixtures are jointless and static; the
    arm is kinematic in this harness -- its pose is the recorded
    trajectory, not a physics outcome)."""
    out = []
    for bid in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if nm in ARM_BODIES:
            continue
        jadr = int(model.body_jntadr[bid])
        if jadr < 0:
            continue
        if int(model.jnt_type[jadr]) in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            out.append(bid)
    return out

def _bodies_at_rest(model, data, mujoco):
    """Engine rest read for the whole scene: every dynamic object body
    at rest per _body_at_rest. Vacuously true when a scene has no
    dynamic object bodies."""
    return all(_body_at_rest(model, data, mujoco, bid)[0]
               for bid in _dynamic_object_bodies(model, mujoco))

def _finish(model, data, task, epi, events, g_lo, g_hi, has_excursion, attached, render, out_dir, mujoco, variant=0, build_defect=None, max_settle_steps=None):
    # Settle until the physics engine reports rest (founder ruling
    # 2026-11-25; replaces the former fixed SETTLE_S = 1.0 s wait, whose
    # read could land mid-transient or long after rest). The step cap is
    # a termination safeguard only -- run_episode passes the episode's
    # own trajectory length in steps -- and the verdict never reads
    # elapsed settle time: evaluate() reads the state the engine
    # resolved, and settle_at_rest records whether rest was reached.
    if max_settle_steps is None:
        raise ValueError("max_settle_steps is required (the episode's own "
                         "trajectory length in steps; see run_episode)")
    mujoco.mj_forward(model, data)
    settle_steps = 0
    while settle_steps < max_settle_steps and not _bodies_at_rest(model, data, mujoco):
        mujoco.mj_step(model, data)
        settle_steps += 1
    mujoco.mj_forward(model, data)
    result = evaluate(model, data, task, epi, mujoco)
    result["settle_steps"] = settle_steps
    result["settle_at_rest"] = bool(_bodies_at_rest(model, data, mujoco))
    result["grasp_events"] = events
    result["gripper_norm"] = {"open_ref_deg": round(g_lo, 2), "closed_ref_deg": round(g_hi, 2),
                              "has_excursion": bool(has_excursion),
                              "still_attached_at_end": attached}
    result["episode"] = epi; result["task"] = task
    result["variant"] = variant; result["suite_version"] = P.SUITE_VERSION
    result["build_defect"] = build_defect
    out_dir = Path(out_dir) if out_dir else (P.TWIN / "validation" / "episodes")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{task}__ep{epi}_v{variant}.json").write_text(json.dumps(result, indent=1, default=str))
    if render:
        try:
            r = mujoco.Renderer(model, height=480, width=640)
            r.update_scene(data, camera="topcam")
            img = r.render()
            from PIL import Image
            Image.fromarray(img).save(out_dir / f"{task}__ep{epi}_v{variant}_top.png")
            r.close()
        except Exception as e:
            result["render_error"] = str(e)[:200]
    return result

def evaluate(model, data, task, epi, mujoco):
    """Pre-registered mechanical outcome criteria (preregistration, verbatim
    definitions; v1.1 fix F-01: the contact/support clauses are now actually
    tested via contacts_with() -- in v1 it was dead code and position proxies
    silently stood in for the criteria. Position quantities are still computed
    and recorded, but only as REPORTED diagnostics / the secondary criterion.)

    Per-task PRIMARY criterion (registered text -> mechanism):
    - block_stack: SUCCESS iff the upper cube is in CONTACT with the lower
      cube, SUPPORTED by it alone (contact pair upper<->lower present at
      settle end and the upper cube's contact set is exactly {lower
      cube}), and SETTLED (not device-held; at rest at the engine's own
      time resolution -- the ring seat test's settled clause applied to
      the upper cube). Scored as a physical state since 2026-11-25
      (founder ruling), replacing the former centre-within-footprint
      inset band (inset by 25%): the planar centre offset is still
      computed and recorded as a diagnostic only.
    - eye_drops_to_basket: SUCCESS iff carton centre inside the basket
      interior footprint AND the carton's resting support is the basket:
      contact with the basket body present, and NO contact with the platform
      body or the world (mat/floor/walls).
    - ring_insert: SUCCESS iff the ring is seated on the peg, scored as a
      physical state (founder ruling 2026-11-25, replacing the former
      height band -- twin-replay/ring-band-postmortem-20261125.md):
      (i) ENGAGED -- the peg passes through the ring hole: planar
      ring-to-peg distance <= physical radial clearance (RING_R_IN -
      PEG_R) and the ring's vertical extent overlaps the peg's span;
      (ii) SEAT SUPPORT -- at settle end the ring is in contact with the
      board or the peg (contact pair present, the same primary style as
      the other tasks); (iii) SETTLED -- the ring is not held by the
      grasp device (no active weld on the ring, no arm contact) and its
      speed is below one integration step of gravitational velocity gain
      (|v| and rim speed |omega| * RING_R_OUT both under |gravity| *
      timestep, read from the model) -- at rest at the engine's own time
      resolution. bottom_above_board_m is still computed and recorded as
      a diagnostic only; it no longer gates.
    - tool_insert: SUCCESS iff tool centre within the slot footprint
      (evaluated in the SLOT frame, yaw applied), tool axis within 6 deg of
      the slot axis (the measured end-wall lean cap of the real channel:
      a ~78 mm bar in the ~81 mm channel cannot tilt past ~6 deg, and
      seated rest angles measured on held-out episodes are <= 5.5 deg;
      the former 20 deg was an underived convention), and the tool is
      resting in the tray (contact with the tray body at settle).
    - cable_clip: SUCCESS iff a cable segment is seated in the clip with the
      clip CLOSED over it. Closure mechanism (documented modelling device):
      the clip is a fixed-geometry sprung clip -- two static jaw geoms
      (clip_jaw_a/b on the clip_block body) whose gap IS the channel; the
      jaws cannot articulate in this model, so closure is registered
      mechanically as RETENTION: at settle end (the engine's rest state,
      gripper released or not) a segment lies inside the channel volume between the
      jaws AND is in contact with the clip_block body -- i.e. the clip, not
      the gripper or the mat, is what holds that segment in the channel.
      A segment merely passing over the block, or resting on the mat beside
      it, does not register closure.
    Solver failure (cable, Amendment 1 A2): any cable segment ending outside
    the physical workspace envelope (see twin_params CABLE_* bounds) or with
    a non-finite position marks the variant solver_failure; such variants
    are excluded from agreement denominators by run_validation.py.
    """
    def pos(nm):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        return np.array(data.xpos[bid]), np.array(data.xquat[bid])
    def contacts_with(nm):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        out = set()
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1]); b2 = int(model.geom_bodyid[c.geom2])
            if b1 == bid or b2 == bid:
                other = b2 if b1 == bid else b1
                out.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other))
        return out
    WORLD = {None, "world"}   # body 0: mat/floor/wall geoms live on it
    poses = B.load_episode_poses(task, epi)
    fixtures = B.load_fixtures(task, poses)
    fin = (poses or {}).get("final", {})
    res = {"objects_final": {}, "checks": {}, "solver_failure": False}
    def planar_err(nm, *aliases):
        for a in (nm,) + aliases:
            if a in fin:
                p, _ = pos(nm)
                return round(float(np.linalg.norm(p[:2] - np.array(fin[a]["pos"][:2]))), 4)
        return None
    if task == "block_stack":
        pa, _ = pos("cube_a"); pb, _ = pos("cube_b")
        upper, lower = (pa, pb) if pa[2] >= pb[2] else (pb, pa)
        uname = "cube_a" if pa[2] >= pb[2] else "cube_b"
        lname = "cube_b" if uname == "cube_a" else "cube_a"
        e = P.CUBE_EDGE
        dz = (upper[2] - e / 2) - (lower[2] + e / 2)   # diagnostic only
        # PRIMARY (founder ruling 2026-11-25): the stack is scored as a
        # physical state, mirroring the ring seat test -- CONTACT with
        # the lower cube, SUPPORT from it alone, and SETTLED (not
        # device-held, at rest at the engine's own time resolution).
        # The former footprint inset band (centre within the lower top
        # footprint inset by 25%) is abolished: a physically stacked
        # cube whose centre overhangs the inset but rests on the lower
        # cube alone is a stack, and a mid-transient contact is not.
        # The planar centre offset is recorded as a diagnostic only.
        ucontacts = contacts_with(uname)
        contact_lower = lname in ucontacts
        no_other_support = ucontacts <= {lname}
        held = bool(ucontacts & set(ARM_BODIES))
        for eq in range(model.neq):
            eq_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
            if eq_name and eq_name.startswith(f"weld_{uname}") and data.eq_active[eq]:
                held = True
        uid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, uname)
        at_rest, lin_speed, rim_speed = _body_at_rest(model, data, mujoco, uid)
        settled = bool(not held and at_rest)
        success = bool(contact_lower and no_other_support and settled)
        res["checks"] = {"upper": uname, "face_gap_m": round(float(dz), 4),
                         "centre_offset_xy_m": [round(float(upper[0] - lower[0]), 4),
                                                round(float(upper[1] - lower[1]), 4)],
                         "contact_with_lower": bool(contact_lower),
                         "no_other_support_contact": bool(no_other_support),
                         "held_by_device": held,
                         "settled": settled,
                         "upper_speed_m_s": round(lin_speed, 6),
                         "upper_rim_speed_m_s": round(rim_speed, 6),
                         "upper_contacts": sorted(str(x) for x in ucontacts)}
        res["objects_final"] = {"cube_a": pa.tolist(), "cube_b": pb.tolist()}
        res["planar_err_m"] = planar_err(uname, "cube_a", "cube_b")
    elif task == "eye_drops_to_basket":
        pc, _ = pos("carton")
        bk = fixtures["basket"]["pos"]
        inside = (abs(pc[0] - bk[0]) <= P.BASKET_OUTER[0] / 2 - P.BASKET_WALL_T and
                  abs(pc[1] - bk[1]) <= P.BASKET_OUTER[1] / 2 - P.BASKET_WALL_T)
        supp = contacts_with("carton")
        supported_by_basket = "basket" in supp
        on_platform = "platform" in supp
        on_mat = bool(supp & WORLD)
        success = bool(inside and supported_by_basket and not on_platform and not on_mat)
        res["checks"] = {"centre_inside_basket": bool(inside),
                         "support_is_basket": bool(supported_by_basket),
                         "contact_platform": bool(on_platform),
                         "contact_mat_or_floor": bool(on_mat),
                         "carton_contacts": sorted(str(x) for x in supp),
                         "carton_z": round(float(pc[2]), 4)}
        res["objects_final"] = {"carton": pc.tolist()}
        res["planar_err_m"] = planar_err("carton", "box", "eye_drops")
    elif task == "ring_insert":
        pr, qr = pos("ring")
        # peg reference = the peg body's actual position in THIS scene
        # (v1.1: variant scenes no longer jitter the peg/board -- the outcome
        # reference stays at nominal per Amendment 1; reading the sim body
        # remains correct for the nominal scene.)
        try:
            bp = np.array(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")])
        except Exception:
            bp = fixtures.get("peg", {}).get("pos", fixtures["board"]["pos"])
        bt = P.BOARD_H
        planar = float(np.linalg.norm(pr[:2] - np.array(bp[:2])))
        clearance = P.RING_R_IN - P.PEG_R   # physical radial clearance, 3 mm
        engaged = planar <= clearance
        # ENGAGED, vertical clause (founder ruling 2026-11-25): the peg
        # must actually pass through the hole, so the ring's vertical
        # extent must overlap the peg's span [board top, peg top]. Ring
        # z half-extent from its own built geometry (build_scenes.py):
        # mid-radius x sin(tilt) + tube radius.
        axis = B.qrot(qr, np.array([0, 0, 1.0]))
        rmid = (P.RING_R_OUT + P.RING_R_IN) / 2
        tube = min((P.RING_R_OUT - P.RING_R_IN) / 2, P.RING_THICK / 2)
        half_ext = rmid * math.sqrt(max(0.0, 1.0 - float(axis[2]) ** 2)) + tube
        peg_lo = bp[2] + bt
        peg_hi = peg_lo + P.PEG_H
        engaged = bool(engaged and (pr[2] - half_ext) < peg_hi and (pr[2] + half_ext) > peg_lo)
        # SEAT SUPPORT: contact with the board or the peg at settle end
        # (contact-pair presence -- the same primary style as the other
        # tasks' criteria; replaces the former height band, whose edge a
        # settled seat cannot reach: band-edge probe 2026-11-24).
        rcontacts = contacts_with("ring")
        seat_support = bool(rcontacts & {"board", "peg"})
        # SETTLED: not held by the grasp device -- no active weld on the
        # ring (the device of physics-parameters.md s.7; replay.py's own
        # weld discovery is by the weld_<obj> name prefix) and no arm
        # contact -- and at rest at the engine's own time resolution:
        # speed below one integration step of gravitational velocity
        # gain, |g| * dt, with g and dt read from the model. A body in
        # free fall for one step reaches exactly |g| * dt; a settled
        # body sits orders of magnitude below it.
        held = bool(rcontacts & set(ARM_BODIES))
        for eq in range(model.neq):
            eq_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
            if eq_name and eq_name.startswith("weld_ring") and data.eq_active[eq]:
                held = True
        rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ring")
        vel6 = np.array(data.cvel[rid])
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, rid, vel6, 0)
        lin_speed = float(np.linalg.norm(vel6[3:]))
        rim_speed = float(np.linalg.norm(vel6[:3])) * P.RING_R_OUT
        rest_bound = float(np.linalg.norm(model.opt.gravity) * model.opt.timestep)
        settled = bool(not held and lin_speed < rest_bound and rim_speed < rest_bound)
        success = bool(engaged and seat_support and settled)
        # diagnostic only (no longer gating): the former band's quantity
        ring_bottom = pr[2] - P.RING_THICK / 2
        res["checks"] = {"planar_to_peg_m": round(planar, 4), "engaged": bool(engaged),
                         "hole_clearance_m": round(float(clearance), 4),
                         "seat_support_contact": seat_support,
                         "ring_contacts": sorted(str(x) for x in rcontacts),
                         "held_by_device": held,
                         "settled": settled,
                         "ring_speed_m_s": round(lin_speed, 6),
                         "ring_rim_speed_m_s": round(rim_speed, 6),
                         "rest_speed_bound_m_s": round(rest_bound, 6),
                         "bottom_above_board_m": round(float(ring_bottom - bt), 4)}
        res["objects_final"] = {"ring": pr.tolist()}
        res["planar_err_m"] = None if engaged else planar_err("ring")
        res["ring_engaged"] = bool(engaged)
    elif task == "tool_insert":
        pt, qt = pos("tool")
        sp = fixtures["slot"]["pos"]; syaw = math.radians(fixtures["slot"].get("yaw_deg", 0.0))
        axis = B.qrot(qt, np.array([1.0, 0, 0]))
        slot_axis = np.array([math.cos(syaw), math.sin(syaw), 0])
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(axis, slot_axis))))))
        # centre-in-slot evaluated in the slot frame (v1 tested world axes)
        d = np.array(pt[:2]) - np.array(sp[:2])
        lx = math.cos(-syaw) * d[0] - math.sin(-syaw) * d[1]
        ly = math.sin(-syaw) * d[0] + math.cos(-syaw) * d[1]
        within = abs(lx) <= P.SLOT_SIZE[0] / 2 and abs(ly) <= P.SLOT_SIZE[1] / 2
        resting_in_tray = "tray" in contacts_with("tool")
        success = bool(within and ang <= 6.0 and resting_in_tray)
        res["checks"] = {"centre_in_slot": bool(within), "axis_angle_deg": round(ang, 1),
                         "resting_in_tray_contact": bool(resting_in_tray),
                         "tool_z": round(float(pt[2]), 4)}
        res["objects_final"] = {"tool": pt.tolist()}
        res["planar_err_m"] = planar_err("tool", "loose_tool")
    elif task == "cable_clip":
        # v1.1 fixpass FP-A (V11-01): clip reference = the clip_block body's
        # ACTUAL pose in this sim scene (mirrors ring_insert reading the peg
        # from the sim body), so scoring always uses actual scene geometry.
        try:
            cbv, cbq = pos("clip_block")
            cb = cbv; yaw = 2 * math.atan2(cbq[3], cbq[0])
        except Exception:
            cbp = (poses or {}).get("objects", {}).get("clip_block")
            if cbp is not None:
                cb = cbp["pos"]; yaw = 2 * math.atan2(cbp["quat"][3], cbp["quat"][0])
            else:
                cb = fixtures["clip_block"]["pos"]; yaw = math.radians(fixtures["clip_block"].get("yaw_deg", 0.0))
        chan_z = P.BLOCK_SIZE[2] + 0.006
        seated, nearest = False, 1e9
        seg_pos = {}
        exploded, max_radial = False, 0.0
        for i in range(P.CABLE_N):
            ps, _ = pos(f"cable_seg_{i}")
            seg_pos[f"seg_{i}"] = [round(float(x), 4) for x in ps]
            if not np.all(np.isfinite(ps)):
                exploded = True; continue
            radial = float(np.linalg.norm(ps[:2]))
            max_radial = max(max_radial, radial)
            if radial > P.CABLE_MAX_RADIAL_M or not (P.CABLE_Z_MIN_M <= ps[2] <= P.CABLE_Z_MAX_M):
                exploded = True
            d = float(np.linalg.norm(ps[:2] - np.array(cb[:2])))
            nearest = min(nearest, d)
            local = B.qrot(B.yaw_quat(-math.degrees(yaw)), ps - np.array([cb[0], cb[1], 0]))
            in_channel = (abs(local[0]) <= P.CABLE_CHAN_X_HALF_M and abs(local[1]) <= P.CLIP_CHANNEL_W / 2 + P.CABLE_CHAN_Y_EXTRA_M
                          and abs(ps[2] - chan_z) <= P.CABLE_CHAN_Z_HALF_M)
            if in_channel and "clip_block" in contacts_with(f"cable_seg_{i}"):
                seated = True
        res["solver_failure"] = bool(exploded)
        success = bool(seated and not exploded)
        res["checks"] = {"cable_seated_in_clip_closed": seated,
                         "solver_failure": bool(exploded),
                         "max_seg_radial_m": round(max_radial, 3),
                         "nearest_seg_to_block_xy_m": round(nearest, 4)}
        res["objects_final"] = seg_pos
        res["planar_err_m"] = None
    # ---- Secondary criterion (F-02): thresholded, reported per episode ----
    # Registered thresholds: planar err <= 3.0 cm (block/eye/tool);
    # ring <= 2.0 cm when NOT engaged (the engagement boolean governs when
    # engaged); cable exempt (deformable -- outcome class only).
    if task in ("block_stack", "eye_drops_to_basket", "tool_insert"):
        thr = P.SECONDARY_THRESH_M[task]
        res["secondary_pass"] = (res["planar_err_m"] is not None and res["planar_err_m"] <= thr)
    elif task == "ring_insert":
        thr = P.SECONDARY_THRESH_M[task]
        res["secondary_pass"] = True if res.get("ring_engaged") else (
            res["planar_err_m"] is not None and res["planar_err_m"] <= thr)
    else:
        thr = None
        res["secondary_pass"] = None
    res["secondary_threshold_m"] = thr
    res["sim_success"] = bool(success)
    return res

if __name__ == "__main__":
    task, epi = sys.argv[1], int(sys.argv[2])
    var = int(sys.argv[sys.argv.index("--variant") + 1]) if "--variant" in sys.argv else 0
    r = run_episode(task, epi, render="--render" in sys.argv, variant=var)
    print(json.dumps({k: r[k] for k in ("task", "episode", "sim_success", "checks", "planar_err_m")}, default=str))
