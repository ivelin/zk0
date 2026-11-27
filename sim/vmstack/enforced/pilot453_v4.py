"""Closed-loop teleoperation pilot for block_stack ep453 on arm v4
(forcerange-capped, 2026-11-27 corrections lane) + folded-home start,
under protocol SSOT SS2A (interactive teleoperation) + SS2B (driver
freedom). Scene: scenes-v4/block_stack__ep453_v0__v4cam.xml (sha256
9bf24a99...); manipulated object cube_b, base cube_a.

GRIP DRIVE (2026-11-27, grip_drive.py gripdrive/1.0-20261127): the
gripper joint is driven by its position actuator with the derived
forcerange cap (+/-1.9123 N m) in the loop — the trace's grip value is
a COMMAND written to ctrl every substep; the servo tracks, saturates
and yields under load (the real ep453 servo yielded 10.26 deg). See
the campaign log SESSION CLOSE + COMPLIANT GRIP DRIVE sections.

FULL ACTUATION — P5 (2026-11-27, actuation-corrections lane): the five
arm joints are now driven exactly like the gripper — the trace's arm
values are servo COMMANDS written to ctrl every substep and the engine
integrates; set_arm_pilot never writes qpos/qvel. Before this change
the arm was teleported (qpos written, qvel zeroed, ctrl[0..4] left at
the reset default 0), which (a) left the STS3215 actuators pulling
toward q=0 on every substep and (b) gave the jaws zero velocity, so
friction could never carry an object upward — lift in certification
mode was mathematically zero (neon-defect-verification-20261127/
VERDICT.md, Defects 2+3 CONFIRMED; the verifier's actuated
counterfactual lifts cube_b +46.58 mm at the real servo's yield).
Init places the arm at the folded_home keyframe ONCE (reset semantics,
as before) and starts every servo at its frame-0 target; from there
the engine integrates. The trace still stores commanded values;
enforce_replay (same drive model) interprets them identically.
summary() reports joint_rad (commanded) and joint_actual_rad /
grip_yield_deg (engine state).

Stepping replicates enforce_replay.run exactly: per 20 FPS frame, 100
substeps at 2000 Hz, set_arm interpolating the commanded trace. The
enforcement scan mirrors enforce_replay.enforcement_scan: any contact
involving an arm body outside the declared set is a fault -- declared:
jaw-assembly x cube_b, jaw x jaw, chain-adjacent arm x arm. On a fault
the pre-command snapshot is restored (teleoperator back-off) and the
fault reported. Grasp bookkeeping mirrors enforce_replay certification
mode (contact-true only; weld unreachable): attach at closedness >=
twin_params.ATTACH_CLOSEDNESS with engine jaw x cube_b contact for 3
consecutive frames; release at closedness <= RELEASE_CLOSEDNESS.

Init resets to the scene's folded_home keyframe (founder folded-home
standard, 2026-11-27). Orientation R_mode is the driver's choice:
default REAL453_CLOSE (site rotation recorded from the real ep453
grasp, calibration-20261127/real_grasp_fk.json); `setR` switches among
real453 / real411 / ring; `yaw` rotates R_mode about world z.

Live feed: after every command, top/front/wrist renders + status.json
are written to validator-sets-20261123/live-watch/ (founder watches).

EXPORT converts the joint trace to the harness grip channel
(chan = 55 - closedness*53, closedness = (1.45-joint)/1.35).

Commands (stdin, one per line; JSON reply + '>>>'):
  init                        reset to folded_home + settle
  state                       numeric summary
  phase <text...>             set live-feed phase label
  note <text...>              set live-feed note text
  setR real453|real411|ring   choose site orientation family
  move dx dy dz frames [joint]  site increment (world m), R_mode fixed
  goto x y z frames [joint]   site to absolute world position
  sitefor cx cy cz vx vy vz frames [joint]
                              site to (centre - R_mode @ V_mm)
  grip joint frames           ramp gripper joint (rad) only
  hold frames
  yaw deg frames              rotate R_mode about world z in place
  render label                named stills in OUT (live feed always on)
  settle                      1.0 s hold + replay.evaluate read (preview)
  export path                 write gt npz (piloted trace)
  quit
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).parent
_WORK = _HERE.parent.parent
sys.path.insert(0, str(_WORK))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import twin_params as P
P.TWIN = _WORK / "staging"
P.RECORDS = _WORK / "staging" / "records"
import mujoco
import grip_drive
from arm_fk import ArmFK, ik_site_full, JOINTS

JOINTS5 = JOINTS[:5]
SCENE = str(_HERE / "scenes-v4" / "block_stack__ep453_v0__v4cam.xml")
OUT = _HERE / "pilot_ep453_v4"
OUT.mkdir(exist_ok=True)
LIVE = _WORK.parent / "live-watch"
LIVE.mkdir(exist_ok=True)
FPS = P.FPS
SPF = max(1, int(round((1.0 / FPS) / P.TIMESTEP)))
ATTACH_C = P.ATTACH_CLOSEDNESS      # 0.50 in twin_params (mirror enforce_replay)
RELEASE_C = P.RELEASE_CLOSEDNESS    # 0.30
DWELL = 3

model = mujoco.MjModel.from_xml_path(SCENE)
model.opt.timestep = P.TIMESTEP
ACT_G = grip_drive.configure(model)   # compliant grip drive (see header)
ACT_ARM = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS5]
data = mujoco.MjData(model)
KEYFRAME = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "folded_home")

ARM_BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist",
              "gripper", "moving_jaw_so101_v1"]
JAW_BODIES = ["gripper", "moving_jaw_so101_v1"]
ALL_JOINTS = JOINTS5 + ["gripper"]
qadr = {j: int(model.joint(j).qposadr[0]) for j in ALL_JOINTS}
vadr = {j: int(model.joint(j).dofadr[0]) for j in ALL_JOINTS}
names = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
         for g in range(model.ngeom)}
arm_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in ARM_BODIES}
jaw_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in JAW_BODIES}
jaw_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in jaw_bids}
obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_b")    # manipulated
base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")   # base
obj_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == obj_bid}
base_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == base_bid}
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
chain = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in ARM_BODIES]
adjacent = {frozenset((chain[i], chain[i + 1])) for i in range(len(chain) - 1)}

fk = ArmFK(str(_WORK / "refetch" / "arm.xml"))
_gt_ring = np.load(str(_WORK / "refetch" / "ep7.npz"))["gt"]
_pf = int(np.argmin(_gt_ring[:, 5]))
Q_TMPL = {n: float(_gt_ring[_pf, i]) for i, n in enumerate(JOINTS5)}
R_RING = fk.site_pose(Q_TMPL)[:3, :3]

_fkrec = json.loads((_HERE / "calibration-20261127" / "real_grasp_fk.json").read_text())


def _R_from_axes(rec):
    return np.column_stack([rec["site_x_world"], rec["site_y_world"], rec["site_z_world"]])


R_LIBRARY = {
    "real453": _R_from_axes(_fkrec["episodes"]["453"]["configs"]["close_complete"]),
    "real411": _R_from_axes(_fkrec["episodes"]["411"]["configs"]["close_complete"]),
    "ring": R_RING,
}

# session state
trace = []            # list of (q5_rad np.array(5), gripper_joint_rad float)
R_mode = R_LIBRARY["real453"].copy()
R_name = "real453"
attached = False
dwell = 0
loss_run = 0
events = []
phase = "boot"
note_text = ""
increment = 0
state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
state_buf = np.empty(mujoco.mj_stateSize(model, state_spec))
_renderer = None


def rotz(deg):
    th = math.radians(deg)
    return np.array([[math.cos(th), -math.sin(th), 0],
                     [math.sin(th), math.cos(th), 0], [0, 0, 1]])


def set_arm_pilot(frame_float):
    T = len(trace)
    i0 = int(math.floor(frame_float))
    i1 = min(i0 + 1, T - 1)
    f = frame_float - i0
    q0, j0 = trace[i0]
    q1, j1 = trace[i1]
    # P5 full actuation: every trace value (arm + grip) is a servo
    # COMMAND written to ctrl; the engine integrates. qpos/qvel are
    # never written here — no teleport, no velocity zeroing.
    for k, a in enumerate(ACT_ARM):
        data.ctrl[a] = q0[k] * (1 - f) + q1[k] * f
    data.ctrl[ACT_G] = j0 * (1 - f) + j1 * f


def scan(frame_idx):
    """Mirror of enforce_replay.enforcement_scan: fault dict or None."""
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if b1 not in arm_bids and b2 not in arm_bids:
            continue
        jaw1, jaw2 = b1 in jaw_bids, b2 in jaw_bids
        obj1, obj2 = b1 == obj_bid, b2 == obj_bid
        if (jaw1 and obj2) or (jaw2 and obj1) or (jaw1 and jaw2):
            continue
        if b1 in arm_bids and b2 in arm_bids and frozenset((b1, b2)) in adjacent:
            continue
        return {"frame": frame_idx, "geom1": names[g1], "geom2": names[g2],
                "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
                "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2),
                "dist_m": float(c.dist)}
    return None


def jaw_contact():
    pairs = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in jaw_geoms and g2 in obj_geoms) or (g2 in jaw_geoms and g1 in obj_geoms):
            pairs.append((names[g1], names[g2], float(c.dist)))
    return pairs


def grip_to_base_min_mm():
    fromto = np.zeros(6)
    best = None
    for gg in jaw_geoms:
        for bg in base_geoms:
            d = float(mujoco.mj_geomDistance(model, data, gg, bg, 0.5, fromto))
            best = d if best is None else min(best, d)
    return round(best * 1000, 2) if best is not None else None


FIXED_GEOMS = [g for g in range(model.ngeom)
               if int(model.geom_bodyid[g]) == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")]
MOVING_GEOMS = [g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")]


def jaw_dist_by_body():
    """geomDistance (mm) from each jaw body to cube_b + closest geom name
    + world delta of the closest pair (cube point - jaw point)."""
    fromto = np.zeros(6)
    out = {}
    for label, geoms in (("fixed", FIXED_GEOMS), ("moving", MOVING_GEOMS)):
        best, bg, bvec = None, None, None
        for gg in geoms:
            for og in obj_geoms:
                d = float(mujoco.mj_geomDistance(model, data, gg, og, 0.5, fromto))
                if best is None or d < best:
                    best, bg = d, gg
                    bvec = (fromto[3:6] - fromto[0:3]).copy()
        out[f"jaw_dist_{label}_mm"] = round(best * 1000, 2) if best is not None else None
        out[f"jaw_dist_{label}_geom"] = names[bg] if bg is not None else None
        out[f"jaw_dist_{label}_dir"] = [round(float(x), 3) for x in bvec] if bvec is not None else None
    return out


def closedness_of(joint):
    return float(np.clip((1.45 - joint) / 1.35, 0, 1))


def summary(extra=None):
    mujoco.mj_forward(model, data)
    jc = jaw_contact()
    s = {"frames": len(trace),
         "site": [round(float(x), 4) for x in data.site_xpos[site_id]],
         "cube_b_manip": [round(float(x), 4) for x in data.xpos[obj_bid]],
         "cube_a_base": [round(float(x), 4) for x in data.xpos[base_bid]],
         "q_deg": [round(math.degrees(trace[-1][0][k]), 1) for k in range(5)] if trace else None,
         "joint_rad": round(float(trace[-1][1]), 3) if trace else None,
         "joint_actual_rad": round(float(data.qpos[qadr["gripper"]]), 4),
         "grip_yield_deg": round(math.degrees(float(data.qpos[qadr["gripper"]]) - trace[-1][1]), 3) if trace else None,
         "closedness": round(closedness_of(trace[-1][1]), 3) if trace else None,
         "R_mode": R_name,
         "attached": attached,
         "jaw_contact": bool(jc),
         "jaw_min_dist_mm": round(min(d for _, _, d in jc) * 1000, 2) if jc else None,
         "grip_to_cube_a_min_mm": grip_to_base_min_mm(),
         "events": events[-4:]}
    s.update(jaw_dist_by_body())
    if extra:
        s.update(extra)
    return s


def write_live(s):
    global _renderer
    try:
        if _renderer is None:
            _renderer = mujoco.Renderer(model, height=480, width=640)
        from PIL import Image
        for cam in ("topcam", "frontcam", "wristcam"):
            _renderer.update_scene(data, camera=cam)
            Image.fromarray(_renderer.render()).save(str(LIVE / f"view_{cam}.png"))
    except Exception as e:
        (LIVE / "render_error.txt").write_text(f"{type(e).__name__}: {e}")
    st = {"increment": increment, "phase": phase, "note": note_text,
          "updated_unix": round(time.time(), 1),
          "site": s.get("site"), "cube_b_manip": s.get("cube_b_manip"),
          "cube_a_base": s.get("cube_a_base"),
          "joint_rad": s.get("joint_rad"), "closedness": s.get("closedness"),
          "attached": s.get("attached"), "jaw_contact": s.get("jaw_contact"),
          "frames_in_trace": s.get("frames"),
          "last_fault": s.get("fault_rewound")}
    (LIVE / "status.json").write_text(json.dumps(st, indent=1))


def snapshot():
    mujoco.mj_getState(model, data, state_buf, state_spec)
    return (state_buf.copy(), len(trace), attached, dwell, loss_run, len(events))


def restore(snap):
    global attached, dwell, loss_run
    buf, ntr, attached, dwell, loss_run, nev = snap
    mujoco.mj_setState(model, data, buf, state_spec)
    del trace[ntr:]
    del events[nev:]


def step_new_frames(n_new):
    """Step the last n_new appended frames with harness semantics.
    Returns fault dict or None (state left wherever the fault hit)."""
    global attached, dwell, loss_run
    T = len(trace)
    start = T - n_new
    for fr in range(start, T):
        for ssub in range(SPF):
            frac = (ssub + 1) / SPF
            set_arm_pilot(min(fr + frac, T - 1))
            mujoco.mj_step(model, data)
            f = scan(fr)
            if f is not None:
                return f
        mujoco.mj_forward(model, data)
        jc = jaw_contact()
        c = closedness_of(trace[fr][1])
        if not attached:
            if c >= ATTACH_C and jc:
                dwell += 1
            else:
                dwell = 0
            if dwell >= DWELL:
                attached = True
                dwell = 0
                loss_run = 0
                events.append({"frame": fr, "type": "grasp_contact_true",
                               "closedness": round(c, 3)})
        else:
            if jc:
                loss_run = 0
            else:
                loss_run += 1
                if loss_run == 10:
                    events.append({"frame": fr, "type": "jaw_contact_lost_10_frames"})
            if c <= RELEASE_C:
                attached = False
                events.append({"frame": fr, "type": "release",
                               "closedness": round(c, 3)})
    return None


def append_interp(q_target_deg, joint_target, frames):
    q0 = {n: math.degrees(trace[-1][0][k]) for k, n in enumerate(JOINTS5)}
    j0 = trace[-1][1]
    new = []
    for i in range(1, frames + 1):
        f = i / frames
        s = f * f * (3 - 2 * f)
        q = np.array([math.radians(q0[n] + (q_target_deg[n] - q0[n]) * s)
                      for n in JOINTS5])
        new.append((q, j0 + (joint_target - j0) * s))
    trace.extend(new)
    return len(new)


def ik(tgt, Rq, seed_deg):
    q, pe, re = ik_site_full(fk, np.asarray(tgt, float), Rq, dict(seed_deg), iters=150)
    if pe > 0.003 or re > 0.26:
        for pan in (-60.0, -30.0, 0.0, 30.0, 60.0):
            for flex in (-40.0, 50.0):
                s = dict(seed_deg)
                s["shoulder_pan"] = pan
                s["wrist_flex"] = flex
                q2, pe2, re2 = ik_site_full(fk, np.asarray(tgt, float), Rq, s, iters=150)
                if pe2 + 0.05 * re2 < pe + 0.05 * re:
                    q, pe, re = q2, pe2, re2
    return q, pe, re


def cur_qdeg():
    return {n: math.degrees(trace[-1][0][k]) for k, n in enumerate(JOINTS5)}


def guarded(append_fn):
    """Snapshot BEFORE appending; on fault restore physics AND truncate
    the appended frames, so the trace only ever holds stepped-clean frames."""
    snap = snapshot()
    n = append_fn()
    fault = step_new_frames(n)
    if fault is not None:
        restore(snap)
        return summary({"fault_rewound": fault})
    return summary({"fault_rewound": None})


def cmd_init():
    global attached, dwell, loss_run, R_mode, R_name
    mujoco.mj_resetDataKeyframe(model, data, KEYFRAME)
    trace.clear()
    events.clear()
    attached, dwell, loss_run = False, 0, 0
    R_mode = R_LIBRARY["real453"].copy()
    R_name = "real453"
    q5 = np.array([float(data.qpos[qadr[j]]) for j in JOINTS5])
    gj = float(data.qpos[qadr["gripper"]])
    data.ctrl[ACT_G] = gj   # servo starts at the keyframe state
    for k, a in enumerate(ACT_ARM):
        data.ctrl[a] = q5[k]   # arm servos start at the keyframe state
    trace.append((q5, gj))
    set_arm_pilot(0.0)
    mujoco.mj_forward(model, data)
    fault = None
    for _ in range(int(0.25 / P.TIMESTEP)):
        set_arm_pilot(0.0)
        mujoco.mj_step(model, data)
        fault = scan(0)
        if fault:
            break
    return summary({"cmd": "init", "settle_fault": fault,
                    "start": "folded_home keyframe"})


def cmd_walk(tgt_site, joint_t, max_frames, stop_on_contact=True):
    """Frame-by-frame LINEAR interpolation toward an IK target, stopping
    early on jaw x cube_b contact (unless stop_on_contact is False), on
    cube_b moving > 1 mm, or on attach. One frame is appended + stepped
    at a time; a fault rewinds the walk."""
    global increment
    seed = cur_qdeg()
    q_fin, pe, re = ik(tgt_site, R_mode, seed)
    q_start = cur_qdeg()
    j_start = trace[-1][1]
    if joint_t is None:
        joint_t = j_start
    mujoco.mj_forward(model, data)
    cube0 = np.array(data.xpos[obj_bid], float)
    snap = snapshot()
    stepped, reason = 0, "completed"
    for i in range(1, max_frames + 1):
        f = i / max_frames
        q = np.array([math.radians(q_start[n] + (q_fin[n] - q_start[n]) * f)
                      for n in JOINTS5])
        trace.append((q, j_start + (joint_t - j_start) * f))
        fault = step_new_frames(1)
        if fault is not None:
            restore(snap)
            return summary({"fault_rewound": fault, "walk_stop": "fault"})
        stepped = i
        if attached:
            reason = "attached"
            break
        if stop_on_contact and jaw_contact():
            reason = "jaw_contact"
            break
        disp = float(np.linalg.norm(np.array(data.xpos[obj_bid], float) - cube0))
        if disp > 0.001:
            reason = f"cube_moved_{disp*1000:.2f}mm"
            break
    increment += 1
    mujoco.mj_forward(model, data)
    disp = float(np.linalg.norm(np.array(data.xpos[obj_bid], float) - cube0))
    return summary({"fault_rewound": None, "walk_frames": stepped,
                    "walk_stop": reason, "cube_disp_mm": round(disp * 1000, 2),
                    "ik_pos_err_mm": round(pe * 1000, 2),
                    "ik_rot_err_deg": round(math.degrees(re), 2),
                    "target_site": [round(float(x), 4) for x in tgt_site]})


def cmd_move_to(tgt, frames, joint=None):
    seed = cur_qdeg()
    q, pe, re = ik(tgt, R_mode, seed)
    jt = trace[-1][1] if joint is None else joint
    out = guarded(lambda: append_interp(q, jt, frames))
    out.update({"ik_pos_err_mm": round(pe * 1000, 2),
                "ik_rot_err_deg": round(math.degrees(re), 2),
                "target_site": [round(float(x), 4) for x in tgt]})
    return out


def main():
    global R_mode, R_name, phase, note_text, increment
    print(json.dumps({"pilot": "ready", "spf": SPF, "scene": SCENE,
                      "harness": "pilot453_v4.py (protocol v2 P1+P3+D2+P4+P5)",
                      "attach_c": ATTACH_C, "release_c": RELEASE_C,
                      "grip_drive": grip_drive.describe(model, ACT_G)}), flush=True)
    print(">>>", flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        cmd = parts[0]
        try:
            if cmd == "quit":
                break
            elif cmd == "init":
                out = cmd_init()
            elif cmd == "state":
                out = summary({"cmd": "state"})
            elif cmd == "phase":
                phase = " ".join(parts[1:])
                out = summary({"cmd": "phase", "phase": phase})
            elif cmd == "note":
                note_text = " ".join(parts[1:])
                out = summary({"cmd": "note"})
            elif cmd == "setR":
                name = parts[1]
                assert name in R_LIBRARY, f"unknown R {name}"
                mujoco.mj_forward(model, data)
                cur = np.array(data.site_xpos[site_id])
                seed = cur_qdeg()
                q, pe, re = ik(cur, R_LIBRARY[name], seed)
                snap = snapshot()
                n = append_interp(q, trace[-1][1], int(parts[2]) if len(parts) > 2 else 20)
                fault = step_new_frames(n)
                if fault is not None:
                    restore(snap)
                    out = summary({"fault_rewound": fault})
                else:
                    R_mode = R_LIBRARY[name].copy()
                    R_name = name
                    out = summary({"fault_rewound": None, "R_mode": name,
                                   "ik_pos_err_mm": round(pe * 1000, 2),
                                   "ik_rot_err_deg": round(math.degrees(re), 2)})
            elif cmd == "move":
                dx, dy, dz = (float(parts[1]), float(parts[2]), float(parts[3]))
                frames = int(parts[4])
                joint = float(parts[5]) if len(parts) > 5 else None
                mujoco.mj_forward(model, data)
                cur = np.array(data.site_xpos[site_id])
                out = cmd_move_to(cur + np.array([dx, dy, dz]), frames, joint)
                increment += 1
            elif cmd == "goto":
                tgt = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                frames = int(parts[4])
                joint = float(parts[5]) if len(parts) > 5 else None
                out = cmd_move_to(tgt, frames, joint)
                increment += 1
            elif cmd == "sitefor":
                c = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                V = np.array([float(parts[4]), float(parts[5]), float(parts[6])]) / 1000.0
                frames = int(parts[7])
                joint = float(parts[8]) if len(parts) > 8 else None
                out = cmd_move_to(c - R_mode @ V, frames, joint)
                increment += 1
            elif cmd == "grip":
                joint, frames = float(parts[1]), int(parts[2])
                out = guarded(lambda: append_interp(cur_qdeg(), joint, frames))
                increment += 1
            elif cmd == "walk":
                dx, dy, dz = (float(parts[1]), float(parts[2]), float(parts[3]))
                maxf = int(parts[4])
                joint = float(parts[5]) if len(parts) > 5 else None
                mujoco.mj_forward(model, data)
                cur = np.array(data.site_xpos[site_id])
                out = cmd_walk(cur + np.array([dx, dy, dz]), joint, maxf)
            elif cmd == "walkg":
                dg, maxf = float(parts[1]), int(parts[2])
                mujoco.mj_forward(model, data)
                cur = np.array(data.site_xpos[site_id])
                out = cmd_walk(cur, trace[-1][1] + dg, maxf)
            elif cmd == "walkx":
                dx, dy, dz = (float(parts[1]), float(parts[2]), float(parts[3]))
                maxf = int(parts[4])
                joint = float(parts[5]) if len(parts) > 5 else None
                mujoco.mj_forward(model, data)
                cur = np.array(data.site_xpos[site_id])
                out = cmd_walk(cur + np.array([dx, dy, dz]), joint, maxf,
                               stop_on_contact=False)
            elif cmd == "hold":
                frames = int(parts[1])
                out = guarded(lambda: append_interp(cur_qdeg(), trace[-1][1], frames))
                increment += 1
            elif cmd == "yaw":
                deg, frames = float(parts[1]), int(parts[2])
                mujoco.mj_forward(model, data)
                cur = np.array(data.site_xpos[site_id])
                R_new = rotz(deg) @ R_mode
                seed = cur_qdeg()
                q, pe, re = ik(cur, R_new, seed)
                snap = snapshot()
                n = append_interp(q, trace[-1][1], frames)
                fault = step_new_frames(n)
                if fault is not None:
                    restore(snap)
                    out = summary({"fault_rewound": fault})
                else:
                    R_mode = R_new
                    R_name = R_name + f"+yaw{deg:g}"
                    out = summary({"fault_rewound": None,
                                   "ik_pos_err_mm": round(pe * 1000, 2),
                                   "ik_rot_err_deg": round(math.degrees(re), 2)})
                increment += 1
            elif cmd == "render":
                label = parts[1] if len(parts) > 1 else "view"
                from PIL import Image
                rend = mujoco.Renderer(model, height=480, width=640)
                paths = {}
                for cam in ("topcam", "frontcam", "wristcam"):
                    rend.update_scene(data, camera=cam)
                    pth = OUT / f"{label}_{cam}.png"
                    Image.fromarray(rend.render()).save(str(pth))
                    paths[cam] = str(pth)
                rend.close()
                out = summary({"rendered": paths})
            elif cmd == "settle":
                snap = snapshot()
                fault = None
                for _ in range(int(P.SETTLE_S / P.TIMESTEP)):
                    set_arm_pilot(len(trace) - 1)
                    mujoco.mj_step(model, data)
                    fault = scan(len(trace) - 1)
                    if fault:
                        break
                if fault:
                    restore(snap)
                    out = summary({"settle_fault": fault})
                else:
                    import replay as _old
                    ev = _old.evaluate(model, data, "block_stack", 453, mujoco)
                    out = summary({"settle_fault": None,
                                   "sim_success": ev.get("sim_success"),
                                   "checks": ev.get("checks"),
                                   "objects_final": ev.get("objects_final")})
                    restore(snap)  # settle is a preview; trace state unchanged
            elif cmd == "export":
                gt = np.zeros((len(trace), 6))
                for i, (q, j) in enumerate(trace):
                    gt[i, :5] = np.degrees(q)
                    c = closedness_of(j)
                    gt[i, 5] = float(np.clip(55.0 - c * 53.0, 2.0, 55.0))
                np.savez(parts[1], gt=gt)
                lo, hi = np.percentile(gt[:, 5], 5), np.percentile(gt[:, 5], 99)
                out = {"exported": parts[1], "frames": len(trace),
                       "chan_p5": round(float(lo), 2), "chan_p99": round(float(hi), 2)}
            else:
                out = {"error": f"unknown cmd {cmd}"}
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        write_live(out if isinstance(out, dict) else {})
        print(json.dumps(out), flush=True)
        print(">>>", flush=True)


if __name__ == "__main__":
    main()
