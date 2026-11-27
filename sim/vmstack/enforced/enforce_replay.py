"""Enforced replay harness — protocol v2 P1 + P3 + D2 (2026-11-26).

Built against the finalized path-defect postmortem (2026-11-25/26) and the
founder's three sim-validity lessons (2026-11-26 00:22 CT). It mirrors
vmstack/replay.py's execution semantics (within-frame interpolation at
2000 Hz, per-episode gripper normalisation, 0.25 s pre-settle, 1.0 s
final settle, replay.evaluate as the final-state read; the DRIVE model
was kinematic at build time and is fully actuated since P5 below)
and differs in exactly the ways the postmortem's §6 mechanism requires:

P1 — NO GHOST ARM. replay.py sets contype=0/conaffinity=0 on every arm
geom at load ("arm cannot collide; grasp is via weld only"). This harness
never touches contype: the arm collides, contact signals exist, and the
loader ASSERTS every collision-class arm geom kept contype != 0.

P3 — CONTACT-TRUE ATTACH. replay.py's attach gate is planar-only (shape /
centre distance + dwell bookkeeping); gripper height is never evaluated,
which is how a weld snapshot locked with the gripper 19.8 mm below the
mat top (postmortem §2). Here a grasp registers only when the ENGINE
reports contact between jaw collision geoms and the object, sustained
across CONTACT_DWELL_FRAMES consecutive frames at attach closedness. The
weld (transport mode only) snapshots at that proven grasp, using
replay.py's own weld mechanics (model.eq_data write + MjData recreation,
MuJoCo 4.0.1 bindings). In CERTIFICATION mode the weld is unreachable —
grasp is the engine's friction grasp or it is nothing (postmortem §7b;
founder lesson: weld attach is not certification-grade).

D2 — IMPOSSIBILITY-BY-ABORT (the v2 draft's §3.2 option 1, adopted as the
working mechanism by the founder's 2026-11-26 build order). After every
physics step, engine contacts are scanned: any contact involving an arm
collision geom OUTSIDE the declared contact set is a hard execution
fault — execution aborts at that step and the fault is recorded. The
declared set is only:
  * jaw-assembly geoms (gripper + moving jaw bodies) x target object
  * jaw-assembly geoms x jaw-assembly geoms (jaws closing on each other)
  * arm body x its chain-ADJACENT arm body (base-shoulder, shoulder-
    upper_arm, upper_arm-lower_arm, lower_arm-wrist, wrist-gripper,
    gripper-moving_jaw): bolted assembly interfaces — the pan motor sits
    inside the base housing and the rendered meshes interpenetrate there
    identically (measured -26.4 mm on the v2 arm at a mid-approach pose),
    so these contacts are mechanism-internal, not task physics. Every
    such contact is still RECORDED in the run record (pair + worst
    depth), never silent. Non-adjacent arm x arm contact is a fault.
Through-table / through-board / through-wall paths are impossible to
EXECUTE under this harness, not merely flagged after the fact.

P4 — COMPLIANT GRIP DRIVE (2026-11-27, grip_drive.py,
gripdrive/1.0-20261127). The five arm joints are still imposed
kinematically (the recordings ARE arm joint trajectories), but the
gripper joint is driven by its position actuator with the derived
forcerange cap (+/-1.9123 N m, STS3215 C001 stall) IN the loop: the
commanded grip value is written to ctrl every substep and the servo
tracks / saturates / yields under contact load, as the real servo did
(recorded yield 10.26 deg, ep453). Before this change the gripper was
qpos-imposed like the arm, the cap was bypassed, a gentle touch stored
~49.9 N on one cube corner, and every press/lift detonated the cube —
the compliance the real grasp depended on was structurally absent
(campaign log SESSION CLOSE 2026-11-27). Seal implication: the harness
element of the (scene, arm, engine, harness) tuple changed; tuples
citing the pre-P4 harness are superseded for future work.

P5 — FULL ACTUATION (2026-11-27, actuation-corrections lane). The five
arm joints are now driven exactly like the gripper: the trajectory's
arm values are servo COMMANDS written to data.ctrl every substep (the
arm model's own STS3215 position actuators, kp 18.45701 / kv 2.73117,
forcerange capped at the C001 stall) and the engine integrates —
set_arm never writes qpos/qvel. At run start the arm + gripper are
placed at the frame-0 command ONCE (the old drive's frame-0 state;
the verifier's cell-(iii) init semantics) and every servo starts at
its frame-0 target. Why: the teleport drive (qpos written, qvel
zeroed every substep, ctrl[0..4] never written) was measured to
(a) leave the arm actuators pulling toward q=0 on every substep
(~6 g of phantom site acceleration) and (b) give the jaws zero
velocity, so friction could never carry an object upward — lift in
certification mode was mathematically zero even at a 28 N-per-jaw
pinch (neon-defect-verification-20261127/VERDICT.md, Defects 2+3
CONFIRMED; actuated counterfactual: pinch yield +11.51 deg vs the
real 10.26 deg, lift +46.58 mm, zero slip). The real arm is
servo-driven; the recordings are its commanded joint trajectories,
and replay fidelity now means the same commands through the same
class of servo, not imposed positions. Frame pacing is unchanged
(100 substeps per 20 FPS frame, linear interpolation of the ctrl
targets between frames — the exact stepping under which the
verifier's cell (iii) reproduced the recorded outcomes). Seal
implication: the harness element changed again; tuples citing the
pre-P5 harness are superseded for future work.

ALARM LAYER (instrumentation only; certifies nothing): the path_check /
alarm_canary baseline rule over this run's own frames — baseline =
frame-0 gripper-vs-static signed distances; ALARM if a penetration pair
absent at frame 0 occurs mid-task or any later frame goes below baseline.
Plus the C2 trace: attached-phase object-vs-support engine contact with
the object's in-contact planar travel.

Usage (mirrors vm_run_one.py):
  enforce_replay.py <task> <epi> <variant> <scene_xml> <npz> <out_json>
                    [--mode=transport|certification] [--object=BODY]
The scene XML must include a v2 arm (arm_padded_v2.xml / arm_tool_v2.xml);
make_v2_scene() derives one from a staging scene. twin_params paths
(P.TWIN / P.RECORDS) are patched by the caller, exactly as in vm_run_one.
"""
import json, math, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))
import twin_params as P
# VM-lane path patch BEFORE importing build_scenes (it binds SCENES from
# P.TWIN at import and reads calibration then) — vm_run_one.py's pattern.
# Only applied when twin_params' DGX defaults don't exist on this machine.
if not P.TWIN.exists():
    _STAGE = _HERE.parent.parent / "staging"
    P.TWIN = _STAGE
    P.RECORDS = _STAGE / "records"
import build_scenes as B
sys.path.insert(0, str(_HERE))
import grip_drive

ARM_BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist",
              "gripper", "moving_jaw_so101_v1"]
JAW_BODIES = ["gripper", "moving_jaw_so101_v1"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Grasp detector parameter of THIS harness (not a scoring threshold): the
# engine-reported jaw-object contact must persist this many consecutive
# trajectory frames at attach closedness before a grasp registers —
# longer than one frame of contact flicker, shorter than any real dwell
# in the recorded demos (replay.py's own plateau minimum is 2 frames).
CONTACT_DWELL_FRAMES = 3


def make_v2_scene(staging_scene, out_dir):
    """Copy a staging scene, pointing its arm include at the v2 arm."""
    txt = Path(staging_scene).read_text()
    root = _HERE.parent.parent
    txt = txt.replace(str(root / "refetch" / "arm_tool.xml"), str(_HERE / "arm_tool_v2.xml"))
    txt = txt.replace(str(root / "refetch" / "arm.xml"), str(_HERE / "arm_padded_v2.xml"))
    out = Path(out_dir) / (Path(staging_scene).stem + "__v2.xml")
    out.write_text(txt)
    return out


class EnforcementAbort(Exception):
    def __init__(self, fault):
        self.fault = fault


def run(task, epi, variant, scene_xml, traj_npz, out_json, mode="certification",
        obj_name=None):
    import mujoco
    assert mode in ("transport", "certification")
    gt = np.load(traj_npz)["gt"].astype(float)
    T = len(gt)
    gt_rad = np.radians(gt[:, :5])
    closed, g_lo, g_hi, chan_ok = _closedness_trace(gt[:, 5])
    grip_joint_rad = 1.45 + closed * (0.10 - 1.45)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.timestep = P.TIMESTEP
    act_g = grip_drive.configure(model)   # P4: compliant grip drive
    # P5: arm actuators, resolved by joint name (actuator names match).
    act_arm = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
               for j in JOINTS[:5]]
    data = mujoco.MjData(model)

    # weld slots discovered exactly as replay.py discovers them
    weld_slots = {}
    for eq in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq)
        if nm and nm.startswith("weld_"):
            weld_slots.setdefault(nm[5:].rsplit("_", 1)[0], []).append(eq)
    for v in weld_slots.values():
        v.sort()
    if obj_name is None:
        assert len(weld_slots) == 1, f"multi-object scene {list(weld_slots)}: pass --object"
        obj_name = next(iter(weld_slots))

    arm_bids = set()
    for nm in ARM_BODIES:
        try:
            arm_bids.add(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm))
        except Exception:
            pass
    names = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
             for g in range(model.ngeom)}
    arm_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bids]
    # P1 proof: no ghost override — every collision-class arm geom collides
    ghost = [names[g] for g in arm_geoms
             if int(model.geom_group[g]) == 3 and int(model.geom_contype[g]) == 0]
    assert not ghost, f"P1 VIOLATION: collision geoms with contype=0 on arm: {ghost}"
    jaw_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in JAW_BODIES}
    jaw_geoms = {g for g in arm_geoms if int(model.geom_bodyid[g]) in jaw_bids}
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    obj_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == obj_bid}
    static_geoms = set()
    for gid in range(model.ngeom):
        b = int(model.geom_bodyid[gid])
        if b == 0:
            static_geoms.add(gid)
        elif b in arm_bids or b == obj_bid:
            continue
        elif int(model.body_jntnum[b]) == 0:
            static_geoms.add(gid)
    grip_geoms = [g for g in arm_geoms if int(model.geom_bodyid[g]) in jaw_bids]

    qadr = {j: int(model.joint(j).qposadr[0]) for j in JOINTS}
    vadr = {j: int(model.joint(j).dofadr[0]) for j in JOINTS}
    grip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    steps_per_frame = max(1, int(round((1.0 / P.FPS) / P.TIMESTEP)))
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_buf = np.empty(mujoco.mj_stateSize(model, state_spec))

    def set_arm(frame_float):
        i0 = int(math.floor(frame_float))
        i1 = min(i0 + 1, T - 1)
        f = frame_float - i0
        # P5: every trajectory value (arm + grip) is a servo COMMAND
        # written to ctrl; the engine integrates. qpos/qvel are never
        # written here — no teleport, no velocity zeroing.
        for k, a in enumerate(act_arm):
            data.ctrl[a] = gt_rad[i0, k] * (1 - f) + gt_rad[i1, k] * f
        data.ctrl[act_g] = grip_joint_rad[i0] * (1 - f) + grip_joint_rad[i1] * f

    def jaw_object_contact():
        pairs = []
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in jaw_geoms and g2 in obj_geoms) or (g2 in jaw_geoms and g1 in obj_geoms):
                pairs.append((names[g1], names[g2], float(c.dist)))
        return pairs

    chain = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) for nm in ARM_BODIES]
    adjacent = {frozenset((chain[i], chain[i + 1])) for i in range(len(chain) - 1)}
    self_contacts = {}

    def enforcement_scan(frame_idx, phase):
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
                k = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
                     mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2))
                if k not in self_contacts or float(c.dist) < self_contacts[k]:
                    self_contacts[k] = float(c.dist)
                continue
            raise EnforcementAbort({
                "frame": frame_idx, "phase": phase,
                "geom1": names[g1], "geom2": names[g2],
                "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
                "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2),
                "dist_m": float(c.dist)})

    fromto = np.zeros(6)

    def frame_row(frame_idx, attached):
        mujoco.mj_forward(model, data)
        grip_pairs = {}
        for gg in grip_geoms:
            for sg in static_geoms:
                d = float(mujoco.mj_geomDistance(model, data, gg, sg, 0.5, fromto))
                if d < 0.01:
                    k = names[sg]
                    grip_pairs[k] = min(grip_pairs.get(k, 1e9), d)
        obj_contact_names = set()
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if obj_bid in (b1, b2) and float(c.dist) <= 0:
                other = c.geom2 if b1 == obj_bid else c.geom1
                if int(other) in static_geoms:
                    obj_contact_names.add(names[int(other)])
        return {"frame": frame_idx,
                "grip_static_pairs_m": {k: round(v, 5) for k, v in grip_pairs.items()},
                "grip_min_m": round(min(grip_pairs.values()), 5) if grip_pairs else None,
                "site_z": round(float(data.site_xpos[site_id][2]), 5),
                "obj_pos": [round(float(x), 5) for x in data.xpos[obj_bid]],
                "obj_static_contacts": sorted(obj_contact_names),
                "attached": bool(attached)}

    rec = {"task": task, "epi": epi, "variant": variant, "mode": mode,
           "harness": "enforce_replay.py (protocol v2 P1+P3+D2+P4+P5)",
           "grip_drive": grip_drive.describe(model, act_g),
           "scene": str(scene_xml), "trajectory": str(traj_npz),
           "object": obj_name, "n_frames": T,
           "grip_channel_ok": bool(chan_ok),
           "grip_norm": {"open_ref_deg": round(g_lo, 2), "closed_ref_deg": round(g_hi, 2)},
           "weld_reachable": mode == "transport", "weld_ever_active": False,
           "grasp_events": [], "enforcement_fault": None, "aborted": False,
           "frames": []}
    used_weld = 0
    try:
        mujoco.mj_resetData(model, data)
        # P5: place the arm + gripper at the frame-0 command ONCE (the
        # old drive's frame-0 state; the verifier's cell-(iii) init) so
        # the run starts from the same configuration, with every servo
        # started at its frame-0 target via set_arm(0.0). From here on
        # the engine integrates — nothing writes qpos/qvel again.
        for k, j in enumerate(JOINTS[:5]):
            data.qpos[qadr[j]] = gt_rad[0, k]
        data.qpos[qadr["gripper"]] = grip_joint_rad[0]
        data.qvel[:] = 0.0
        set_arm(0.0)
        mujoco.mj_forward(model, data)
        rec["frames"].append(frame_row(0, False))   # alarm baseline row
        for _ in range(int(0.25 / P.TIMESTEP)):
            set_arm(0.0)
            mujoco.mj_step(model, data)
            enforcement_scan(0, "settle")
        attached = False
        dwell = 0
        contact_loss_run = 0
        ineligible_until_clear = False
        prev_obj_xy = None
        c2_travel = 0.0
        c2_contact_frames = 0
        for fr in range(T):
            c = float(closed[fr])
            for ssub in range(steps_per_frame):
                frac = (ssub + 1) / steps_per_frame
                set_arm(min(fr + frac, T - 1))
                mujoco.mj_step(model, data)
                enforcement_scan(fr, "attached" if attached else "free")
            mujoco.mj_forward(model, data)
            jc = jaw_object_contact()
            if ineligible_until_clear:
                site_xy = np.array(data.site_xpos[site_id])[:2]
                if float(np.linalg.norm(np.array(data.xpos[obj_bid])[:2] - site_xy)) >= 0.03:
                    ineligible_until_clear = False
            if not attached:
                if chan_ok and c >= P.ATTACH_CLOSEDNESS and jc and not ineligible_until_clear:
                    dwell += 1
                else:
                    dwell = 0
                if dwell >= CONTACT_DWELL_FRAMES:
                    attached = True
                    dwell = 0
                    contact_loss_run = 0
                    ev = {"frame": fr, "type": "grasp_contact_true",
                          "closedness": round(c, 4),
                          "jaw_object_contacts": sorted({(a, b) for a, b, _ in jc}),
                          "min_contact_dist_m": round(min(d for _, _, d in jc), 6)}
                    if mode == "transport" and used_weld < len(weld_slots[obj_name]):
                        eq = weld_slots[obj_name][used_weld]
                        used_weld += 1
                        gp = np.array(data.xpos[grip_bid]); gq = np.array(data.xquat[grip_bid])
                        op = np.array(data.xpos[obj_bid]); oq = np.array(data.xquat[obj_bid])
                        row = model.eq_data[eq]
                        row[0:3] = 0.0
                        row[3:6] = B.qrot(B.qinv(gq), op - gp)
                        row[6:10] = B.qmul(B.qinv(gq), oq)
                        mujoco.mj_getState(model, data, state_buf, state_spec)
                        data = mujoco.MjData(model)
                        mujoco.mj_setState(model, data, state_buf, state_spec)
                        data.eq_active[eq] = 1
                        rec["weld_ever_active"] = True
                        rec["_active_eq"] = eq
                        ev["weld"] = "activated_at_contact_true_grasp"
                    rec["grasp_events"].append(ev)
            else:
                if jc:
                    contact_loss_run = 0
                else:
                    contact_loss_run += 1
                    if contact_loss_run == 10:
                        rec["grasp_events"].append(
                            {"frame": fr, "type": "jaw_contact_lost_10_frames"})
                if c <= P.RELEASE_CLOSEDNESS:
                    attached = False
                    if mode == "transport" and "_active_eq" in rec:
                        data.eq_active[rec.pop("_active_eq")] = 0
                    ineligible_until_clear = True
                    rec["grasp_events"].append(
                        {"frame": fr, "type": "release", "closedness": round(c, 4)})
            row = frame_row(fr, attached)
            if attached:
                if row["obj_static_contacts"]:
                    c2_contact_frames += 1
                    if prev_obj_xy is not None:
                        c2_travel += float(np.linalg.norm(np.array(row["obj_pos"][:2]) - prev_obj_xy))
                prev_obj_xy = np.array(row["obj_pos"][:2])
            else:
                prev_obj_xy = None
            rec["frames"].append(row)
        # final settle 1.0 s, arm held (replay._finish semantics), enforced
        for _ in range(int(P.SETTLE_S / P.TIMESTEP)):
            set_arm(T - 1)
            mujoco.mj_step(model, data)
            enforcement_scan(T - 1, "final_settle")
        mujoco.mj_forward(model, data)
        rec["c2_trace"] = {"attached_phase_obj_support_contact_frames": c2_contact_frames,
                           "attached_phase_in_contact_planar_travel_m": round(c2_travel, 5)}
    except EnforcementAbort as e:
        rec["aborted"] = True
        rec["enforcement_fault"] = e.fault
        rec["frames_completed"] = len(rec["frames"])
    rec.pop("_active_eq", None)
    rec["arm_self_contacts_adjacent"] = {f"{a}x{b}": round(v, 5)
                                         for (a, b), v in self_contacts.items()}
    if rec["frames"]:
        base = rec["frames"][0]["grip_static_pairs_m"]
        alarm_pairs = {}
        for row in rec["frames"][1:]:
            for k, v in row["grip_static_pairs_m"].items():
                if v < 0 and (k not in base or v < base[k]):
                    if k not in alarm_pairs or v < alarm_pairs[k][1]:
                        alarm_pairs[k] = (row["frame"], v)
        rec["alarm"] = {"baseline_frame0": base,
                        "alarm_fired": bool(alarm_pairs),
                        "alarm_pairs": {k: {"first_frame": f0, "worst_m": v}
                                        for k, (f0, v) in alarm_pairs.items()}}
    if not rec["aborted"]:
        try:
            import replay as _old
            ev = _old.evaluate(model, data, task, epi, mujoco)
            rec["final_eval_instrumentation"] = {
                "sim_success_criteria_read": ev.get("sim_success"),
                "objects_final": ev.get("objects_final"),
                "checks": ev.get("checks")}
        except Exception as e:
            rec["final_eval_instrumentation"] = {"error": f"{type(e).__name__}: {e}"}
        rec["final_obj_pos"] = [round(float(x), 5) for x in data.xpos[obj_bid]]
    Path(out_json).write_text(json.dumps(rec, indent=1))
    return rec


def _closedness_trace(grip_deg):
    """replay.closedness_trace, verbatim semantics (LOW = jaws closed)."""
    g = np.asarray(grip_deg, float)
    lo, hi = np.percentile(g, 5), np.percentile(g, 99)
    if hi - lo < P.GRIP_MIN_EXCURSION_DEG:
        return np.zeros_like(g), float(lo), float(hi), False
    return np.clip((hi - g) / (hi - lo), 0, 1), float(lo), float(hi), True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode, obj = "certification", None
    for a in sys.argv[1:]:
        if a.startswith("--mode="):
            mode = a.split("=", 1)[1]
        if a.startswith("--object="):
            obj = a.split("=", 1)[1]
    r = run(args[0], int(args[1]), args[2], args[3], args[4], args[5], mode=mode, obj_name=obj)
    print(json.dumps({"aborted": r["aborted"], "fault": r["enforcement_fault"],
                      "alarm_fired": (r.get("alarm") or {}).get("alarm_fired"),
                      "grasp_events": r["grasp_events"],
                      "weld_ever_active": r["weld_ever_active"]}, indent=1))
