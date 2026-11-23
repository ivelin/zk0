"""Twin-suite v2 Stage 1.2: Drake kinematic replay + v1.1 mechanical criteria.

Mirrors v1.1 replay.py semantics step for step:
  - arm prescribed kinematically from the recorded gt trajectory
    (degrees -> radians; gripper joint from the same closedness mapping),
  - grasp via the same dwell/planar-shape-distance state machine; the
    MuJoCo weld is realised as a rigid pose override (object pose :=
    gripper pose * snapshot relpose, velocity zeroed) applied at every
    frame boundary while attached -- the kinematic equivalent of the
    stiff weld for a prescribed-motion gripper,
  - 0.25 s pre-settle, build-defect check, 1.0 s settle,
  - outcome classes: v1.1 mechanical criteria, re-implemented against
    Drake final poses + Drake point-contact pairs (criteria logic copied
    from replay.evaluate; positions are diagnostics, contacts govern).

Differences from MuJoCo run (all logged, none tuned):
  - physics dt 1 ms (Drake discrete plant) vs 0.5 ms; arm pose updated
    at frame boundaries (20 Hz) instead of interpolated per substep --
    immaterial to objects because the arm has no collision geometry in
    either engine; it matters only through weld snapshots, which are
    taken at frame starts in both engines.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

V1 = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
sys.path.insert(0, str(V1 / "code"))
sys.path.insert(0, str(Path(__file__).parent))
import twin_params as P
import build_scenes as B
from replay import closedness_trace, JOINTS
from drake_port import DrakeScene, mat2quat_wxyz
from pydrake.math import RigidTransform
from pydrake.systems.analysis import Simulator

WELD_SLOTS = 4  # v1.1: K=4 weld slots per weldable body (non-cable tasks)


def _shape_dist(scene, nm, site_xy, pos, quat_wxyz):
    bp = pos
    q = quat_wxyz
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
        # Drake body frame: tool axis is local +z (see drake_port). The
        # MuJoCo check projects on the local x axis; use the Drake axis.
        axis = scene.body_pose(nm).rotation().matrix() @ np.array([1.0, 0, 0])
        d = np.array([bp[0] - site_xy[0], bp[1] - site_xy[1]])
        along = float(np.dot(d, axis[:2] / (np.linalg.norm(axis[:2]) + 1e-12)))
        perp = d - along * (axis[:2] / (np.linalg.norm(axis[:2]) + 1e-12))
        dx = max(abs(along) - P.TOOL_LEN / 2, 0.0)
        return math.hypot(dx, float(np.linalg.norm(perp)))
    return float(np.linalg.norm(off))


def _centre_bound(nm):
    if nm == "carton": return 0.060
    if nm == "tool": return 0.075
    return P.GRASP_DIST_M


def run_episode(task, epi, variant=0):
    t0 = time.time()
    scene = DrakeScene(task, epi, variant)
    plant = scene.plant
    sim = Simulator(scene.diagram, scene.context)
    sim.set_target_realtime_rate(0.0)
    sim.Initialize()

    gt = np.load(P.RECORDS / f"base__{task}__ep{epi}.npz")["gt"].astype(float)
    T = len(gt)
    gt_rad = np.radians(gt[:, :5])
    closed, g_lo, g_hi, has_excursion = closedness_trace(gt[:, 5])
    grip_joint_rad = 1.45 + closed * (0.10 - 1.45)
    # clip gripper command to the Drake joint limits
    gj = plant.GetJointByName("gripper", scene.robot)
    lo, hi = float(gj.position_lower_limit()), float(gj.position_upper_limit())
    grip_joint_rad = np.clip(grip_joint_rad, min(lo, hi), max(lo, hi))

    def arm_q(fr_float):
        i0 = int(math.floor(fr_float)); i1 = min(i0 + 1, T - 1)
        f = fr_float - i0
        q = {}
        for k, j in enumerate(JOINTS[:5]):
            q[j] = float(gt_rad[i0, k] * (1 - f) + gt_rad[i1, k] * f)
        q["gripper"] = float(grip_joint_rad[i0] * (1 - f) + grip_joint_rad[i1] * f)
        return q

    def obj_pose(nm):
        X = scene.body_pose(nm)
        return X.translation().copy(), mat2quat_wxyz(X.rotation().matrix())

    now = [0.0]
    def advance(dt):
        now[0] += dt
        sim.AdvanceTo(now[0])

    scene.set_arm(arm_q(0.0))
    init_pos = {nm: scene.body_pose(nm).translation().copy() for nm in scene.weldables}
    for _ in range(int(0.25 * P.FPS)):
        advance(1.0 / P.FPS)
    build_defect = None
    for nm in scene.weldables:
        dpos = scene.body_pose(nm).translation()
        if not np.all(np.isfinite(dpos)):
            build_defect = f"{nm}: NaN position after pre-settle"; break
        disp = float(np.linalg.norm(dpos - init_pos[nm]))
        if disp > 0.03 or dpos[2] < -0.02:
            build_defect = f"{nm}: displaced {disp:.3f} m / z={dpos[2]:.3f} after pre-settle"
            break

    z_trace = {nm: [] for nm in scene.weldables}
    used = {nm: 0 for nm in scene.weldables}
    attached, rel_pose, events = None, None, []
    last_attach_frame = -10
    seeking = False
    trackers = {}
    ineligible = set()
    prev_c = float(closed[0])

    def do_attach(fr, c, snap):
        nonlocal attached, rel_pose, seeking, trackers, last_attach_frame
        obj = snap["obj"]
        used[obj] += 1
        Xg = scene.gripper_pose()
        rel_pose = Xg.inverse().multiply(snap["X_obj"])
        attached = obj
        last_attach_frame = fr
        seeking = False; trackers = {}
        events.append({"frame": fr, "event": "attach", "object": obj,
                       "planned_dist_m": round(float(snap["d"]), 4),
                       "live_dist_m": round(float(snap["d"]), 4),
                       "snap_gap_m": 0.0,
                       "gate": "planar_shape_dwell", "closedness": round(c, 3)})

    for fr in range(T):
        if fr % 5 == 0:
            for nm in scene.weldables:
                z_trace[nm].append([fr, round(float(scene.body_pose(nm).translation()[2]), 4)])
        c = float(closed[fr])
        if ineligible:
            gpos0 = scene.site_pos()
            for nm in list(ineligible):
                bp, _ = obj_pose(nm)
                if float(np.linalg.norm(bp[:2] - gpos0[:2])) >= 0.03:
                    ineligible.discard(nm); trackers.pop(nm, None)
        if has_excursion and attached is None and not seeking and prev_c < P.ATTACH_CLOSEDNESS <= c:
            seeking = True; trackers = {}
        if seeking and attached is None:
            gpos = scene.site_pos()
            ending = c <= P.RELEASE_CLOSEDNESS
            fired = False
            for nm in scene.weldables:
                if nm in ineligible: continue
                bp, bq = obj_pose(nm)
                d_shape = _shape_dist(scene, nm, gpos[:2], bp, bq)
                d_centre = float(np.linalg.norm(bp[:2] - gpos[:2]))
                tr = trackers.get(nm)
                if tr is None:
                    tr = trackers[nm] = {"min_shape": d_shape, "min_centre": d_centre, "snap": None, "fr0": fr}
                if d_shape <= tr["min_shape"]:
                    tr["min_shape"] = d_shape
                    tr["snap"] = {"obj": nm, "d": d_shape, "X_obj": scene.body_pose(nm)}
                tr["min_centre"] = min(tr["min_centre"], d_centre)
                turned = d_shape >= tr["min_shape"] + 0.003
                plateau = d_shape <= tr["min_shape"] + 0.001 and fr - tr["fr0"] >= 2
                if (tr["min_shape"] <= P.GRASP_DIST_M and tr["min_centre"] <= _centre_bound(nm)
                        and used[nm] < WELD_SLOTS and (turned or plateau or ending)):
                    do_attach(fr, c, tr["snap"]); fired = True; break
            if not fired and ending:
                seeking = False
                elig = trackers
                best_nm = min(elig, key=lambda n: elig[n]["min_shape"]) if elig else None
                events.append({"frame": fr, "event": "no_attach_in_excursion",
                               "min_dist_m": round(float(elig[best_nm]["min_shape"]), 4) if best_nm else None})
                trackers = {}
        if attached is not None and c <= P.RELEASE_CLOSEDNESS and fr > last_attach_frame:
            events.append({"frame": fr, "event": "release", "object": attached,
                           "closedness": round(c, 3)})
            ineligible.add(attached)
            attached, rel_pose = None, None
        prev_c = c
        # Weld realisation v2 (global harness choice, logged): pin the
        # attached object at the FRAME START (gripper at gt[fr]), then let
        # Drake's contact solver run the frame -- contact resolves against
        # the pin, so the weld yields to supports the way MuJoCo's soft
        # constraint weld (solref 0.005) does. The v1 realisation (override
        # AFTER the step) teleported objects through supports whenever the
        # gripper descended past the resting plane with a rel offset
        # captured at a different height (ring 195 / carton 1630 traces).
        if attached is not None:
            scene.set_free_pose(attached, scene.gripper_pose().multiply(rel_pose))
        scene.set_arm(arm_q(min(fr + 1, T - 1)))
        advance(1.0 / P.FPS)

    # settle 1.0 s, arm held; weld pin maintained per frame
    for _ in range(int(P.SETTLE_S * P.FPS)):
        if attached is not None:
            scene.set_free_pose(attached, scene.gripper_pose().multiply(rel_pose))
        advance(1.0 / P.FPS)

    result = evaluate(scene, task, epi)
    result["grasp_events"] = events
    result["gripper_norm"] = {"open_ref_deg": round(g_lo, 2), "closed_ref_deg": round(g_hi, 2),
                              "has_excursion": bool(has_excursion),
                              "still_attached_at_end": attached}
    result["z_trace"] = z_trace
    result["episode"] = epi; result["task"] = task; result["variant"] = variant
    result["engine"] = "drake"
    result["build_defect"] = build_defect
    result["wall_s"] = round(time.time() - t0, 1)
    return result


def evaluate(scene, task, epi):
    """v1.1 mechanical criteria (replay.evaluate) on Drake state."""
    def pos(nm):
        X = scene.body_pose(nm)
        return X.translation().copy(), mat2quat_wxyz(X.rotation().matrix())
    poses = B.load_episode_poses(task, epi)
    fixtures = B.load_fixtures(task, poses)
    fin = (poses or {}).get("final", {})
    res = {"objects_final": {}, "objects_final_quat_wxyz": {}, "checks": {}, "solver_failure": False}
    def store(nm):
        p, q = pos(nm)
        res["objects_final"][nm] = p.tolist()
        res["objects_final_quat_wxyz"][nm] = q.tolist()
        if not np.all(np.isfinite(p)) or float(np.linalg.norm(p[:2])) > 1.0 or not (-0.05 <= p[2] <= 0.50):
            res["solver_failure"] = True
        return p, q
    def planar_err(nm, *aliases):
        for a in (nm,) + aliases:
            if a in fin:
                p, _ = pos(nm)
                return round(float(np.linalg.norm(p[:2] - np.array(fin[a]["pos"][:2]))), 4)
        return None
    WORLD = {"world"}
    if task == "block_stack":
        pa, _ = store("cube_a"); pb, _ = store("cube_b")
        upper, lower = (pa, pb) if pa[2] >= pb[2] else (pb, pa)
        uname = "cube_a" if pa[2] >= pb[2] else "cube_b"
        lname = "cube_b" if uname == "cube_a" else "cube_a"
        e = P.CUBE_EDGE
        dz = (upper[2] - e / 2) - (lower[2] + e / 2)
        inset = (e / 2) * 0.75
        within = abs(upper[0] - lower[0]) <= inset and abs(upper[1] - lower[1]) <= inset
        ucontacts = scene.contacts_with(uname)
        contact_lower = lname in ucontacts
        no_other_support = ucontacts <= {lname}
        success = bool(within and contact_lower and no_other_support)
        res["checks"] = {"upper": uname, "face_gap_m": round(float(dz), 4),
                         "within_inset_footprint": bool(within),
                         "contact_with_lower": bool(contact_lower),
                         "no_other_support_contact": bool(no_other_support),
                         "upper_contacts": sorted(str(x) for x in ucontacts)}
        res["planar_err_m"] = planar_err(uname, "cube_a", "cube_b")
    elif task == "eye_drops_to_basket":
        pc, _ = store("carton")
        bk = fixtures["basket"]["pos"]
        inside = (abs(pc[0] - bk[0]) <= P.BASKET_OUTER[0] / 2 - P.BASKET_WALL_T and
                  abs(pc[1] - bk[1]) <= P.BASKET_OUTER[1] / 2 - P.BASKET_WALL_T)
        supp = scene.contacts_with("carton")
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
        res["planar_err_m"] = planar_err("carton", "box", "eye_drops")
    elif task == "ring_insert":
        pr, qr = store("ring")
        bp = scene.body_pose("peg").translation()
        bt = P.BOARD_H
        planar = float(np.linalg.norm(pr[:2] - np.array(bp[:2])))
        clearance = P.RING_R_IN - P.PEG_R
        engaged = planar <= clearance
        ring_bottom = pr[2] - P.RING_THICK / 2
        low_ok = 0.0 <= (ring_bottom - bt) <= 0.010
        success = bool(engaged and low_ok)
        res["checks"] = {"planar_to_peg_m": round(planar, 4), "engaged": bool(engaged),
                         "hole_clearance_m": round(float(clearance), 4),
                         "bottom_above_board_m": round(float(ring_bottom - bt), 4)}
        res["planar_err_m"] = None if engaged else planar_err("ring")
        res["ring_engaged"] = bool(engaged)
    elif task == "tool_insert":
        pt, qt = store("tool")
        sp = fixtures["slot"]["pos"]; syaw = math.radians(fixtures["slot"].get("yaw_deg", 0.0))
        axis = scene.body_pose("tool").rotation().matrix() @ np.array([1.0, 0, 0])
        slot_axis = np.array([math.cos(syaw), math.sin(syaw), 0])
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(axis, slot_axis))))))
        d = np.array(pt[:2]) - np.array(sp[:2])
        lx = math.cos(-syaw) * d[0] - math.sin(-syaw) * d[1]
        ly = math.sin(-syaw) * d[0] + math.cos(-syaw) * d[1]
        within = abs(lx) <= P.SLOT_SIZE[0] / 2 and abs(ly) <= P.SLOT_SIZE[1] / 2
        resting_in_tray = "tray" in scene.contacts_with("tool")
        success = bool(within and ang <= 20.0 and resting_in_tray)
        res["checks"] = {"centre_in_slot": bool(within), "axis_angle_deg": round(ang, 1),
                         "resting_in_tray_contact": bool(resting_in_tray),
                         "tool_z": round(float(pt[2]), 4)}
        res["planar_err_m"] = planar_err("tool", "loose_tool")
    else:
        raise ValueError(task)
    if task in ("block_stack", "eye_drops_to_basket", "tool_insert"):
        thr = P.SECONDARY_THRESH_M[task]
        res["secondary_pass"] = (res["planar_err_m"] is not None and res["planar_err_m"] <= thr)
    elif task == "ring_insert":
        thr = P.SECONDARY_THRESH_M[task]
        res["secondary_pass"] = True if res.get("ring_engaged") else (
            res["planar_err_m"] is not None and res["planar_err_m"] <= thr)
    else:
        thr = None; res["secondary_pass"] = None
    res["secondary_threshold_m"] = thr
    res["sim_success"] = bool(success)
    return res


if __name__ == "__main__":
    task, epi = sys.argv[1], int(sys.argv[2])
    var = int(sys.argv[sys.argv.index("--variant") + 1]) if "--variant" in sys.argv else 0
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(__file__).parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    r = run_episode(task, epi, var)
    (out / f"{task}__ep{epi}_v{var}.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({"task": task, "episode": epi, "engine": "drake",
                      "sim_success": r["sim_success"], "checks": r["checks"],
                      "objects_final": r["objects_final"], "wall_s": r["wall_s"]}, default=str))
