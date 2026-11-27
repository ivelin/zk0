"""Scene admission gate — protocol v2 P4 (2026-11-26), as code.

Founder lesson (2026-11-26 ruling, Muse's additions): a scene is admitted
only after a rest-state/settle audit — no interpenetration at spawn,
objects stable under gravity. Method = the path-defect postmortem's §5
admission census (admission_census.py), hardened into a gate:

  RAW SPAWN (mj_resetData + mj_forward, no stepping):
    - object-vs-static signed distances (mj_geomDistance)
    - engine contacts involving the object with dist < 0
    - gripper-vs-static minimum distance (arm home baseline, reported)
  POST-SETTLE (arm pinned at the trajectory's frame 0, 0.25 s of steps —
  the harness's own pre-roll):
    - same object measures + object displacement from raw spawn

VERDICT: FLAG if the object interpenetrates any static geom deeper than
PEN_GATE_M at raw spawn or after the settle (a scene that starts
interpenetrated and relies on the settle to push the object out is a
defective scene, not a hard scene). Scope (founder ruling 2026-11-26
08:22 CT): this gate is an admission INSTRUMENT over engine state — it
certifies nothing. Solution validity comes only from enforced
end-to-end replay plus frontier/human judging; the gate's one numeric
line is the measurement-derived census classification below. The arm home-pose overlap is REPORTED
as its own class (trajectory-level defect — enforcement catches it at
execution time); it does not change the scene verdict.

PEN_GATE_M = 0.002 — derivation (measured, from the postmortem census
itself, admission_census.json): resting-contact solver penetration on
the census's clean scenes never exceeds 0.97 mm (ring resting in its
board recess, ring_ep11); every defective scene measures >= 4.8 mm
post-settle and >= 7.11 mm at raw spawn. The 2 mm line sits inside the
measured empty band (0.97 .. 4.8 mm); it is a classification line over
the census distribution, not a physics claim.

Known-answer calibration (postmortem §5.3): the gate MUST flag the five
spawn-defective scenes — ring ep48 (ring|peg -7.11 mm at raw spawn),
tool ep697 (tool|wall_left -9.13 mm), tool ep622 / ep885 / ep852
(tool|tray_geom -10.0 mm) — and admit the other thirteen top-up scenes.

Usage:
  admission_gate.py --all                 # the 18 top-up scenes, staging
  admission_gate.py <scene_xml> <object> [--traj NPZ]
"""
import json, math, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).parent
_WORK = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))
import twin_params as P
_STAGE = _WORK / "staging"
P.TWIN = _STAGE
P.RECORDS = _STAGE / "records"
import mujoco  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
GRIP_BODIES = ["gripper", "moving_jaw_so101_v1"]
fromto = np.zeros(6)

SCENES_18 = [
    ("ring_ep48",    "ring_insert__ep48_v0.xml",    "ring_insert__ep48_a1.npz", "ring"),
    ("ring_ep7",     "ring_insert__ep7_v0.xml",     "ring_insert__ep7_a1.npz", "ring"),
    ("ring_ep11",    "ring_insert__ep11_v0.xml",    "ring_insert__ep11_a1.npz", "ring"),
    ("ring_ep31",    "ring_insert__ep31_v0.xml",    "ring_insert__ep31_a1.npz", "ring"),
    ("ring_ep160",   "ring_insert__ep160_v0.xml",   "ring_insert__ep160_a1.npz", "ring"),
    ("ring_ep165",   "ring_insert__ep165_v0.xml",   "ring_insert__ep165_a1.npz", "ring"),
    ("ring_s1_spec00", "ring_insert__ep91000_v0.xml", "ring_insert__s1_spec00_v0_a1.npz", "ring"),
    ("ring_s1_spec01", "ring_insert__ep91001_v0.xml", "ring_insert__s1_spec01_v0_a1.npz", "ring"),
    ("ring_s1_spec02", "ring_insert__ep91002_v0.xml", "ring_insert__s1_spec02_v0_a4.npz", "ring"),
    ("ring_s1_spec03", "ring_insert__ep91003_v0.xml", "ring_insert__s1_spec03_v0_a1.npz", "ring"),
    ("ring_s2_spec00", "ring_insert__ep91020_v0.xml", "ring_insert__s2_spec00_v0_a1.npz", "ring"),
    ("ring_s2_spec04", "ring_insert__ep91024_v0.xml", "ring_insert__s2_spec04_v0_a1.npz", "ring"),
    ("tool_ep697",   "tool_insert__ep697_v0.xml",   "tool_insert__ep697_a1.npz", "tool"),
    ("tool_ep622",   "tool_insert__ep622_v0.xml",   "tool_insert__ep622_a5.npz", "tool"),
    ("tool_ep885",   "tool_insert__ep885_v0.xml",   "tool_insert__ep885_a3.npz", "tool"),
    ("tool_ep852",   "tool_insert__ep852_v0.xml",   "tool_insert__ep852_a1.npz", "tool"),
    ("eye_ep1639",   "eye_drops_to_basket__ep1639_v0.xml", "eye_drops_to_basket__ep1639_a1.npz", "carton"),
    ("eye_ep1643",   "eye_drops_to_basket__ep1643_v0.xml", "eye_drops_to_basket__ep1643_a1.npz", "carton"),
]
KNOWN_FLAG = {"ring_ep48", "tool_ep697", "tool_ep622", "tool_ep885", "tool_ep852"}
PEN_GATE_M = 0.002   # derivation in the module docstring (census empty band)


def _measure(model, data, obj_bid, grip_geoms, obj_geoms, static_geoms, names):
    mujoco.mj_forward(model, data)
    obj_pairs = {}
    for og in obj_geoms:
        for sg in static_geoms:
            d = float(mujoco.mj_geomDistance(model, data, og, sg, 0.5, fromto))
            if d < 0.001:
                k = names[sg]
                obj_pairs[k] = min(obj_pairs.get(k, 1e9), d)
    grip_min = (1e9, None)
    for gg in grip_geoms:
        for sg in static_geoms:
            d = float(mujoco.mj_geomDistance(model, data, gg, sg, 0.5, fromto))
            if d < grip_min[0]:
                grip_min = (d, names[sg])
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if obj_bid in (b1, b2) and float(c.dist) < 0:
            other = c.geom2 if b1 == obj_bid else c.geom1
            contacts.append({"geom": names[int(other)], "dist_m": round(float(c.dist), 5)})
    return {"obj_static_pairs_m": {k: round(v, 5) for k, v in sorted(obj_pairs.items())},
            "obj_contact_penetrations": contacts,
            "grip_min_dist_m": round(grip_min[0], 5), "grip_min_pair": grip_min[1],
            "obj_pos": [round(float(x), 5) for x in data.xpos[obj_bid]]}


def audit_scene(scene_xml, obj_name, traj_npz=None):
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    model.opt.timestep = P.TIMESTEP
    data = mujoco.MjData(model)
    bid = lambda nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
    obj_bid = bid(obj_name)
    arm_ids = set()
    for nm in ARM_BODIES:
        try:
            arm_ids.add(bid(nm))
        except Exception:
            pass
    names = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
             for g in range(model.ngeom)}
    static_geoms, obj_geoms, grip_geoms = [], [], []
    grip_bids = {bid(nm) for nm in GRIP_BODIES}
    for gid in range(model.ngeom):
        b = int(model.geom_bodyid[gid])
        if b == obj_bid:
            obj_geoms.append(gid)
        if b in grip_bids:
            grip_geoms.append(gid)
        if b == 0:
            static_geoms.append(gid)
        elif b in arm_ids or b == obj_bid:
            continue
        elif int(model.body_jntnum[b]) == 0:
            static_geoms.append(gid)
    mujoco.mj_resetData(model, data)
    raw = _measure(model, data, obj_bid, grip_geoms, obj_geoms, static_geoms, names)
    settled, shift = None, None
    if traj_npz is not None:
        gt = np.load(traj_npz)["gt"].astype(float)
        g = gt[0]
        lo, hi = np.percentile(gt[:, 5], 5), np.percentile(gt[:, 5], 99)
        closed0 = float(np.clip((hi - g[5]) / (hi - lo), 0, 1)) if hi - lo >= P.GRIP_MIN_EXCURSION_DEG else 0.0
        qadr = {j: int(model.joint(j).qposadr[0]) for j in JOINTS}

        def set_arm0():
            for k, j in enumerate(JOINTS[:5]):
                data.qpos[qadr[j]] = math.radians(float(g[k]))
            data.qpos[qadr["gripper"]] = 1.45 + closed0 * (0.10 - 1.45)

        mujoco.mj_resetData(model, data)
        set_arm0()
        mujoco.mj_forward(model, data)
        for _ in range(int(0.25 / P.TIMESTEP)):
            set_arm0()
            mujoco.mj_step(model, data)
        settled = _measure(model, data, obj_bid, grip_geoms, obj_geoms, static_geoms, names)
        shift = round(float(np.linalg.norm(np.array(settled["obj_pos"]) - np.array(raw["obj_pos"]))), 5)
    reasons = []
    raw_pen = {k: v for k, v in raw["obj_static_pairs_m"].items() if v < -PEN_GATE_M}
    if raw_pen:
        reasons.append(f"raw spawn interpenetration beyond gate: {raw_pen}")
    if settled is not None:
        set_pen = {k: v for k, v in settled["obj_static_pairs_m"].items() if v < -PEN_GATE_M}
        if set_pen:
            reasons.append(f"still interpenetrating after settle beyond gate: {set_pen}")
    return {"scene_xml": str(scene_xml), "object": obj_name,
            "verdict": "FLAG" if reasons else "ADMIT",
            "flag_reasons": reasons,
            "raw_spawn": raw, "post_settle": settled, "settle_shift_m": shift,
            "arm_home_overlap_m": raw["grip_min_dist_m"] if raw["grip_min_dist_m"] < 0 else None,
            "arm_home_pair": raw["grip_min_pair"] if raw["grip_min_dist_m"] < 0 else None}


def audit_all(out_path=None):
    results = []
    for name, scene, traj, obj in SCENES_18:
        r = audit_scene(_STAGE / "scenes" / scene, obj, _WORK / "proposals" / traj)
        r["name"] = name
        r["expected"] = "FLAG" if name in KNOWN_FLAG else "ADMIT"
        r["matches_postmortem"] = (r["verdict"] == r["expected"])
        results.append(r)
        print(f"{name}: {r['verdict']} (expected {r['expected']}) "
              f"raw={r['raw_spawn']['obj_static_pairs_m']} shift={r['settle_shift_m']}", flush=True)
    ok = all(r["matches_postmortem"] for r in results)
    rec = {"gate": "admission_gate.py (protocol v2 P4)", "all_match_postmortem": ok,
           "n_flag": sum(1 for r in results if r["verdict"] == "FLAG"), "results": results}
    if out_path:
        Path(out_path).write_text(json.dumps(rec, indent=1))
    print("ALL_MATCH_POSTMORTEM:", ok)
    return rec


if __name__ == "__main__":
    if sys.argv[1] == "--all":
        out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
        audit_all(out)
    else:
        traj = None
        if "--traj" in sys.argv:
            traj = sys.argv[sys.argv.index("--traj") + 1]
        print(json.dumps(audit_scene(sys.argv[1], sys.argv[2], traj), indent=1))
