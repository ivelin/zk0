"""Builder 2 stage 1 (twin-suite v1.1): dwell tie-point archive + FK-anchor stats.
Deterministic. CPU only. No scene JSON mutation.
Outputs: scenes/dwell_tiepoints_v11.json, validation/builder2_stage1_stats.json
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco

W = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
sys.path.insert(0, str(W / "code"))
import build_scenes as B  # _affine_px, _det_px, correct_xy, EYE_CARTON_OFFSET, _parallax

REC = Path("/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/records")
MODEL = "/home/openclaw/muse-work/zk0-sim-v01-20261115/sim/model/so101_new_calib.xml"
SC = W / "scenes"
JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

m = mujoco.MjModel.from_xml_path(MODEL); d = mujoco.MjData(m)
qadr = [m.joint(n).qposadr[0] for n in JN]
gid = m.body("gripper").id; wid = m.body("wrist").id

def fk(gt_row):
    d.qpos[:] = 0
    for a, v in zip(qadr, gt_row): d.qpos[a] = np.deg2rad(v)
    mujoco.mj_forward(m, d)
    return d.xpos[gid].copy(), d.xpos[wid].copy()

def closedness(grip):
    g = np.asarray(grip, float)
    lo, hi = np.percentile(g, 5), np.percentile(g, 99)
    if hi - lo < 5.0: return np.zeros_like(g), False
    return np.clip((hi - g) / (hi - lo), 0, 1), True

def excursions(cl):
    runs = []; i = 0
    while i < len(cl):
        if cl[i] >= 0.5:
            j = i
            while j + 1 < len(cl) and cl[j + 1] >= 0.5: j += 1
            if j - i + 1 >= 5: runs.append((i, j))
            i = j + 1
        else: i += 1
    return runs

# ---- object reference positions (FK frame) ----
det = json.loads((SC / "detections_px.json").read_text())
TOOL_GROUND_V1 = {"tool_insert__ep622":[550,434],"tool_insert__ep629":[320,291],"tool_insert__ep683":[580,462],
 "tool_insert__ep684":[520,401],"tool_insert__ep824":[575,437],"tool_insert__ep830":[487,417],
 "tool_insert__ep852":[563,468],"tool_insert__ep859":[550,438],"tool_insert__ep883":[482,363],"tool_insert__ep902":[363,313]}
# Builder 2 new grounding picks (visual-grounding locator, 2026-11-22; px = norm1000*(1024,576))
GROUND_B2 = {697:[244,343], 779:[499,286], 784:[500,291], 885:[531,442], 888:[506,411], 892:[629,502], 903:[354,313], 871:[315,310]}
# V-Env independent reads (verification-environments-poses.md table)
GROUND_VENV = {697:[240,350], 779:[220,360], 784:[340,320], 885:[500,452], 888:[475,392], 892:[605,505], 903:[360,302]}
EYE_FIRST_GROUND = {"eye_drops_to_basket__ep1609":[567,217],"eye_drops_to_basket__ep1618":[481,207],
 "eye_drops_to_basket__ep1630":[526,187],"eye_drops_to_basket__ep1639":[543,154],"eye_drops_to_basket__ep1643":[619,222],
 "eye_drops_to_basket__ep1757":[470,82],"eye_drops_to_basket__ep1760":[577,94],"eye_drops_to_basket__ep1778":[505,105],
 "eye_drops_to_basket__ep1808":[441,101],"eye_drops_to_basket__ep1810":[473,108],"eye_drops_to_basket__ep1821":[589,156]}
EYE_FINAL_GROUND = {"eye_drops_to_basket__ep1609":[236,254],"eye_drops_to_basket__ep1630":[130,377],
 "eye_drops_to_basket__ep1639":[118,338],"eye_drops_to_basket__ep1643":[100,343],"eye_drops_to_basket__ep1724":[159,222],
 "eye_drops_to_basket__ep1741":[162,256],"eye_drops_to_basket__ep1746":[160,247],"eye_drops_to_basket__ep1747":[166,224],
 "eye_drops_to_basket__ep1757":[164,248],"eye_drops_to_basket__ep1778":[162,246],"eye_drops_to_basket__ep1780":[127,302],
 "eye_drops_to_basket__ep1783":[138,265],"eye_drops_to_basket__ep1798":[154,251],"eye_drops_to_basket__ep1808":[172,254],
 "eye_drops_to_basket__ep1810":[140,272],"eye_drops_to_basket__ep1811":[124,286],"eye_drops_to_basket__ep1814":[152,254],
 "eye_drops_to_basket__ep1821":[150,287]}
SLOT_PX = [450, 222]

def scene(task, epi): return json.loads((SC / f"{task}__ep{epi}.json").read_text())

def refs(task, epi):
    """Return dict initial=(px, fkxy, name), final=(px, fkxy, name) or None entries."""
    key = f"{task}__ep{epi}"; ini = fin = None
    if task == "block_stack":
        sc = scene(task, epi); objs = sc["objects"]
        man = next((k for k, v in objs.items() if not k.endswith("_final") and v.get("role") == "manipulated"), None)
        base = next((k for k, v in objs.items() if not k.endswith("_final") and v.get("role") == "base"), None)
        if man:
            px = B._det_px(key, "first", man)
            if px is not None: ini = (px.tolist(), B._affine_px(px).tolist(), man)
        if man and base:
            cands = []
            for nm, kk in (("man", man), ("base", base)):
                p = B._det_px(key, "final", kk)
                if p is not None: cands.append(p)
            binit = B._det_px(key, "first", base)
            if cands and binit is not None:
                spx = min(cands, key=lambda q: float(np.linalg.norm(q - binit)))
                fin = (np.asarray(spx).tolist(), B._affine_px(np.asarray(spx, float)).tolist(), "stack")
    elif task == "ring_insert":
        px = B._det_px(key, "first", "ring")
        if px is not None: ini = (px.tolist(), B._affine_px(px).tolist(), "ring")
        px = B._det_px(key, "final", "ring")
        if px is not None: fin = (px.tolist(), B._affine_px(px).tolist(), "ring_final")
    elif task == "tool_insert":
        px = None
        if key in TOOL_GROUND_V1: px = TOOL_GROUND_V1[key]
        elif epi in GROUND_B2: px = GROUND_B2[epi]
        if px is not None: ini = (px, B._affine_px(np.array(px, float)).tolist(), "tool")
        fin = (SLOT_PX, B._affine_px(np.array(SLOT_PX, float)).tolist(), "tool_final_slot")
    elif task == "eye_drops_to_basket":
        d4 = json.loads((SC / "detections4_px.json").read_text()).get(key, {}).get("first", {}).get("carton")
        px = d4["px"] if d4 else EYE_FIRST_GROUND.get(key)
        if px is not None:
            xy = B._affine_px(np.array(px, float)) + np.array(B.EYE_CARTON_OFFSET)
            ini = (list(px), xy.tolist(), "carton")
        px = EYE_FINAL_GROUND.get(key)
        if px is not None:
            xy = B._affine_px(np.array(px, float)) + np.array(B._parallax(0.03))
            fin = (list(px), xy.tolist(), "carton_final")
    return ini, fin

archive = []
episodes = []
for f in sorted(REC.glob("base__*.npz")):
    stem = f.name[6:-4]; task, eps = stem.rsplit("__ep", 1); epi = int(eps)
    if task == "cable_clip": continue  # arm grasps the cable; no point-object reference
    episodes.append((task, epi))
episodes.sort()
for task, epi in episodes:
    gt = np.load(REC / f"base__{task}__ep{epi}.npz")["gt"].astype(float)
    cl, has = closedness(gt[:, 5])
    runs = excursions(cl) if has else []
    ini, fin = refs(task, epi)
    for idx, (a, b) in enumerate(runs):
        dwell = a + int(np.argmax(cl[a:b + 1]))
        g, w = fk(gt[dwell])
        kind = "other"
        ref = None
        if idx == 0: kind, ref = "first_grasp", ini
        if idx == len(runs) - 1 and len(runs) > 0: kind, ref = "last_place", fin
        if idx == 0 and len(runs) == 1: kind, ref = "first_grasp", ini  # single excursion: grasp only
        rec = {"episode_key": f"{task}__ep{epi}", "task": task, "episode": epi,
               "excursion_idx": idx, "kind": kind, "frame_start": a, "frame_end": b,
               "dwell_frame": int(dwell), "closedness_at_dwell": round(float(cl[dwell]), 4),
               "fk_gripper_xyz": [round(float(x), 5) for x in g],
               "fk_wrist_xyz": [round(float(x), 5) for x in w],
               "paired_object": None, "object_px": None, "object_fk_xy": None,
               "resid_vs_fk_affine_v2_cm": None}
        if ref is not None:
            px, fkxy, name = ref
            rec["paired_object"] = name; rec["object_px"] = px
            rec["object_fk_xy"] = [round(float(x), 5) for x in fkxy]
            rec["resid_vs_fk_affine_v2_cm"] = round(float(np.linalg.norm(np.array(fkxy) - g[:2]) * 100), 3)
        archive.append(rec)
    print(task, epi, "excursions", len(runs), flush=True)

json.dump({"method": "closedness p5/p99 inverted (replay.py); excursions closedness>=0.5, len>=5; dwell=argmax closedness; FK=mujoco gripper body xpos, so101_new_calib.xml, recorded gt (base__ records)",
           "n_pairs_total": len(archive),
           "n_paired": sum(1 for r in archive if r["paired_object"]),
           "pairs": archive}, open(SC / "dwell_tiepoints_v11.json", "w"), indent=1)

# ---- stats: affine refit check (block+ring grasp pairs, px->FK) + FK-anchor held-out sigma ----
def stats_for(pairs):
    r = np.array([p["resid_vs_fk_affine_v2_cm"] for p in pairs if p["resid_vs_fk_affine_v2_cm"] is not None])
    if len(r) == 0: return {}
    return {"n": int(len(r)), "median_cm": round(float(np.median(r)), 2), "mean_cm": round(float(np.mean(r)), 2),
            "p90_cm": round(float(np.percentile(r, 90)), 2), "frac_lt_3cm": round(float((r < 3).mean()), 3)}

out = {}
grasp = [r for r in archive if r["kind"] == "first_grasp"]
place = [r for r in archive if r["kind"] == "last_place"]
out["resid_fk_vs_object_all_grasp"] = stats_for(grasp)
out["resid_fk_vs_object_all_place"] = stats_for(place)
for task in ("block_stack", "ring_insert", "tool_insert", "eye_drops_to_basket"):
    out[f"resid_grasp_{task}"] = stats_for([r for r in grasp if r["task"] == task])
    out[f"resid_place_{task}"] = stats_for([r for r in place if r["task"] == task])

# FK-anchor: per task+kind, offset = median(object - FK) on TRAIN (even episode id), held-out on TEST (odd)
anchor = {}
for task in ("block_stack", "ring_insert", "tool_insert", "eye_drops_to_basket"):
    for kind, pool in (("grasp", grasp), ("place", place)):
        pp = [r for r in pool if r["task"] == task and r["object_fk_xy"] is not None]
        tr = [r for r in pp if r["episode"] % 2 == 0]; te = [r for r in pp if r["episode"] % 2 == 1]
        if len(tr) < 3 or len(te) < 3: anchor[f"{task}/{kind}"] = {"n_train": len(tr), "n_test": len(te), "note": "insufficient pairs"}; continue
        off = np.median([np.array(r["object_fk_xy"]) - np.array(r["fk_gripper_xyz"][:2]) for r in tr], axis=0)
        res = np.array([np.array(r["object_fk_xy"]) - (np.array(r["fk_gripper_xyz"][:2]) + off) for r in te])
        rad = np.linalg.norm(res, axis=1) * 100
        anchor[f"{task}/{kind}"] = {"n_train": len(tr), "n_test": len(te),
            "offset_m": [round(float(x), 4) for x in off],
            "heldout_radial_median_cm": round(float(np.median(rad)), 2),
            "heldout_radial_rms_cm": round(float(np.sqrt((rad ** 2).mean())), 2),
            "heldout_sigma_peraxis_cm": round(float(np.sqrt((rad ** 2).mean()) / np.sqrt(2)), 2),
            "heldout_residuals_cm": {r["episode_key"]: round(float(x), 2) for r, x in zip(te, rad)}}
out["fk_anchor_heldout"] = anchor
json.dump(out, open(W / "validation" / "builder2_stage1_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("ARCHIVE pairs:", len(archive), "paired:", sum(1 for r in archive if r["paired_object"]))
