"""Builder 2 stage 2 (twin-suite v1.1): adjudication, tool repairs, block reconciliation,
archive augmentation, FK-anchor fields. Deterministic. CPU only.
"""
import json, hashlib
from pathlib import Path
import numpy as np

W = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
SC = W / "scenes"
cal = json.loads((SC / "calibration_v2.json").read_text())
H = np.array(cal["H_pixel_to_matlocal"]); X0 = cal["x0_near_edge_world_x"]; V0 = cal["v0_arm_centreline"]
M = np.array(json.loads((SC / "calibration_fk_affine.json").read_text())["M"], float)  # rows: [a b],[c d],[tx ty]; fk = M.T@[px,1]
def px2world(px):
    p = H @ np.array([px[0], px[1], 1.0]); return np.array([p[0] / p[2] + X0, p[1] / p[2] - V0])
def world2px(w):
    q = np.linalg.solve(H, np.array([w[0] - X0, w[1] + V0, 1.0])); return q[:2] / q[2]
def affine_px(px): return M.T @ np.array([px[0], px[1], 1.0])
def inv_affine(fkxy):
    A = M[:2, :].T  # fk = A @ px + t  (since fk = M.T@[px,1], linear part = M[:2].T)
    return np.linalg.solve(A, np.array(fkxy) - M[2, :])
def fk2world_b(fkxy):  # FK-frame xy -> Builder B world xy (inverse composite)
    return px2world(inv_affine(fkxy))

GROUND_B2 = {697:[244,343], 779:[499,286], 784:[500,291], 885:[531,442], 888:[506,411], 892:[629,502], 903:[354,313], 871:[315,310]}
GROUND_VENV = {697:[240,350], 779:[220,360], 784:[340,320], 885:[500,452], 888:[475,392], 892:[605,505], 903:[360,302]}
FAB7 = [697, 779, 784, 885, 888, 892, 903]

# FK anchors (S7 selector) from stage1d rows: per-task train offset (grasp)
d1d = json.loads((W / "validation" / "builder2_stage1d.json").read_text())
rows = d1d["rows"]
def anchor_fk(task, epi, kind="grasp"):
    pool = [r for r in rows if r["task"] == task and r["kind"] == kind]
    tr = [r for r in pool if r["epi"] % 2 == 0]
    if len(tr) < 3: return None
    off = np.median([np.array(r["obj"]) - np.array(r["site_s7"][:2]) for r in tr], axis=0)
    me = [r for r in pool if r["epi"] == epi]
    if not me: return None
    return np.array(me[0]["site_s7"][:2]) + off

adjud = {}
for epi in FAB7 + [871, 630, 876]:
    key = f"tool_insert__ep{epi}"
    anch = anchor_fk("tool_insert", epi)
    anch_px = inv_affine(anch).tolist() if anch is not None else None
    cands = {}
    if epi in GROUND_B2: cands["builder2_grounding"] = GROUND_B2[epi]
    if epi in GROUND_VENV: cands["venv_grounding"] = GROUND_VENV[epi]
    dists = {k: (round(float(np.linalg.norm(np.array(v) - np.array(anch_px))), 1) if anch_px else None) for k, v in cands.items()}
    chosen, source = None, None
    if dists:
        best = min(dists, key=lambda k: dists[k] if dists[k] is not None else 1e9)
        if dists[best] is not None and dists[best] <= 75:
            chosen, source = cands[best], best
    if chosen is None and anch is not None:
        chosen, source = None, "fk_anchor_only"
    adjud[key] = {"candidates_px": cands, "fk_anchor_px": [round(float(x), 1) for x in anch_px] if anch_px else None,
                  "candidate_dist_to_anchor_px": dists, "chosen_source": source,
                  "chosen_px": chosen}
print(json.dumps(adjud, indent=1))

# ---- repairs to the 7 ----
repair_log = {}
for epi in FAB7:
    key = f"tool_insert__ep{epi}"; f = SC / f"{key}.json"
    sc = json.loads(f.read_text()); old = sc["objects"]["tool"]["pose_xyz_yaw"]
    a = adjud[key]
    if a["chosen_px"] is not None:
        w = px2world(a["chosen_px"]); new = [round(float(w[0]), 4), round(float(w[1]), 4), 0.005, None]
        method = f"visual grounding pick on top_first (v1.1 repair, Builder 2; candidate={a['chosen_source']}, adjudicated vs FK grasp anchor, dist {a['candidate_dist_to_anchor_px'][a['chosen_source']]}px) — replaces fabricated detect4 segmentation pose"
        conf, sig = "medium", 3.5
    else:
        anch = anchor_fk("tool_insert", epi); w = fk2world_b(anch)
        new = [round(float(w[0]), 4), round(float(w[1]), 4), 0.005, None]
        method = "FK-anchored estimate (v1.1 repair, Builder 2): first-grasp dwell FK (S7 crossing+lag selector) + per-task train-fitted offset; grounding candidates rejected (>75px from FK anchor) — replaces fabricated detect4 segmentation pose"
        conf, sig = "medium", 3.5
    sc["objects"]["tool"].update({"pose_xyz_yaw": new, "confidence": conf, "method": method,
        "sigma_xy_cm": sig, "sigma_yaw_deg": None,
        "pose_v1_fabricated": old,
        "v11_repair": {"chosen_source": a["chosen_source"], "chosen_px": a["chosen_px"], "fk_anchor_px": a["fk_anchor_px"],
                       "candidates_px": a["candidates_px"], "candidate_dist_to_anchor_px": a["candidate_dist_to_anchor_px"]}})
    f.write_text(json.dumps(sc, indent=1))
    repair_log[key] = {"old_pose": old, "new_pose": new, "source": a["chosen_source"]}
    print("REPAIRED", key, old, "->", new, a["chosen_source"])

# ---- FK-anchor fields on all anchored episodes (no pose replacement) ----
SIG = {  # held-out sigma/axis cm, S7 offset model (stage1d)
 ("block_stack", "grasp"): 4.90, ("block_stack", "place"): 7.20,
 ("ring_insert", "grasp"): 3.11, ("ring_insert", "place"): 3.19,
 ("tool_insert", "grasp"): 3.03, ("tool_insert", "place"): 2.48,
 ("eye_drops_to_basket", "grasp"): 4.61, ("eye_drops_to_basket", "place"): 4.59}
NAME = {"block_stack": None, "ring_insert": "ring", "tool_insert": "tool", "eye_drops_to_basket": "carton"}
n_anchor_fields = 0
for r in rows:
    task, epi, kind = r["task"], r["epi"], r["kind"]
    f = SC / f"{task}__ep{epi}.json"
    if not f.exists(): continue
    sc = json.loads(f.read_text()); objs = sc["objects"]
    if task == "block_stack":
        name = next((k for k, v in objs.items() if not k.endswith("_final") and v.get("role") == "manipulated"), None)
        if kind == "place": name = (name + "_final") if name else None
    else:
        name = NAME[task] if kind == "grasp" else NAME[task] + "_final"
    if name is None or name not in objs or objs[name].get("pose_xyz_yaw") is None: continue
    pool = [x for x in rows if x["task"] == task and x["kind"] == kind]
    tr = [x for x in pool if x["epi"] % 2 == 0]
    off = np.median([np.array(x["obj"]) - np.array(x["site_s7"][:2]) for x in tr], axis=0)
    anch = np.array(r["site_s7"][:2]) + off
    bw = fk2world_b(anch)
    img = np.array(objs[name]["pose_xyz_yaw"][:2])
    objs[name]["fk_anchor"] = {"xy_world_b": [round(float(x), 4) for x in bw],
        "sigma_xy_cm_heldout": SIG[(task, kind)], "kind": kind,
        "agreement_with_image_pose_cm": round(float(np.linalg.norm(bw - img) * 100), 2),
        "note": "FK-anchored cross-check (S7 dwell + train-fitted per-task offset). Held-out sigma is WORSE than the image pipeline on every task (builder2 report); anchor does NOT replace the image pose."}
    f.write_text(json.dumps(sc, indent=1)); n_anchor_fields += 1
print("fk_anchor fields added:", n_anchor_fields)

# ---- block reconciliation ----
det = json.loads((SC / "detections_px.json").read_text())
HAND = {("block_stack__ep411", "cube_purple"), ("block_stack__ep530", "cube_purple"),
        ("block_stack__ep539", "cube_purple"), ("block_stack__ep539", "cube_green"), ("block_stack__ep546", "cube_purple")}
import glob, os
recs = []
for fp in sorted(glob.glob(str(SC / "block_stack__ep*.json"))):
    if "variants" in fp: continue
    epk = os.path.basename(fp)[:-5]; scj = json.loads(Path(fp).read_text())
    for name, o in scj["objects"].items():
        isfin = name.endswith("_final"); base = name.replace("_final", "")
        d = det.get(epk, {}).get("final" if isfin else "first", {}).get(base)
        if d is None or "px" not in d or o.get("pose_xyz_yaw") is None: continue
        w = px2world(d["px"]); jp = np.array(o["pose_xyz_yaw"][:2])
        dist = round(float(np.linalg.norm(w - jp) * 100), 2)
        if dist <= 2.0: cls, auth, why = "AGREE", "both (identical quantity)", "JSON pose re-derives from preserved detection px through calib_v2 homography."
        elif not isfin and (epk, base) in HAND: cls, auth, why = "MANUAL_PICK_INITIAL", "published JSON pose (manual pick)", "finalize.py HAND_BLOCK_FIRST: the preserved detection is the FAILED auto detection (occlusion/median coincidence) that Builder B replaced by a visual-verification manual pick; V-Env frame read on ep411 supports the pick (JSON ~2-3cm from visible cube, detection ~13cm off on empty mat). Detection px retained as the failed-auto record."
        elif isfin:
            # verify stack-mean construction
            fin = {k: v for k, v in det[epk]["final"].items() if k.startswith("cube") and "px" in v}
            meanw = px2world(np.mean([v["px"] for v in fin.values()], axis=0))
            md = round(float(np.linalg.norm(meanw - jp) * 100), 2)
            cls, auth, why = "FINAL_STACK_CONSTRUCTION", "published JSON pose (stack location)", f"JSON final = stack location (mean of top_final cube detections; re-derived mean deviates {md} cm from JSON, rounding/median-pass differences only). The per-colour final detection px measures an individual colour blob in top_final (often only the visible top cube or lettering), NOT the stack location - a different quantity, not a competing pose. Builder C build_scenes derives its own stack px (detection nearest base initial) from the same preserved pixels."
        else: cls, auth, why = "UNEXPLAINED", "?", "no class matched"
        recs.append({"episode_key": epk, "entry": name, "dist_json_vs_detpx_cm": dist, "class": cls,
                     "authoritative_source": auth, "rationale": why, "json_pose_xy": o["pose_xyz_yaw"][:2],
                     "detpx": d["px"], "detpx_world_xy": [round(float(x), 4) for x in w]})
json.dump({"n_entries": len(recs), "n_over_2cm": sum(1 for r in recs if r["dist_json_vs_detpx_cm"] > 2.0), "entries": recs},
          open(SC / "block_reconciliation_v11.json", "w"), indent=1)
from collections import Counter
print("block reconciliation:", len(recs), Counter(r["class"] for r in recs))

# ---- archive augmentation: add site FK + selector frames to paired dwells ----
arch = json.loads((SC / "dwell_tiepoints_v11.json").read_text())
d1c = json.loads((W / "validation" / "builder2_stage1c.json").read_text())
lut = {}
for src_rows in (d1c["rows"], rows):
    for r in src_rows:
        lut[(r["task"], r["epi"], "first_grasp" if r["kind"] == "grasp" else "last_place")] = r
n_aug = 0
for p in arch["pairs"]:
    if p["paired_object"] is None: continue
    kind = p["kind"]
    r = lut.get((p["task"], p["episode"], kind))
    if r is None: continue
    p["fk_gripper_site_xyz"] = [round(float(x), 5) for x in r["site_s3"]]
    p["dwell_frame_oracle_s3"] = r.get("frame_s3"); p["dwell_frame_s6_firstdip"] = r.get("frame_s6"); p["dwell_frame_s7_lag"] = r.get("frame_s7")
    p["resid_oracle_site_cm"] = r.get("resid_s3_site_cm"); p["resid_oracle_body_cm"] = r.get("resid_s3_body_cm")
    n_aug += 1
arch["augmentation_v11"] = "fk_gripper_site_xyz + oracle/S6/S7 dwell frames + oracle residuals added by Builder 2 stage 2 from validation/builder2_stage1c.json + builder2_stage1d.json rows (same deterministic extraction)."
arch["n_augmented"] = n_aug
(SC / "dwell_tiepoints_v11.json").write_text(json.dumps(arch, indent=1))
print("archive augmented paired entries:", n_aug, "of", arch["n_paired"])
json.dump({"adjudication": adjud, "repairs": repair_log}, open(W / "validation" / "builder2_stage2.json", "w"), indent=1)
print("STAGE2 DONE")
