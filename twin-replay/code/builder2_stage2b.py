"""Builder 2 stage 2b: re-apply tool repairs under adjudication rule v2 (changelog B2-4).
Reads candidates/anchor from each file's v11_repair block (written by stage 2); deterministic.
"""
import json
from pathlib import Path
import numpy as np
W = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122"); SC = W / "scenes"
cal = json.loads((SC / "calibration_v2.json").read_text())
H = np.array(cal["H_pixel_to_matlocal"]); X0 = cal["x0_near_edge_world_x"]; V0 = cal["v0_arm_centreline"]
M = np.array(json.loads((SC / "calibration_fk_affine.json").read_text())["M"], float)
def px2world(px):
    p = H @ np.array([px[0], px[1], 1.0]); return np.array([p[0] / p[2] + X0, p[1] / p[2] - V0])
def inv_affine(fkxy):
    return np.linalg.solve(M[:2, :].T, np.array(fkxy) - M[2, :])
def fk2world_b(fkxy): return px2world(inv_affine(fkxy))
import sys
sys.path.insert(0, str(W / "code"))
d1d = json.loads((W / "validation" / "builder2_stage1d.json").read_text())["rows"]
def anchor_fk(epi):
    pool = [r for r in d1d if r["task"] == "tool_insert" and r["kind"] == "grasp"]
    tr = [r for r in pool if r["epi"] % 2 == 0]
    off = np.median([np.array(r["obj"]) - np.array(r["site_s7"][:2]) for r in tr], axis=0)
    me = [r for r in pool if r["epi"] == epi]
    return (np.array(me[0]["site_s7"][:2]) + off) if me else None

out = {}
for epi in [697, 779, 784, 885, 888, 892, 903]:
    key = f"tool_insert__ep{epi}"; f = SC / f"{key}.json"
    sc = json.loads(f.read_text()); t = sc["objects"]["tool"]; vr = t["v11_repair"]
    cands = {k: np.array(v, float) for k, v in vr["candidates_px"].items()}
    anch_px = np.array(vr["fk_anchor_px"], float)
    dists = {k: float(np.linalg.norm(v - anch_px)) for k, v in cands.items()}
    chosen_px, source = None, None
    if len(cands) == 2:
        k1, k2 = list(cands)
        if np.linalg.norm(cands[k1] - cands[k2]) <= 25:
            chosen_px = ((cands[k1] + cands[k2]) / 2).tolist(); source = "grounding_consensus"
    if chosen_px is None:
        best = min(dists, key=dists.get)
        if dists[best] <= 75: chosen_px, source = cands[best].tolist(), best
    if chosen_px is not None:
        w = px2world(chosen_px); new = [round(float(w[0]), 4), round(float(w[1]), 4), 0.005, None]
        method = f"visual grounding pick on top_first (v1.1 repair rule v2, Builder 2; source={source}) — replaces fabricated detect4 segmentation pose"
    else:
        w = fk2world_b(anchor_fk(epi)); new = [round(float(w[0]), 4), round(float(w[1]), 4), 0.005, None]
        source = "fk_anchor_only"
        method = "FK-anchored estimate (v1.1 repair rule v2, Builder 2): first-grasp dwell FK (S7) + train-fitted offset; no grounding candidate within 75px of anchor and no consensus — replaces fabricated detect4 segmentation pose"
    t.update({"pose_xyz_yaw": new, "confidence": "medium", "method": method, "sigma_xy_cm": 3.5, "sigma_yaw_deg": None})
    t["v11_repair"] = {**vr, "rule_v2": {"chosen_source": source, "chosen_px": chosen_px,
                        "candidate_dist_to_anchor_px": {k: round(v, 1) for k, v in dists.items()}}}
    if "fk_anchor" in t:  # refresh agreement after pose change
        t["fk_anchor"]["agreement_with_image_pose_cm"] = round(float(np.linalg.norm(np.array(t["fk_anchor"]["xy_world_b"]) - np.array(new[:2])) * 100), 2)
    f.write_text(json.dumps(sc, indent=1))
    out[key] = {"old_fabricated": t["pose_v1_fabricated"], "new_pose": new, "source": source, "chosen_px": chosen_px}
    print(key, "->", new, source, chosen_px)
json.dump(out, open(W / "validation" / "builder2_stage2b.json", "w"), indent=1)
print("STAGE2B DONE")
