"""Builder 2 stage 1c: dwell-frame definition comparison.
S3 oracle dwell = argmin planar site<->reference-object distance in excursion (replay.py definition; pairing only).
S4 object-free: grasp = argmin site-z over first half of run; place = argmin site-z over second half.
S5 object-free: grasp = argmax closedness in first 25% of run; place = argmax closedness in last 25%.
Report S3 residuals (body & site) and held-out offset-model sigma for S4/S5 (site).
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco
W = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
sys.path.insert(0, str(W / "code"))
src = Path(W / "code" / "builder2_stage1.py").read_text()
exec(src.split("archive = []")[0])
site_id = m.site("gripperframe").id
def fk_all(gt_row):
    d.qpos[:] = 0
    for a, v in zip(qadr, gt_row): d.qpos[a] = np.deg2rad(v)
    mujoco.mj_forward(m, d)
    return d.site_xpos[site_id].copy(), d.xpos[gid].copy()

rows = []
for f in sorted(REC.glob("base__*.npz")):
    stem = f.name[6:-4]; task, eps = stem.rsplit("__ep", 1); epi = int(eps)
    if task == "cable_clip": continue
    gt = np.load(f)["gt"].astype(float)
    cl, has = closedness(gt[:, 5])
    if not has: continue
    runs = excursions(cl)
    if not runs: continue
    ini, fin = refs(task, epi)
    for kind, run, ref in (("grasp", runs[0], ini), ("place", runs[-1], fin)):
        if ref is None: continue
        a, b = run; obj = np.array(ref[1]); L = b - a + 1
        sites = np.array([fk_all(gt[fr])[0] for fr in range(a, b + 1)])
        bodys = np.array([fk_all(gt[fr])[1] for fr in range(a, b + 1)])
        dist = np.linalg.norm(sites[:, :2] - obj, axis=1)
        i_s3 = int(np.argmin(dist))
        if kind == "grasp":
            half = max(1, L // 2); i_s4 = int(np.argmin(sites[:half, 2]))
            q = max(1, L // 4); i_s5 = int(np.argmax(cl[a:a + q]))
        else:
            half = max(1, L // 2); i_s4 = half + int(np.argmin(sites[half:, 2]))
            q = max(1, L // 4); i_s5 = (L - q) + int(np.argmax(cl[a + L - q:a + L]))
        rows.append({"task": task, "epi": epi, "kind": kind, "obj": obj.tolist(),
                     "run": [a, b], "frame_s3": a + i_s3, "frame_s4": a + i_s4, "frame_s5": a + i_s5,
                     "site_s3": sites[i_s3].tolist(), "body_s3": bodys[i_s3].tolist(),
                     "site_s4": sites[i_s4].tolist(), "site_s5": sites[i_s5].tolist(),
                     "resid_s3_site_cm": round(float(dist[i_s3] * 100), 3),
                     "resid_s3_body_cm": round(float(np.linalg.norm(bodys[i_s3, :2] - obj) * 100), 3)})
print("rows", len(rows))
for kind in ("grasp", "place"):
    for kk in ("resid_s3_site_cm", "resid_s3_body_cm"):
        r = np.array([x[kk] for x in rows if x["kind"] == kind])
        print(kind, kk, "n", len(r), "median", round(float(np.median(r)), 2), "mean", round(float(np.mean(r)), 2), "frac<3cm", round(float((r < 3).mean()), 3))
def ev(pool, key, task=None):
    tr = [r for r in pool if r["epi"] % 2 == 0 and (task is None or r["task"] == task)]
    te = [r for r in pool if r["epi"] % 2 == 1 and (task is None or r["task"] == task)]
    if len(tr) < 3 or len(te) < 3: return None
    off = np.median([np.array(r["obj"]) - np.array(r[key][:2]) for r in tr], axis=0)
    rad = np.array([np.linalg.norm(np.array(r["obj"]) - (np.array(r[key][:2]) + off)) * 100 for r in te])
    return {"n_test": len(te), "median_cm": round(float(np.median(rad)), 2), "rms_cm": round(float(np.sqrt((rad ** 2).mean())), 2), "sigma_peraxis_cm": round(float(np.sqrt((rad ** 2).mean()) / np.sqrt(2)), 2), "offset_m": [round(float(x), 4) for x in off]}
out = {}
for kind in ("grasp", "place"):
    pool = [r for r in rows if r["kind"] == kind]
    for key in ("site_s4", "site_s5", "site_s3"):
        out[f"{kind}/{key}/offset/ALL"] = ev(pool, key)
        for task in ("block_stack", "ring_insert", "tool_insert", "eye_drops_to_basket"):
            out[f"{kind}/{key}/offset/{task}"] = ev(pool, key, task)
json.dump({"rows": rows, "eval": out}, open(W / "validation" / "builder2_stage1c.json", "w"), indent=1)
for k, v in out.items():
    if v: print(k, v)
