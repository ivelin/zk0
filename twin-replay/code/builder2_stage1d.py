"""Builder 2 stage 1d: deployment-realistic dwell selectors.
S6 first-dip/last-dip: first local min of site-z after run start with closedness>=0.9, search ends when z rises >=2cm above the dip (grasp); mirrored from run end (place).
S7 crossing+lag: dwell = run start + per-task median lag (lag = S3 frame - run start, fitted on TRAIN only) for grasp; run end - median lag for place.
Also: C-claim analogue: affine px->FK(site) fit on block+ring grasp pairs, in-sample + held-out residuals.
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
def fk_site(gt_row):
    d.qpos[:] = 0
    for a, v in zip(qadr, gt_row): d.qpos[a] = np.deg2rad(v)
    mujoco.mj_forward(m, d)
    return d.site_xpos[site_id].copy()

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
        a, b = run; obj = np.array(ref[1]); px = np.array(ref[0], float)
        sites = np.array([fk_site(gt[fr]) for fr in range(a, b + 1)])
        dist = np.linalg.norm(sites[:, :2] - obj, axis=1)
        i_s3 = int(np.argmin(dist))
        z = sites[:, 2]
        # S6
        if kind == "grasp":
            i_s6 = 0; best = z[0]
            for i in range(1, len(z)):
                if z[i] < best - 1e-9: best = z[i]; i_s6 = i
                elif z[i] > best + 0.02 and cl[a + i] >= 0.0: break
        else:
            i_s6 = len(z) - 1; best = z[-1]
            for i in range(len(z) - 2, -1, -1):
                if z[i] < best - 1e-9: best = z[i]; i_s6 = i
                elif z[i] > best + 0.02: break
        rows.append({"task": task, "epi": epi, "kind": kind, "obj": obj.tolist(), "px": px.tolist(),
                     "run": [a, b], "lag_s3": int(i_s3) if kind == "grasp" else int(len(z) - 1 - i_s3),
                     "site_s3": sites[i_s3].tolist(), "site_s6": sites[i_s6].tolist(), "frame_s6": a + i_s6, "frame_s3": a + i_s3})
print("rows", len(rows))
# S7 lags from train
lags = {}
for task in ("block_stack", "ring_insert", "tool_insert", "eye_drops_to_basket"):
    for kind in ("grasp", "place"):
        tr = [r["lag_s3"] for r in rows if r["task"] == task and r["kind"] == kind and r["epi"] % 2 == 0]
        lags[f"{task}/{kind}"] = int(np.median(tr)) if tr else 0
print("S7 lags (train medians):", lags)
# need site at crossing+lag: recompute per row is expensive; approximate S7 site by interpolating? Instead store sites arrays? redo quickly:
for r in rows:
    a, b = r["run"]; lag = lags[f"{r['task']}/{r['kind']}"]
    idx = lag if r["kind"] == "grasp" else (b - a) - lag
    idx = min(max(idx, 0), b - a)
    gt = np.load(REC / f"base__{r['task']}__ep{r['epi']}.npz")["gt"].astype(float)
    r["site_s7"] = fk_site(gt[a + idx]).tolist(); r["frame_s7"] = a + idx
def ev(pool, key, task=None):
    tr = [r for r in pool if r["epi"] % 2 == 0 and (task is None or r["task"] == task)]
    te = [r for r in pool if r["epi"] % 2 == 1 and (task is None or r["task"] == task)]
    if len(tr) < 3 or len(te) < 3: return None
    off = np.median([np.array(r["obj"]) - np.array(r[key][:2]) for r in tr], axis=0)
    rad = np.array([np.linalg.norm(np.array(r["obj"]) - (np.array(r[key][:2]) + off)) * 100 for r in te])
    return {"n_test": len(te), "median_cm": round(float(np.median(rad)), 2), "rms_cm": round(float(np.sqrt((rad ** 2).mean())), 2), "sigma_peraxis_cm": round(float(np.sqrt((rad ** 2).mean()) / np.sqrt(2)), 2)}
out = {"lags_s7": lags}
for kind in ("grasp", "place"):
    pool = [r for r in rows if r["kind"] == kind]
    for key in ("site_s6", "site_s7"):
        out[f"{kind}/{key}/offset/ALL"] = ev(pool, key)
        for task in ("block_stack", "ring_insert", "tool_insert", "eye_drops_to_basket"):
            out[f"{kind}/{key}/offset/{task}"] = ev(pool, key, task)
# C-claim analogue: affine px -> FK site (S3 dwell), block+ring grasp+place pairs
for poolname, filt in (("block+ring", lambda r: r["task"] in ("block_stack", "ring_insert")), ("all", lambda r: True)):
    pp = [r for r in rows if filt(r)]
    tr = [r for r in pp if r["epi"] % 2 == 0]; te = [r for r in pp if r["epi"] % 2 == 1]
    X = np.array([r["px"] for r in tr]); Y = np.array([r["site_s3"][:2] for r in tr])
    A = np.concatenate([X, np.ones((len(X), 1))], 1); coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    rin = np.linalg.norm(A @ coef - Y, axis=1) * 100
    Xte = np.array([r["px"] for r in te]); Yte = np.array([r["site_s3"][:2] for r in te])
    rte = np.linalg.norm(np.concatenate([Xte, np.ones((len(Xte), 1))], 1) @ coef - Yte, axis=1) * 100
    out[f"affine_px_to_FKsite_{poolname}"] = {"n_train": len(tr), "n_test": len(te),
        "insample_median_cm": round(float(np.median(rin)), 2), "heldout_median_cm": round(float(np.median(rte)), 2),
        "heldout_rms_cm": round(float(np.sqrt((rte ** 2).mean())), 2)}
json.dump({"rows": rows, "eval": out}, open(W / "validation" / "builder2_stage1d.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
