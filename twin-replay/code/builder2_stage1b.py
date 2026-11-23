"""Builder 2 stage 1b: dwell selector + anchor-model comparison. Deterministic, CPU only.
Selectors: argmax-closedness vs min-site-z within excursion. Points: gripper BODY vs gripperframe SITE.
Models: raw distance, per-task constant offset, per-task affine FK->object, global affine. Train=even ep, test=odd ep.
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco
W = Path("/home/openclaw/muse-work/zk0-study001-twin-20261122")
sys.path.insert(0, str(W / "code"))
import importlib.util
spec = importlib.util.spec_from_file_location("s1", str(W / "code" / "builder2_stage1.py"))
# do NOT execute s1 (it runs); instead re-implement refs by copying its functions via exec of the top part
src = Path(W / "code" / "builder2_stage1.py").read_text()
head = src.split("archive = []")[0]
exec(head)  # defines fk? no - defines closedness, excursions, refs, model m/d (body fk). Extend below.

site_id = m.site("gripperframe").id
def fk_site(gt_row):
    d.qpos[:] = 0
    for a, v in zip(qadr, gt_row): d.qpos[a] = np.deg2rad(v)
    mujoco.mj_forward(m, d)
    return d.site_xpos[site_id].copy(), d.xpos[gid].copy()

def fit_affine(X, Y):
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return coef
def apply_affine(coef, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ coef

rows = []  # task, epi, kind, obj_fk_xy, site_argmax, site_minz, body_argmax
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
        a, b = run
        dw = a + int(np.argmax(cl[a:b + 1]))
        # min site-z frame within excursion
        zs = []
        for fr in range(a, b + 1):
            s, _ = fk_site(gt[fr]); zs.append(s[2])
        mz = a + int(np.argmin(zs))
        s_am, b_am = fk_site(gt[dw]); s_mz, _ = fk_site(gt[mz])
        rows.append({"task": task, "epi": epi, "kind": kind, "obj": ref[1],
                     "site_argmax": s_am[:2].tolist(), "site_minz": s_mz[:2].tolist(), "body_argmax": b_am[:2].tolist()})
print("paired rows:", len(rows))
def ev(pool, key, model, task=None):
    tr = [r for r in pool if r["epi"] % 2 == 0 and (task is None or r["task"] == task)]
    te = [r for r in pool if r["epi"] % 2 == 1 and (task is None or r["task"] == task)]
    if len(tr) < 3 or len(te) < 3: return None
    Xtr = np.array([r[key] for r in tr]); Ytr = np.array([r["obj"] for r in tr])
    Xte = np.array([r[key] for r in te]); Yte = np.array([r["obj"] for r in te])
    if model == "raw": pred = Xte
    elif model == "offset":
        off = np.median(Ytr - Xtr, axis=0); pred = Xte + off
    elif model == "affine":
        pred = apply_affine(fit_affine(Xtr, Ytr), Xte)
    rad = np.linalg.norm(Yte - pred, axis=1) * 100
    return {"n_test": len(te), "median_cm": round(float(np.median(rad)), 2),
            "rms_cm": round(float(np.sqrt((rad ** 2).mean())), 2),
            "sigma_peraxis_cm": round(float(np.sqrt((rad ** 2).mean()) / np.sqrt(2)), 2)}
out = {}
for kind in ("grasp", "place"):
    pool = [r for r in rows if r["kind"] == kind]
    for key in ("site_argmax", "site_minz", "body_argmax"):
        for model in ("raw", "offset", "affine"):
            out[f"{kind}/{key}/{model}/ALL"] = ev(pool, key, model)
    for task in ("block_stack", "ring_insert", "tool_insert", "eye_drops_to_basket"):
        out[f"{kind}/site_minz/offset/{task}"] = ev(pool, "site_minz", "offset", task)
        out[f"{kind}/site_argmax/offset/{task}"] = ev(pool, "site_argmax", "offset", task)
        out[f"{kind}/site_minz/affine/{task}"] = ev(pool, "site_minz", "affine", task)
json.dump({"rows": rows, "eval": out}, open(W / "validation" / "builder2_stage1b.json", "w"), indent=1)
for k, v in out.items():
    if v: print(k, v)
