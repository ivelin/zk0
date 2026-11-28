"""Assemble judge packets + provisional rubric judgments from eval records.
Implements the rubric in eval-plan.md (+A1). Provisional judgments are
finalized only after the reviewing agent inspects borderline packets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DS = Path("/home/openclaw/muse-work/zk0-onarm-20261117/data/armnetbench_v01_lerobot_so101")
import os
E = Path(os.environ.get("QA_EVAL_TREE", "/home/openclaw/muse-work/zk0-qa-resume-20261126/eval"))
# P1 eval gate (audit F2, 2026-11-22): judging consumes eval records;
# refuse without a PASS eval gate report covering every checkpoint
# present in the records dir. EVAL_GATE_OVERRIDE=1 = deliberate escape.
import os as _os, sys as _sys
if _os.environ.get("EVAL_GATE_OVERRIDE") == "1":
    print("WARNING: EVAL GATE OVERRIDDEN DELIBERATELY (EVAL_GATE_OVERRIDE=1)")
else:
    _rep = E / "eval_gate_report.json"
    _names = sorted({q.stem.split("__")[0]
                     for q in (E / "records").glob("*__*__ep*.npz")})
    try:
        _r = json.loads(_rep.read_text())
        _covered = set(_r.get("checkpoints", {}).keys())
        _ok = _r.get("verdict") == "PASS" and set(_names) <= _covered
        _why = "" if _ok else (f"verdict={_r.get(verdict)}, "
                               f"uncovered={sorted(set(_names) - _covered)}")
    except Exception as e:
        _ok, _why = False, f"unreadable report {_rep}: {e}"
    if not _ok:
        print(f"REFUSING to judge: P1 eval gate — {_why}. Run "
              "study_prep/eval_gate_check.py over the full comparison "
              "set first, or set EVAL_GATE_OVERRIDE=1 to override "
              "deliberately.")
        _sys.exit(3)
    print(f"eval gate PASS verified ({_rep})")

split = json.loads((E / "split/split.json").read_text())

# p99 step-to-step jump across each task's successful executions
ep = pd.read_parquet(DS / "meta/episodes/chunk-000/file-000.parquet")
ep["task_name"] = ep["tasks"].apply(lambda t: t[0])
import json as j
man = j.loads(Path("/home/openclaw/muse-work/zk0-study001-20261119/manifest.json").read_text())
name2id = {v["instruction"]: k for k, v in man["subsets"].items()}
ep["short"] = ep["task_name"].map(name2id)
succ_eps = {t: set(ep[(ep["short"] == t) & (ep["success"] == 1)]["episode_index"].astype(int)) for t in name2id.values()}
jumps = {t: [] for t in succ_eps}
for f in sorted((DS / "data").glob("chunk-*/file-*.parquet")):
    d = pq.read_table(f, columns=["episode_index", "action", "frame_index"]).to_pandas()
    for epi, g in d.groupby("episode_index"):
        for t, s in succ_eps.items():
            if int(epi) in s:
                a = np.asarray(g.sort_values("frame_index")["action"].tolist(), dtype=np.float64)
                jumps[t].extend(np.linalg.norm(np.diff(a, axis=0), axis=1).tolist())
jump_p99 = {t: float(np.percentile(v, 99)) for t, v in jumps.items()}
print("jump p99 per task:", {k: round(v, 2) for k, v in jump_p99.items()})

out_dir = E / "judgments"; out_dir.mkdir(exist_ok=True)
n = border = 0
for npz in sorted((E / "records").glob("*__*__ep*.npz")):
    ckpt, task, epfile = npz.stem.split("__")
    epi = int(epfile[2:])
    rec = json.loads((out_dir / f"{ckpt}__{task}__ep{epi}.json").read_text()) if (out_dir / f"{ckpt}__{task}__ep{epi}.json").exists() else None
    d = np.load(npz)
    pred, gt = d["pred"], d["gt"]
    cl = split["clusters"][task]
    centroid = np.asarray(cl["final_state_centroid"]); tol = cl["tolerance_p90_pairwise_L2"]
    p5 = np.asarray(cl["final_state_p5"]); p95 = np.asarray(cl["final_state_p95"])
    final = pred[-1]
    end_dist = float(np.linalg.norm(final - centroid))
    rng = (p95 - p5)
    in_range = int(np.sum((final >= p5 - 0.2 * rng) & (final <= p95 + 0.2 * rng)))
    gate_a = bool(end_dist <= tol and in_range >= 5)
    g_lo, g_hi = cl["gripper_final_p10"], cl["gripper_final_p90"]
    g_pad = 0.25 * (g_hi - g_lo)
    gate_b = bool(g_lo - g_pad <= final[5] <= g_hi + g_pad)
    _j = np.linalg.norm(np.diff(pred, axis=0), axis=1) if len(pred) > 1 else np.array([0.0])
    _within = np.array([v for i, v in enumerate(_j) if (i + 1) % 50 != 0])  # A6: exclude chunk boundaries
    max_jump = float(np.max(_within)) if len(_within) else 0.0
    boundary_jump = float(np.max([v for i, v in enumerate(_j) if (i + 1) % 50 == 0])) if len(_j) >= 50 else 0.0
    gate_c = bool(max_jump <= 3 * jump_p99[task])
    completed = bool(gate_a and gate_b and gate_c)
    ratio = end_dist / tol if tol else 999
    borderline = bool(0.8 <= ratio <= 1.25 or (gate_a + gate_b + gate_c) in (1, 2) and completed is False and ratio <= 1.6)
    codes = []
    if not gate_a: codes.append("END_STATE_OUT")
    if not gate_b: codes.append("GRIPPER_MISMATCH")
    if not gate_c: codes.append("PATHOLOGY_JUMP")
    if completed: codes = ["ALL_GATES_PASS"]
    pkt = {"checkpoint": ckpt, "task": task, "episode_index": epi,
           "label": "JUDGED completion (offline replay)",
           "features": {"final_state": final.tolist(), "end_dist_to_cluster_centroid": end_dist,
                        "tolerance": tol, "end_dist_ratio": ratio, "joints_in_success_range": in_range,
                        "gripper_final": float(final[5]), "gripper_cluster_p10_p90": [g_lo, g_hi],
                        "pred_max_step_jump": max_jump, "pred_max_boundary_jump": boundary_jump, "jump_gate": 3 * jump_p99[task],
                        "pred_path_length": float(np.sum(np.linalg.norm(np.diff(pred, axis=0), axis=1))),
                        "gt_final_state": gt[-1].tolist()},
           "gates": {"a_end_state": gate_a, "b_gripper": gate_b, "c_no_pathology": gate_c},
           "provisional_completed": completed, "borderline": borderline,
           "rationale_codes": codes, "judge_status": "PROVISIONAL"}
    (out_dir / f"{ckpt}__{task}__ep{epi}.json").write_text(json.dumps(pkt, indent=1))
    n += 1; border += int(borderline)
print(f"packets: {n}, borderline: {border}")
