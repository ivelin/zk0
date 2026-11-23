"""Compare Drake vs MuJoCo nominal-variant replays on the dev subset.

Per object: position delta (m) and orientation delta (deg, angle of the
relative quaternion). Per episode: outcome class (sim_success) and
secondary-criterion agreement, grasp-event comparison for diagnosis.
Writes equiv_summary.json + prints per-task divergence distributions.
"""
import json, math, sys
from pathlib import Path
import numpy as np

DEV = {"block_stack": [411, 453, 489, 498],
       "eye_drops_to_basket": [1630, 1746, 1783, 1798],
       "ring_insert": [7, 33, 167, 195],
       "tool_insert": [683, 697, 852, 883]}


def qangle_deg(qa, qb):
    qa = np.array(qa, float); qb = np.array(qb, float)
    d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def ev_summary(events):
    return [(e["event"], e.get("object"), e["frame"]) for e in events
            if e["event"] in ("attach", "release")]


def main(mj_dir, dk_dir, out_path):
    rows, disagreements = [], []
    for task, eps in DEV.items():
        for epi in eps:
            mj = json.loads((Path(mj_dir) / f"{task}__ep{epi}_v0.json").read_text())
            dk = json.loads((Path(dk_dir) / f"{task}__ep{epi}_v0.json").read_text())
            objs = {}
            for nm, pm in mj["objects_final"].items():
                pd = dk["objects_final"].get(nm)
                if pd is None:
                    continue
                entry = {"pos_delta_m": round(float(np.linalg.norm(np.array(pm) - np.array(pd))), 4),
                         "mj_pos": pm, "dk_pos": pd}
                qm = mj.get("objects_final_quat_wxyz", {}).get(nm)
                qd = dk.get("objects_final_quat_wxyz", {}).get(nm)
                if qm and qd:
                    entry["ori_delta_deg"] = round(qangle_deg(qm, qd), 2)
                objs[nm] = entry
            row = {"task": task, "episode": epi,
                   "mj_success": mj["sim_success"], "dk_success": dk["sim_success"],
                   "outcome_agree": mj["sim_success"] == dk["sim_success"],
                   "mj_secondary": mj.get("secondary_pass"), "dk_secondary": dk.get("secondary_pass"),
                   "secondary_agree": mj.get("secondary_pass") == dk.get("secondary_pass"),
                   "mj_events": ev_summary(mj.get("grasp_events", [])),
                   "dk_events": ev_summary(dk.get("grasp_events", [])),
                   "mj_checks": mj["checks"], "dk_checks": dk["checks"],
                   "mj_build_defect": mj.get("build_defect"), "dk_build_defect": dk.get("build_defect"),
                   "dk_solver_failure": dk.get("solver_failure"),
                   "objects": objs}
            rows.append(row)
            if not row["outcome_agree"]:
                disagreements.append(row)
    stats = {}
    for task in DEV:
        sub = [r for r in rows if r["task"] == task]
        posd = [o["pos_delta_m"] for r in sub for o in r["objects"].values()]
        orid = [o["ori_delta_deg"] for r in sub for o in r["objects"].values() if "ori_delta_deg" in o]
        stats[task] = {
            "n": len(sub),
            "outcome_agree": sum(r["outcome_agree"] for r in sub),
            "secondary_agree": sum(r["secondary_agree"] for r in sub),
            "mj_successes": sum(r["mj_success"] for r in sub),
            "dk_successes": sum(r["dk_success"] for r in sub),
            "pos_delta_m": {"median": round(float(np.median(posd)), 4) if posd else None,
                            "max": round(float(np.max(posd)), 4) if posd else None,
                            "all": posd},
            "ori_delta_deg": {"median": round(float(np.median(orid)), 2) if orid else None,
                              "max": round(float(np.max(orid)), 2) if orid else None,
                              "all": orid}}
    out = {"dev_subset": DEV, "per_task": stats, "episodes": rows,
           "disagreements": [{"task": r["task"], "episode": r["episode"]} for r in disagreements]}
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))
    print("DISAGREEMENTS:", out["disagreements"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
