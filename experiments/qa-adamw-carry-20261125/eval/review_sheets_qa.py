"""Borderline review sheets: GT vs predicted per borderline packet."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
E = Path(os.environ.get("QA_EVAL_TREE", "/home/openclaw/muse-work/zk0-qa-resume-20261126/eval"))
out = E / "judgments/review"; out.mkdir(parents=True, exist_ok=True)
pk = []
for p in sorted((E / "judgments").glob("*__*__ep*.json")):
    d = json.loads(p.read_text())
    if d["borderline"]:
        pk.append(d)
print("borderline packets:", len(pk))
PER = 6
for si in range(0, len(pk), PER):
    batch = pk[si:si + PER]
    fig, axes = plt.subplots(len(batch), 1, figsize=(11, 2.2 * len(batch)))
    if len(batch) == 1: axes = [axes]
    for ax, d in zip(axes, batch):
        rec = np.load(E / "records" / f"{d['checkpoint']}__{d['task']}__ep{d['episode_index']}.npz")
        gt, pred = rec["gt"], rec["pred"]
        for j in range(6):
            ax.plot(gt[:, j], lw=0.8)
            ax.plot(pred[:, j], lw=0.8, ls="--")
        f = d["features"]; g = d["gates"]
        ax.set_title(f"{d['checkpoint']} {d['task']} ep{d['episode_index']} prov={d['provisional_completed']} "
                     f"ratio={f['end_dist_ratio']:.2f} inrange={f['joints_in_success_range']}/6 "
                     f"gates a={g['a_end_state']} b={g['b_gripper']} c={g['c_no_pathology']} "
                     f"grip={f['gripper_final']:.0f} (cluster {f['gripper_cluster_p10_p90']})", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"sheet_{si // PER:02d}.png", dpi=95)
    plt.close(fig)
print("sheets:", len(list(out.glob("sheet_*.png"))))
# machine-readable list of borderline packets in sheet order
(E / "judgments/review/borderline_list.json").write_text(json.dumps(
    [{"checkpoint": d["checkpoint"], "task": d["task"], "episode_index": d["episode_index"]} for d in pk], indent=1))
