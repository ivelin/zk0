#!/usr/bin/env python3
"""Settle / load check for spec-only scenes (NOT solvability validation).

Loads every generated XML in MuJoCo, steps 1.0 s of settle, and records:
load success, finite state, solver-failure flags (non-finite, radial or z
out of bounds per the v1.1 standard), and manipulated-object drift.
A scene passing this check is well-formed and stable at rest - nothing
more. Solvability (A2.2 executed pass) is a separate workstream.
"""
import json, sys
from pathlib import Path
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MANIP = {"block_stack": ["cube_a", "cube_b"], "eye_drops_to_basket": ["carton"],
         "ring_insert": ["ring"], "tool_insert": ["tool"],
         "cable_clip": ["cable_seg_0", "clip_block"]}

def main():
    man = json.loads((HERE / "suite_manifest.json").read_text())
    results = []
    for task, tinfo in man["tasks"].items():
        for entry in tinfo["scenes"]:
            for v in entry["variants"]:
                p = HERE / "scenes" / v["scene"]
                rec = {"scene": v["scene"], "task": task, "k": v["k"]}
                try:
                    model = mujoco.MjModel.from_xml_path(str(p))
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    start = {}
                    for b in MANIP[task]:
                        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                        start[b] = data.xpos[bid].copy()
                    for _ in range(2000):  # 1.0 s at 0.0005 s
                        mujoco.mj_step(model, data)
                    rec["finite"] = bool(np.all(np.isfinite(data.qpos)))
                    drift, oob = {}, False
                    for b, s in start.items():
                        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                        e = data.xpos[bid]
                        drift[b] = round(float(np.linalg.norm(e - s)), 5)
                        if np.linalg.norm(e[:2]) > 1.0 or e[2] < -0.05 or e[2] > 0.5:
                            oob = True
                    rec["drift_m"] = drift
                    rec["solver_failure"] = bool((not rec["finite"]) or oob)
                    rec["load"] = "ok"
                except Exception as ex:
                    rec["load"] = f"FAIL: {type(ex).__name__}: {ex}"
                    rec["solver_failure"] = True
                results.append(rec)
                print(rec["scene"], rec["load"], "solver_failure=", rec["solver_failure"], flush=True)
    (HERE / "settle_check.json").write_text(json.dumps(results, indent=1))
    bad = [r for r in results if r["load"] != "ok" or r["solver_failure"]]
    print(f"settle check: {len(results)-len(bad)}/{len(results)} clean; failures: {len(bad)}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
