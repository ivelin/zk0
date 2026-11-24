#!/usr/bin/env python3
"""Build the spec-only FULL-suite solvability register (Amendment 3).

Founder rules bound (2026-11-23): scene solution rule 13:05 (a scene
counts ONLY with a proven, certified solution trajectory), backup
13:08, certification 13:09 (a controller pass = PROVEN; only separate-
context verification = CERTIFIED, and only CERTIFIED scenes enter
denominators), fair-shot 13:11 (bounded attempt budget per scene,
attempts recorded, budget-exhausted scenes logged UNPROVEN), one
solution per (scene, task) 13:12 (the register names ONE canonical
solution trajectory per validated scene; further passing runs are
recorded as duplicates in the attempt log and are not separately
backed up).

Attempts: Route A = every admitted controller version x nominal k=0;
Route B = best controller x sampled variants k=1..4; Route C = best
controller x pose corners k=5..8 (precision tasks only). Budget and
operationalization: attempt-budget-and-rules-20261123.md.

Certification merge: an optional JSON file (argv after settle files,
prefixed cert:) maps scene_id -> {"certified": bool, "record": str}
from the independent verification; scenes absent from it are
"pending".

Usage: build_register.py <results_dir> <manifest.json> <settle.json...>
       [cert:<certification.json>] > register.json
"""
import json, sys, glob
from pathlib import Path

res_dir = Path(sys.argv[1])
man = json.loads(Path(sys.argv[2]).read_text())
settle_files, cert = [], {}
for a in sys.argv[3:]:
    if a.startswith("cert:"):
        cert = json.loads(Path(a[5:]).read_text())
    else:
        settle_files.append(a)

settle = {}
for f in settle_files:
    for r in json.loads(Path(f).read_text()):
        settle[r["scene"]] = r

VER_ORDER = {"rev21": 0, "frozen": 1, "rev3": 2}

def ver_key(controller_label):
    if controller_label.startswith("rev2.1"):
        return "rev21"
    if controller_label.startswith("frozen"):
        return "frozen"
    if controller_label.startswith("rev3"):
        return "rev3"
    return "other"

runs = {}
for f in sorted(glob.glob(str(res_dir / "*__spec*.json"))):
    r = json.loads(Path(f).read_text())
    if "smoke" in Path(f).name:
        continue
    k = int(str(r.get("scene_variant", "k=0")).split("=")[1].split()[0])
    runs.setdefault(r["scene_id"], []).append({
        "controller": r["controller"], "version": ver_key(r["controller"]),
        "variant_k": k, "verdict": r.get("verdict"),
        "sim_success": bool(r.get("sim_success")),
        "solver_failure": r.get("solver_failure"),
        "checks": r.get("checks"), "file": Path(f).name})

tasks_out = {}
for task, tinfo in man["tasks"].items():
    scenes_out = []
    counts = {"attempted": 0, "proven_nominal": 0, "variant_only_solutions": 0,
              "certified": 0, "budget_exhausted_unproven": 0, "in_progress": 0}
    esc_done = (res_dir / f"escalation_done_{task}.marker").exists()
    for entry in tinfo["scenes"]:
        sid = entry["scene_id"]
        att = sorted(runs.get(sid, []),
                     key=lambda a: (a["variant_k"], VER_ORDER.get(a["version"], 9)))
        passes = [a for a in att if a["sim_success"]]
        nominal_passes = [a for a in passes if a["variant_k"] == 0]
        proven_nominal = bool(nominal_passes)
        # canonical solution (13:12): nominal passes first, then the
        # program's controller order, then lowest variant k.
        canon = None
        if passes:
            canon = sorted(passes, key=lambda a: (a["variant_k"] != 0,
                                                  VER_ORDER.get(a["version"], 9),
                                                  a["variant_k"]))[0]
        c = cert.get(sid)
        certified = bool(c and c.get("certified"))
        if att:
            counts["attempted"] += 1
        if proven_nominal:
            counts["proven_nominal"] += 1
        elif passes:
            counts["variant_only_solutions"] += 1
        if certified:
            counts["certified"] += 1
        if att and not passes:
            if esc_done:
                counts["budget_exhausted_unproven"] += 1
            else:
                counts["in_progress"] += 1
        scenes_out.append({
            "scene_id": sid,
            "seed": entry["seed"],
            "proven_nominal": proven_nominal,
            "solution_within_envelope": bool(passes) and not proven_nominal,
            "canonical_solution": ({"controller": canon["controller"],
                                    "variant_k": canon["variant_k"],
                                    "record_file": canon["file"],
                                    "checks": canon["checks"]} if canon else None),
            "duplicate_solution_records": (
                [a["file"] for a in passes if canon and a["file"] != canon["file"]]),
            "certification": ({"status": "certified", "record": c.get("record")}
                              if certified else
                              {"status": "failed_verification", "record": c.get("record")}
                              if c else {"status": "pending"}),
            "attempts": att,
            "status": ("certified" if certified else
                       "proven_nominal" if proven_nominal else
                       "solution_within_envelope" if passes else
                       "budget_exhausted_unproven" if (att and esc_done) else
                       "in_progress" if att else "not_attempted"),
        })
    tasks_out[task] = {"denominator_nominal": tinfo["denominator_nominal"],
                       "counts": counts, "scenes": scenes_out}

nom_files = [v["scene"] for t in man["tasks"].values() for e in t["scenes"]
             for v in e["variants"] if v["k"] == 0]
settle_block = {
    "nominal_variants": {"checked": len(nom_files),
                         "clean": sum(1 for n in nom_files
                                      if settle.get(n, {}).get("load") == "ok"
                                      and not settle.get(n, {}).get("solver_failure", True))},
    "all_variants": {"checked": len(settle),
                     "clean": sum(1 for r in settle.values()
                                  if r.get("load") == "ok" and not r.get("solver_failure", True))}}

register = {
    "schema": "twin-suite-v2/spec-only-solvability-register/2",
    "suite": man["suite"],
    "master_seed": man["master_seed"],
    "rules": "Founder rules 2026-11-23: 13:05 scene solution rule; 13:08 SSOT backup; "
             "13:09 independent certification before denominators; 13:11 fair-shot budget; "
             "13:12 one solution per (scene, task). Budget + operationalization: "
             "attempt-budget-and-rules-20261123.md",
    "generation": "ORACLE-GUIDED (A2.2 route b); controllers global, no per-scene patches; "
                  "controllers ran with gt=None (no demo exists at Tier C)",
    "settle_check": settle_block,
    "tasks": tasks_out,
}
print(json.dumps(register, indent=1, default=str))
