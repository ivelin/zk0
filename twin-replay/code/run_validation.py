"""Study 001 twin validation runner (Builder C, Amendment 1 semantics).

Per included episode (Builder B initial-pose confidence medium+; cable_clip
governed by clip_block confidence with the cable layout coarse — see
report), builds K=5 variant scenes (seed 20261122) and runs all variants.
Episode REPRODUCED iff the Wilson score LOWER bound (z=1.96) of its
VALID-variant agreement fraction is >= 0.60 -- the founder-approved
twin-suite v2 standard, aligned 2026-11-25 (the former flat >=60%
fraction overclaimed at small variant counts). Secondary position
criterion is assessed on the nominal variant (v0). Outputs
validation/validation_results.json with per-task summaries, certified
variant sets, fragile episodes and named exclusions.
Usage: run_validation.py [--workers 6] [--render] [--tasks a,b]
"""
import json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import twin_params as P
import build_scenes as B
import replay as R

CONF_OK = {"medium", "high"}

def wilson_lower_bound(k, n, z=1.96):
    """Wilson score interval lower bound for k successes in n trials --
    the founder-approved twin-suite v2 standard (twin-suite-v2-
    preregistration-20261122.md: PASS iff the Wilson lower bound of
    pooled success is >= 60%; z=1.96 as in the v2 controls-battery
    analyzer, twin-suite-v2/controls_battery/analyze.py)."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h)

def one(job):
    task, epi, k, render = job
    try:
        return R.run_episode(task, epi, render=render, variant=k)
    except Exception as e:
        return {"task": task, "episode": epi, "variant": k,
                "error": f"{type(e).__name__}: {e}"[:300]}

def main():
    workers = 6; render = "--render" in sys.argv
    tasks_filter = None
    for a in sys.argv:
        if a.startswith("--workers"): workers = int(a.split("=")[1])
        if a.startswith("--tasks"): tasks_filter = a.split("=")[1].split(",")
    split = json.loads(P.SPLIT.read_text())
    rec_map = {}
    for task, v in split["tasks"].items():
        for rec in v["episodes"]:
            rec_map[(task, rec["episode_index"])] = rec["recorded_success"]
    included, excluded = [], []
    for task, v in split["tasks"].items():
        if tasks_filter and task not in tasks_filter: continue
        for rec in v["episodes"]:
            epi = rec["episode_index"]
            poses = B.load_episode_poses(task, epi)
            if poses is None:
                excluded.append({"task": task, "episode": epi, "reason": "no Builder B pose file"})
                continue
            conf = poses.get("confidence")
            if conf not in CONF_OK:
                excluded.append({"task": task, "episode": epi,
                                 "reason": f"initial-pose confidence={conf} (per-object: {poses.get('confidences')})"})
                continue
            fixtures = B.load_fixtures(task, poses)
            vlog = {"suite_version": P.SUITE_VERSION, "seed": P.VARIANT_SEED,
                    "param_hash": B.param_table_hash(), "task": task, "episode": epi,
                    "confidence": conf, "variants": []}
            for k in range(P.VARIANT_K):
                draws = B.make_draws(task, epi, poses, k)
                xml, _, _ = B.build(task, epi, poses, fixtures, var=draws)
                out = P.TWIN / "scenes" / f"{task}__ep{epi}_v{k}.xml"
                out.write_text(xml)
                if k == 0: (P.TWIN / "scenes" / f"{task}__ep{epi}.xml").write_text(xml)
                vlog["variants"].append({"k": k, "draws": draws, "scene": out.name})
            (P.TWIN / "scenes" / f"{task}__ep{epi}_variants.json").write_text(json.dumps(vlog, indent=1))
            included.append((task, epi))
    jobs = [(t, e, k, render) for (t, e) in included for k in range(P.VARIANT_K)]
    print(f"episodes included: {len(included)}; excluded: {len(excluded)}; variant jobs: {len(jobs)}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            results.append(r)
            if (i + 1) % 50 == 0: print(f"{i+1}/{len(jobs)}", flush=True)
    by_ep = {}
    for r in results:
        by_ep.setdefault((r["task"], r["episode"]), []).append(r)
    episodes_out, per_task = [], {}
    for (task, epi) in sorted(by_ep):
        rs = sorted(by_ep[(task, epi)], key=lambda r: r["variant"])
        rec_succ = rec_map[(task, epi)]
        valid, agreeing, certified, defects = [], [], [], []
        nominal = None
        for r in rs:
            if "error" in r:
                defects.append({"variant": r["variant"], "defect": "run error: " + r["error"]}); continue
            if r.get("build_defect"):
                defects.append({"variant": r["variant"], "defect": r["build_defect"]}); continue
            if r.get("solver_failure"):
                # Amendment 1 A2: solver failure is an exclusion ground;
                # excluded from agreement denominators, logged per variant.
                defects.append({"variant": r["variant"], "defect": "solver_failure"}); continue
            valid.append(r["variant"])
            agree = bool(r["sim_success"]) == bool(rec_succ)
            r["agree"] = agree
            if agree: agreeing.append(r["variant"]); certified.append(r["variant"])
            if r["variant"] == 0: nominal = r
        frac = len(agreeing) / len(valid) if valid else 0.0
        reproduced = bool(valid) and wilson_lower_bound(len(agreeing), len(valid)) >= 0.6
        ep_out = {"task": task, "episode": epi, "recorded_success": rec_succ,
                  "n_variants": len(rs), "valid_variants": valid,
                  "agreeing_variants": agreeing, "certified_variants": certified,
                  "build_defects": defects, "variant_agreement_frac": round(frac, 3),
                  "reproduced": reproduced, "fragile": not reproduced,
                  "nominal_planar_err_m": (nominal or {}).get("planar_err_m"),
                  "nominal_sim_success": (nominal or {}).get("sim_success"),
                  "nominal_secondary_pass": (nominal or {}).get("secondary_pass"),
                  "secondary_threshold_m": (nominal or {}).get("secondary_threshold_m"),
                  "n_solver_failures": sum(1 for d in defects if d["defect"] == "solver_failure"),
                  "variants_detail": rs}
        episodes_out.append(ep_out)
        d = per_task.setdefault(task, {"included": 0, "reproduced": 0, "fracs": [],
                                       "certified_total": 0, "valid_total": 0, "fragile": [],
                                       "solver_failures": 0, "secondary_passes": 0})
        d["solver_failures"] += ep_out["n_solver_failures"]
        d["secondary_passes"] += int(ep_out["nominal_secondary_pass"] is True)
        d["included"] += 1; d["reproduced"] += int(reproduced)
        d["fracs"].append(frac); d["certified_total"] += len(certified)
        d["valid_total"] += len(valid)
        if not reproduced: d["fragile"].append(epi)
    summary = {}
    for t, d in per_task.items():
        summary[t] = {"episodes_included": d["included"],
                      "episodes_reproduced": d["reproduced"],
                      "reproduction_rate": round(d["reproduced"] / d["included"], 4) if d["included"] else None,
                      "mean_variant_agreement": round(float(np.mean(d["fracs"])), 4) if d["fracs"] else None,
                      "certified_variants": d["certified_total"],
                      "valid_variants": d["valid_total"],
                      "fragile_episodes": d["fragile"],
                      "solver_failure_variants": d["solver_failures"],
                      "nominal_secondary_passes": d["secondary_passes"]}
    out = {"suite_version": P.SUITE_VERSION, "seed": P.VARIANT_SEED,
           "param_hash": B.param_table_hash(), "summary": summary,
           "excluded": excluded, "episodes": episodes_out}
    (P.TWIN / "validation" / "validation_results.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summary, indent=1))
    print("excluded:", json.dumps(excluded))

if __name__ == "__main__":
    main()
