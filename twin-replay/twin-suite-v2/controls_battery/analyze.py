#!/usr/bin/env python3
"""Controls battery Part B analysis: reads all rollouts.jsonl under out/,
emits analysis.json + printed tables. Aggregation per pre-reg S5: successes
pooled over episodes x variants, Wilson score intervals (z=1.96).
Solver-failure rollouts are EXCLUDED from denominators (v1.1 standard:
excluded, counted, reported -- never scored). Drake records with blank
frame_content (min non-clear fraction ~0 / unique colors <= 2 in any view)
are INVALIDATED: excluded and flagged, per the task brief."""
import json, math
from pathlib import Path

BAT = Path("/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/controls_battery")
TASKS = ["block_stack", "eye_drops_to_basket", "tool_insert", "cable_clip", "ring_insert"]
CKPTS = ["base", "fl_round_30_a5repair", "c_exp"]
CORNERS = ["pose_+x", "pose_-x", "pose_+y", "pose_-y", "friction_low", "friction_high",
           "mass_low", "mass_high"]
ENV_SCENES = {"block_stack": [499, 527, 546, 549], "eye_drops_to_basket": [1609]}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def verdict(k, n):
    lo, hi = wilson(k, n)
    if lo >= 0.60: return "PASS"
    if hi < 0.60: return "FAIL"
    return "INDETERMINATE"

def load(layer, ck):
    p = BAT / "out" / layer / ck / "rollouts.jsonl"
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def drake_valid(r):
    fc = r.get("frame_content")
    if not fc: return False
    try:
        mn = fc["min_non_clear_frac"]
        return all(v > 0.05 for v in mn.values()) and all(
            fc["chunk0"][v]["unique_colors"] > 10 for v in ("front", "top", "wrist"))
    except Exception:
        return False

def rate(recs):
    """(successes, n_excl_solver, n) over non-solver-failure records."""
    ok = [r for r in recs if not r.get("solver_failure")]
    return sum(1 for r in ok if r.get("sim_success")), len(recs) - len(ok), len(ok)

def main():
    L1 = {ck: load("L1", ck) for ck in CKPTS}
    L2 = {ck: load("L2", ck) for ck in CKPTS}
    L3raw = {ck: load("L3", ck) for ck in CKPTS}
    invalid_drake, L3 = [], {}
    for ck in CKPTS:
        L3[ck] = []
        for r in L3raw[ck]:
            if drake_valid(r): L3[ck].append(r)
            else: invalid_drake.append({"checkpoint": ck, "task": r["task"], "episode": r["episode"],
                                        "frame_content": r.get("frame_content")})
    out = {"invalid_drake_units": invalid_drake}
    # ---- L1 per-task / pooled rates + S5 verdicts
    l1tab = {}
    for ck in CKPTS:
        l1tab[ck] = {}
        allk = alln = alls = 0
        for t in TASKS:
            recs = [r for r in L1[ck] if r["task"] == t]
            k, s, n = rate(recs); lo, hi = wilson(k, n)
            l1tab[ck][t] = {"success": k, "n": n, "solver_failures": s,
                            "rate": (k / n if n else None), "wilson": [lo, hi],
                            "verdict": verdict(k, n) if n else "NO GATE (0 validated scenes)"}
            allk += k; alln += n; alls += s
        lo, hi = wilson(allk, alln)
        l1tab[ck]["POOLED"] = {"success": allk, "n": alln, "solver_failures": alls,
                               "rate": (allk / alln if alln else None), "wilson": [lo, hi],
                               "verdict": verdict(allk, alln) if alln else "NO DATA"}
    out["L1_nominal"] = l1tab
    # ---- ordering (S3.4)
    ordering = {}
    for t in TASKS + ["POOLED"]:
        rates = {ck: l1tab[ck][t]["rate"] for ck in CKPTS}
        ordering[t] = {"rates": rates,
                       "base_fails": (rates["base"] in (0, 0.0) or (rates["base"] is not None and rates["base"] <= 0.05)),
                       "strict_order": (None not in rates.values() and rates["base"] < rates["fl_round_30_a5repair"] < rates["c_exp"])}
    out["ordering"] = ordering
    # ---- cross-engine (L3 vs L1 nominal instance = variant '0', corner nominal)
    xe = {"paired": 0, "agree": 0, "disagreements": [], "per_checkpoint": {}}
    for ck in CKPTS:
        pk = pa = 0
        for r in L3[ck]:
            m = [x for x in L1[ck] if x["task"] == r["task"] and x["episode"] == r["episode"]
                 and str(x["variant"]) == "0" and x.get("corner", "nominal") == "nominal"]
            if not m: continue
            m = m[0]; pk += 1
            agree = (bool(m.get("sim_success")) == bool(r.get("sim_success"))
                     and bool(m.get("solver_failure")) == bool(r.get("solver_failure")))
            pa += agree
            if not agree:
                xe["disagreements"].append({"checkpoint": ck, "task": r["task"], "episode": r["episode"],
                                            "mujoco": {"success": m.get("sim_success"), "secondary": m.get("secondary_pass"), "solver": m.get("solver_failure")},
                                            "drake": {"success": r.get("sim_success"), "secondary": r.get("secondary_pass"), "solver": r.get("solver_failure")}})
        xe["per_checkpoint"][ck] = {"paired": pk, "agree": pa}
        xe["paired"] += pk; xe["agree"] += pa
    out["cross_engine"] = xe
    # ---- envelope stability: nominal verdict (L1 restricted to envelope scenes, K=5)
    # vs each corner's pooled outcome (L2, 1 rollout/scene) -- verdict + raw flips.
    env = {}
    for ck in CKPTS:
        env[ck] = {}
        for t, eps in ENV_SCENES.items():
            nom = [r for r in L1[ck] if r["task"] == t and r["episode"] in eps]
            k, s, n = rate(nom); nv = verdict(k, n) if n else "NO DATA"
            nom_inst = {(r["episode"]): r for r in L1[ck] if r["task"] == t and r["episode"] in eps
                        and str(r["variant"]) == "0"}
            corners = {}
            for c in CORNERS:
                recs = [r for r in L2[ck] if r["task"] == t and r.get("corner") == c]
                ck_k, ck_s, ck_n = rate(recs)
                flips = []
                for r in recs:
                    m = nom_inst.get(r["episode"])
                    if m and bool(m.get("sim_success")) != bool(r.get("sim_success")):
                        flips.append({"episode": r["episode"], "nominal_success": m.get("sim_success"),
                                      "corner_success": r.get("sim_success")})
                corners[c] = {"success": ck_k, "n": ck_n, "solver_failures": ck_s,
                              "verdict": verdict(ck_k, ck_n) if ck_n else "NO DATA",
                              "outcome_flips_vs_nominal_instance": flips}
            env[ck][t] = {"nominal_verdict": nv, "nominal_success": k, "nominal_n": n,
                          "nominal_solver_failures": s, "corners": corners,
                          "verdict_stable": all(corners[c]["verdict"] == nv for c in CORNERS)}
    out["envelope"] = env
    # ---- solver failure totals
    tot = {}
    for layer, D in (("L1", L1), ("L2", L2), ("L3", L3)):
        recs = [r for ck in CKPTS for r in D[ck]]
        tot[layer] = {"records": len(recs), "solver_failures": sum(1 for r in recs if r.get("solver_failure")),
                      "mean_wall_s": (sum(r.get("wall_s", 0) for r in recs) / len(recs) if recs else None)}
    out["totals"] = tot
    (BAT / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
