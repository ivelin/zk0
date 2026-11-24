#!/usr/bin/env python3
"""Merge Grok-leg verdicts into register.csv.

Inputs:
  grok-staging/prompts/<run>.txt  — run -> [(ep_idx, sheet)] mapping
  grok-verdicts-raw.txt           — lines '<run>.raw:ep<idx> | VERDICT | evidence'
                                    (grep -H output pulled from the DGX)
  grok-rc.txt                     — lines '<run> <rc>' from the .done files
  register.csv                    — updated in place (grok_verdict,
                                    grok_evidence, contested)

Prints a JSON summary: per-source agreement, contested list, defective
runs (rc != 0 or verdict-line count != episode count in the prompt).
Does NOT resolve any disagreement; differing Grok verdicts only set
contested=yes.
"""
import csv, json, os, re, sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(BASE, "grok-staging", "prompts")
RAW = os.path.join(BASE, "grok-verdicts-raw.txt")
RCF = os.path.join(BASE, "grok-rc.txt")
REG = os.path.join(BASE, "register.csv")

SHEET2KEY = [
    (re.compile(r"^spa__ep(\d+)\.jpg$"), "spa-bench"),
    (re.compile(r"^rc-cube-drawer__ep(\d+)\.jpg$"), "rc-cube-drawer"),
    (re.compile(r"^rc-screwdriver__ep(\d+)\.jpg$"), "rc-screwdriver"),
    (re.compile(r"^armnetbench-heldout__ep(\d+)\.jpg$"), "armnetbench"),
]

def key_for(sheet, ep):
    for pat, prefix in SHEET2KEY:
        if pat.match(sheet):
            return f"{prefix}:{ep}"
    raise ValueError(f"unknown sheet {sheet}")

# 1. run -> {ep_idx: episode_key}
run_eps = {}
for fn in sorted(os.listdir(PROMPTS)):
    if not fn.endswith(".txt"):
        continue
    run = fn[:-4]
    eps = {}
    for line in open(os.path.join(PROMPTS, fn)):
        m = re.match(r"- ep(\d+) — sheets/(\S+?) —", line) or re.match(r"- ep(\d+) — sheets/(\S+)\s*$", line)
        if m:
            ep, sheet = int(m.group(1)), m.group(2)
            eps[ep] = key_for(sheet, ep)
    if eps:
        run_eps[run] = eps

# 2. rc per run
rcs = {}
for line in open(RCF):
    parts = line.split()
    if len(parts) == 2:
        rcs[parts[0]] = int(parts[1])

# 3. verdicts: episode_key -> (verdict, evidence)
verdicts = {}
dupes = []
for line in open(RAW):
    m = re.match(r"(grok_[a-z0-9-]+)\.raw:ep(\d+) \| (PASS|FAIL|INDETERMINATE) \| (.*)", line.rstrip("\n"))
    if not m:
        continue
    run, ep, verdict, evidence = m.group(1), int(m.group(2)), m.group(3), m.group(4).strip()
    if run not in run_eps or ep not in run_eps[run]:
        continue
    key = run_eps[run][ep]
    if key in verdicts:
        dupes.append(key)
    verdicts[key] = (verdict, evidence)

# 4. defective runs
defective = []
for run, eps in sorted(run_eps.items()):
    got = sum(1 for ep in eps if eps[ep] in verdicts)
    rc = rcs.get(run)
    if rc != 0 or got != len(eps):
        defective.append({"run": run, "rc": rc, "episodes": len(eps), "verdicts": got})

# 5. merge
rows = list(csv.DictReader(open(REG)))
fields = list(rows[0].keys())
by_key = {r["episode_key"]: r for r in rows}
missing_rows = [k for k in verdicts if k not in by_key]
for key, (verdict, evidence) in verdicts.items():
    if key in by_key:
        by_key[key]["grok_verdict"] = verdict
        by_key[key]["grok_evidence"] = evidence

agree = defaultdict(lambda: [0, 0])  # source -> [agree, total]
new_contested = []
for key, (verdict, _ev) in verdicts.items():
    r = by_key.get(key)
    if not r:
        continue
    src = key.split(":")[0]
    agree[src][1] += 1
    if r["primary_verdict"] == verdict:
        agree[src][0] += 1
    else:
        if r["contested"] != "yes":
            r["contested"] = "yes"
            new_contested.append({"key": key, "primary": r["primary_verdict"], "grok": verdict})
        else:
            new_contested.append({"key": key, "primary": r["primary_verdict"], "grok": verdict, "already_contested": True})

with open(REG, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

summary = {
    "runs_total": len(run_eps),
    "runs_with_rc": len(rcs),
    "verdicts_merged": len(verdicts),
    "dupes": dupes,
    "defective_runs": defective,
    "missing_register_rows": missing_rows,
    "agreement": {s: {"agree": a, "total": t} for s, (a, t) in sorted(agree.items())},
    "new_contested": new_contested,
}
print(json.dumps(summary, indent=1))
