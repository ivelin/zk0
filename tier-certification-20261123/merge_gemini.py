#!/usr/bin/env python3
"""Merge Gemini-leg verdicts (agy, model pinned) into the tier register
and/or the held-out audit master.

Verdict line files hold: [path:]ep<idx> | PASS|FAIL|INDETERMINATE | <evidence>
Register merges are KEY-EXPLICIT (episode indices collide across sources).
Held-out master merges match on episode_index (unique within that file).
Adds gemini_verdict / gemini_evidence / gemini_model columns if absent.
Also adds per-read model attribution columns to the register
(primary_model / verifier_model / grok_model) per the rotation rule,
filled uniformly: primary+verifier reads are Muse judge subagents;
Grok reads are Grok Build CLI 1.0.88. Gemini never overwrites or
resolves anything: contested flags are left exactly as they are.
"""
import csv, json, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gemini-3.8-flash-high (agy 1.5.15)"
CLAUDE = "Muse Spark (judge subagent)"
GROK = "Grok Build CLI 1.0.88"

def parse_lines(paths):
    out = {}
    for p in paths:
        for line in open(p):
            m = re.search(r"ep(\d+) \| (PASS|FAIL|INDETERMINATE) \| (.*)$", line.rstrip("\n"))
            if m:
                out[int(m.group(1))] = (m.group(2), m.group(3).strip())
    return out

def ensure_cols(rows, fields, cols):
    for c in cols:
        if c not in fields:
            fields.append(c)
            for r in rows:
                r[c] = ""

def write_back(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def merge_register(keymap):
    reg = os.path.join(BASE, "register.csv")
    rows = list(csv.DictReader(open(reg)))
    fields = list(rows[0].keys())
    ensure_cols(rows, fields, ["gemini_verdict", "gemini_evidence", "gemini_model",
                               "primary_model", "verifier_model", "grok_model"])
    by_key = {r["episode_key"]: r for r in rows}
    n = 0
    for idx, (verdict, evidence) in sorted(keymap["verdicts"].items()):
        key = keymap["keys"][str(idx)]
        r = by_key[key]
        r["gemini_verdict"] = verdict
        r["gemini_evidence"] = evidence
        r["gemini_model"] = MODEL
        n += 1
    for r in rows:
        if r["primary_verdict"] and not r["primary_model"]:
            r["primary_model"] = CLAUDE
        if r["verifier_verdict"] and not r["verifier_model"]:
            r["verifier_model"] = CLAUDE
        if r["grok_verdict"] and not r["grok_model"]:
            r["grok_model"] = GROK
    write_back(reg, rows, fields)
    print(json.dumps({"register_rows_updated": n}))

def merge_heldout_master(verdicts):
    path = os.path.expanduser("~/workspace/research_notes/zk0-bench003-design-20261119/heldout-audit-20261123/verdicts_master.csv")
    rows = list(csv.DictReader(open(path)))
    fields = list(rows[0].keys())
    ensure_cols(rows, fields, ["gemini_verdict", "gemini_evidence", "gemini_model"])
    n = 0
    for r in rows:
        idx = int(r["episode_index"])
        if idx in verdicts:
            r["gemini_verdict"], r["gemini_evidence"] = verdicts[idx]
            r["gemini_model"] = MODEL
            n += 1
    write_back(path, rows, fields)
    print(json.dumps({"heldout_master_rows_updated": n}))

if __name__ == "__main__":
    out = os.path.join(BASE, "gemini-staging", "out")
    reg_verdicts = parse_lines([os.path.join(out, "register-contested-lines.txt")])
    merge_register({"verdicts": reg_verdicts,
                    "keys": {"859": "spa-bench:859", "573": "armnetbench:573"}})
    heldout_verdicts = parse_lines([os.path.join(out, "contested-cable-lines.txt"),
                                    os.path.join(out, "contested-rest-lines.txt")])
    merge_heldout_master(heldout_verdicts)
