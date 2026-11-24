#!/usr/bin/env python3
"""Append a new source's rows to register.csv from verdict files.

Usage:
  add_source.py --source-dataset <hf id> --source-revision <sha> \
    --key-prefix <prefix> --episodes-csv <prefix>_episodes.csv --fps <n> \
    --sheet-prefix <prefix> --notes "<notes template>" \
    --task-name "<fixed task name>" | --task-map <json file {task_string: name}> \
    --primary <glob> [--cleanliness <glob>] [--verifier <glob>]

Verdict files hold lines: ep<idx> | VERDICT | <evidence>
Cleanliness files hold:   ep<idx> | GRADE [| <incidents>]
Rows already in the register (same episode_key) are skipped.
subset is derived: PASS+CLEAN -> imitation core; PASS+RECOVERED ->
recovery-training; PASS+INCIDENT -> pass evidence only; else blank.
contested is set when the verifier verdict differs from primary.
"""
import argparse, csv, glob, json, os, re

BASE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(BASE, "register.csv")

def parse_verdicts(pattern, kinds):
    out = {}
    if not pattern:
        return out
    for path in glob.glob(os.path.join(BASE, pattern)):
        for line in open(path):
            m = re.match(r"ep(\d+) \| (" + "|".join(kinds) + r")(?: \| (.*))?", line.rstrip("\n"))
            if m:
                out[int(m.group(1))] = (m.group(2), (m.group(3) or "").strip())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dataset", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--key-prefix", required=True)
    ap.add_argument("--episodes-csv", required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--sheet-prefix", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--task-name", default=None)
    ap.add_argument("--task-map", default=None)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--cleanliness", default=None)
    ap.add_argument("--verifier", default=None)
    a = ap.parse_args()

    task_map = json.load(open(a.task_map)) if a.task_map else {}
    eps = {int(r["episode_index"]): r for r in csv.DictReader(open(a.episodes_csv))}
    primary = parse_verdicts(a.primary, ["PASS", "FAIL", "INDETERMINATE"])
    clean = parse_verdicts(a.cleanliness, ["CLEAN", "RECOVERED", "INCIDENT"])
    verifier = parse_verdicts(a.verifier, ["PASS", "FAIL", "INDETERMINATE"])

    rows = list(csv.DictReader(open(REG)))
    fields = list(rows[0].keys())
    have = {r["episode_key"] for r in rows}
    added, skipped = 0, 0
    for idx in sorted(eps):
        key = f"{a.key_prefix}:{idx}"
        if key in have:
            skipped += 1
            continue
        r = eps[idx]
        pv, pev = primary.get(idx, ("", ""))
        cv, cev = clean.get(idx, ("", ""))
        vv, vev = verifier.get(idx, ("", ""))
        subset = ""
        if pv == "PASS":
            subset = {"CLEAN": "imitation core", "RECOVERED": "recovery-training",
                      "INCIDENT": "pass evidence only"}.get(cv, "")
        frames = int(r["length_frames"])
        rows.append({
            "episode_key": key,
            "source_dataset": a.source_dataset,
            "source_revision": a.source_revision,
            "episode_index": idx,
            "task": a.task_name or task_map.get(r["task_string"], ""),
            "task_string": r["task_string"],
            "frames": frames,
            "duration_s": f"{frames / a.fps:.2f}",
            "primary_verdict": pv,
            "primary_evidence": pev,
            "verifier_verdict": vv,
            "verifier_evidence": vev,
            "grok_verdict": "",
            "grok_evidence": "",
            "cleanliness": cv,
            "incidents": cev,
            "subset": subset,
            "contested": "yes" if (vv and pv and vv != pv) else "",
            "human_status": "PENDING HUMAN APPROVAL",
            "sheet": f"sheets/{a.sheet_prefix}__ep{idx}.jpg",
            "notes": a.notes,
            "verdict_origin": "tier10h-certification program 2026-11-23 (primary + cleanliness" + (" + re-judge" if vv else "") + ")",
        })
        added += 1
    with open(REG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    missing_primary = [i for i in sorted(eps) if i not in primary]
    print(json.dumps({"added": added, "skipped_existing": skipped,
                      "primary_parsed": len(primary), "cleanliness_parsed": len(clean),
                      "verifier_parsed": len(verifier), "missing_primary": missing_primary}))

if __name__ == "__main__":
    main()
