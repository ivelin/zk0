#!/usr/bin/env python3
"""Build founder review packs per source from register.csv.

Pack layout (review-pack/<source>/):
  fail/                    every primary FAIL sheet
  contested/               every contested sheet (any leg disagreement)
  pass_sample_clean/       seeded PASS+CLEAN sample (imitation core)
  pass_sample_recovered/   seeded PASS+RECOVERED sample
  pass_sample_incident/    seeded PASS+INCIDENT sample (pass evidence only)
  README.md                counts, seed, per-file verdict summary
Seed 20261123 (program seed), sample cap 12 per subset. Sheets are
copied from the register's sheet paths. Everything in the pack remains
PENDING HUMAN APPROVAL.
"""
import csv, os, random, shutil, sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(BASE, "register.csv")
PACK = os.path.join(BASE, "review-pack")
SEED = 20261123
CAP = 12

def build(source_filter, pack_name, rows):
    sel = [r for r in rows if source_filter(r)]
    if not sel:
        print(f"{pack_name}: no rows, skipped")
        return
    root = os.path.join(PACK, pack_name)
    for d in ("fail", "contested", "pass_sample_clean", "pass_sample_recovered", "pass_sample_incident"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    rng = random.Random(SEED)
    lines = [f"# Review pack — {pack_name}", "",
             f"Source rows: {len(sel)}. Seed {SEED}, PASS-sample cap {CAP} per cleanliness subset.",
             "Everything here remains PENDING HUMAN APPROVAL.", ""]
    def place(r, folder):
        src = os.path.join(BASE, r["sheet"])
        dst = os.path.join(root, folder, os.path.basename(r["sheet"]))
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            lines.append(f"- {folder}/{os.path.basename(r['sheet'])} — {r['episode_key']} "
                         f"primary {r['primary_verdict'] or '—'} / verifier {r['verifier_verdict'] or '—'} "
                         f"/ grok {r['grok_verdict'] or '—'} / gemini {r['gemini_verdict'] or '—'} "
                         f"/ cleanliness {r['cleanliness'] or '—'}")
        else:
            lines.append(f"- MISSING SHEET {r['sheet']} — {r['episode_key']}")
    fails = [r for r in sel if r["primary_verdict"] == "FAIL"]
    for r in fails:
        place(r, "fail")
    contested = [r for r in sel if r["contested"] == "yes"]
    for r in contested:
        place(r, "contested")
    for grade, folder in (("CLEAN", "pass_sample_clean"), ("RECOVERED", "pass_sample_recovered"),
                          ("INCIDENT", "pass_sample_incident")):
        pool = sorted([r for r in sel if r["primary_verdict"] == "PASS" and r["cleanliness"] == grade],
                      key=lambda r: r["episode_key"])
        sample = rng.sample(pool, min(CAP, len(pool))) if pool else []
        for r in sorted(sample, key=lambda r: r["episode_key"]):
            place(r, folder)
        lines.insert(4, f"{folder}: {len(sample)} of {len(pool)} PASS+{grade} episodes.")
    lines.insert(4, f"contested: {len(contested)}; fail: {len(fails)}.")
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{pack_name}: rows={len(sel)} fail={len(fails)} contested={len(contested)}")

if __name__ == "__main__":
    rows = list(csv.DictReader(open(REG)))
    build(lambda r: r["episode_key"].startswith("spa-bench:"), "spa-bench", rows)
    build(lambda r: r["episode_key"].startswith("rc-cube-drawer:"), "rc-cube-drawer", rows)
    build(lambda r: r["episode_key"].startswith("rc-screwdriver:"), "rc-screwdriver", rows)
    if "--arm-heldout" in sys.argv:
        build(lambda r: r["task"] in ("eye_drops_to_shelf", "tool_removal", "cable_unclip"),
              "armnetbench-heldout", rows)
