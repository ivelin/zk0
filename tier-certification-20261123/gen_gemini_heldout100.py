#!/usr/bin/env python3
"""Generate Gemini prompts for the held-out 100 (5 runs x 20, task-pure)
from the held-out audit's verdicts_master.csv + JUDGING_BRIEF.md."""
import csv, os

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.expanduser("~/workspace/research_notes/zk0-bench003-design-20261119/heldout-audit-20261123")
OUT = os.path.join(BASE, "gemini-staging", "prompts")
os.makedirs(OUT, exist_ok=True)

brief = open(os.path.join(AUDIT, "JUDGING_BRIEF.md")).read().rstrip()
# the audit brief says "contact-sheet PNG" already in spirit; keep verbatim
rows = list(csv.DictReader(open(os.path.join(AUDIT, "verdicts_master.csv"))))
tasks = {}
for r in rows:
    tasks.setdefault(r["task"], []).append(int(r["episode_index"]))

runs = []
for task, idxs in tasks.items():
    idxs.sort()
    run = f"gemini_heldout100-{task}"
    lines = [brief, "", "",
             f"Judge exactly these {len(idxs)} episodes. For each one, open the contact-sheet "
             f"image with your image/file reading tool and judge from the sheet only. "
             f"Working directory: /home/openclaw/muse-work/zk0-tier10h-cert-20261123", ""]
    for i in idxs:
        lines.append(f"- ep{i} — gemini/sheets-heldout/{task}__ep{i}.png — task: {task}")
    lines += ["", f"Output ONLY the {len(idxs)} verdict lines in the format specified above, one per episode, in the order listed. No preamble, no summary, no other text."]
    with open(os.path.join(OUT, run + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    runs.append(run)
print("\n".join(runs))
