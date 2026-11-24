#!/usr/bin/env python3
"""Generate Gemini prompts for the 1h training set (250 eps, 10 runs x 25)
from the training-demo audit's verdicts_master.csv + JUDGING_BRIEF.md.
Sheets ship to DGX gemini/sheets-train250/ as PNG before these run."""
import csv, os

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.expanduser("~/workspace/research_notes/zk0-bench003-design-20261119/training-demo-audit-20261123")
OUT = os.path.join(BASE, "gemini-staging", "prompts")
os.makedirs(OUT, exist_ok=True)

brief = open(os.path.join(AUDIT, "JUDGING_BRIEF.md")).read().rstrip()
rows = list(csv.DictReader(open(os.path.join(AUDIT, "verdicts_master.csv"))))
tasks = {}
for r in rows:
    tasks.setdefault(r["task"], []).append(int(r["episode_index"]))

runs = []
for task, idxs in sorted(tasks.items()):
    idxs.sort()
    for part in range(0, len(idxs), 25):
        chunk = idxs[part:part + 25]
        run = f"gemini_train250-{task}-{part // 25 + 1}"
        lines = [brief, "", "",
                 f"Judge exactly these {len(chunk)} episodes. For each one, open the contact-sheet "
                 f"image with your image/file reading tool and judge from the sheet only. "
                 f"Working directory: /home/openclaw/muse-work/zk0-tier10h-cert-20261123", ""]
        for i in chunk:
            lines.append(f"- ep{i} — gemini/sheets-train250/{task}__ep{i}.png — task: {task}")
        lines += ["", f"Output ONLY the {len(chunk)} verdict lines in the format specified above, one per episode, in the order listed. No preamble, no summary, no other text."]
        with open(os.path.join(OUT, run + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        runs.append(run)
print("\n".join(runs))
