#!/usr/bin/env python3
"""Gemini (Google) DENSE-sheet re-judge prompts for the 250 training
episodes (second-judge completion under the sharpened overnight
target; video quota cannot carry 250 per-episode calls — dense sheets
per criteria Amendment 7, input mode recorded). Brief: the training
audit's JUDGING_BRIEF.md (frozen criteria) + binding override block.
Sheets: gemini/sheets-dense-arm/armnet__ep<idx>.png (front+top)."""
import csv, os

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIT = "/home/hatch/workspace/research_notes/zk0-bench003-design-20261119/training-demo-audit-20261123"
OUT = os.path.join(BASE, "gemini-staging", "prompts-dense")
os.makedirs(OUT, exist_ok=True)
WD = "/home/openclaw/muse-work/zk0-tier10h-cert-20261123"

OVERRIDE = """BINDING INPUT + STANDARD NOTES (supersede any sheet description above):
- Each sheet is a DENSE 16-frame sheet (PNG): FOUR rows of 8 frames.
  Rows 1-2 are the FRONT camera (row 1 = frames 1-8, row 2 = frames
  9-16); rows 3-4 are the TOP camera in the same arrangement. Frames
  run left-to-right in time at 0%, 2%, 5%, 12%, 25%, 40%, 55%, 68%,
  78%, 85%, 90%, 94%, 96.5%, 98.2%, 99.3%, ~100% of the episode — a
  few start frames (scene setup), rising density toward the end, and
  the final frames are always included. Judge the END STATE.
- RELEASE STANDARD (binding): a PASS requires the gripper VISIBLY
  OPEN / released at episode end. If the episode ends with the
  gripper still closed on / holding the object (no release shown),
  return INDETERMINATE regardless of how the object sits. State the
  final gripper state in your one-line evidence."""

brief = open(os.path.join(AUDIT, "JUDGING_BRIEF.md")).read().rstrip()
brief = brief.replace("contact sheet JPG", "contact sheet PNG")
rows = list(csv.DictReader(open(os.path.join(BASE, "register.csv"))))
sel = [r for r in rows if r["source_dataset"] == "armnet/armnetbench_v01_lerobot_so101"
       and r["task"] in ("block_stack", "eye_drops_to_basket", "ring_insert", "tool_insert", "cable_clip")
       and r["primary_verdict"]]
sel.sort(key=lambda r: int(r["episode_index"]))
eps = [(int(r["episode_index"]), r["task_string"]) for r in sel]
runs = []
for part in range(0, len(eps), 25):
    chunk = eps[part:part + 25]
    run = f"gemini_traindense-{part // 25 + 1:02d}"
    lines = [brief, "", OVERRIDE, "", "",
             f"Judge exactly these {len(chunk)} episodes. For each one, open the contact-sheet "
             f"image with your image/file reading tool and judge from the sheet only. "
             f"Working directory: {WD}", ""]
    for idx, tstr in chunk:
        lines.append(f"- ep{idx} — gemini/sheets-dense-arm/armnet__ep{idx}.png — task: \"{tstr}\"")
    lines += ["", f"Output ONLY the {len(chunk)} verdict lines in the format specified above, one per episode, in the order listed. No preamble, no summary, no other text."]
    with open(os.path.join(OUT, run + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    runs.append(run)
print("training rows:", len(eps), "runs:", len(runs))
with open(os.path.join(BASE, "gemini-staging", "runs-traindense.txt"), "w") as f:
    f.write("\n".join(runs) + "\n")
