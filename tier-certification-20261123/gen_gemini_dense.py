#!/usr/bin/env python3
"""Generate Gemini (Google) DENSE-sheet prompts for the bulk register
legs: Spa-Bench remainder (eps 601-1200; eps 1-600 already read on
6-frame sheets, reads on record), RoboColosseum cube-drawer +
screwdriver, rc2 (stack-bowls, stack-cubes, bb-push-green,
bb-flip-left). Sheets: gemini/sheets-dense/<prefix>__ep<idx>.png
(16-frame dense recipe, criteria Amendment 7). Prompts carry a binding
override block: dense layout + gripper-release standard (Amendment 5)
+ input mode. Models: the dense driver alternates Gemini (Google)
gemini-4-neon-xhigh / gemini-4-carbon-high per run."""
import csv, os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "gemini-staging", "prompts-dense")
os.makedirs(OUT, exist_ok=True)
WD = "/home/openclaw/muse-work/zk0-tier10h-cert-20261123"

OVERRIDE = """BINDING INPUT + STANDARD NOTES (supersede any sheet description above):
- Each sheet is a DENSE 16-frame sheet (PNG): FOUR rows of 8 frames.
  Rows 1-2 are camera 1 (row 1 = frames 1-8, row 2 = frames 9-16);
  rows 3-4 are camera 2 in the same arrangement. Frames run
  left-to-right in time at 0%, 2%, 5%, 12%, 25%, 40%, 55%, 68%, 78%,
  85%, 90%, 94%, 96.5%, 98.2%, 99.3%, ~100% of the episode — a few
  start frames (scene setup), rising density toward the end, and the
  final frames are always included. Judge the END STATE.
- RELEASE STANDARD (binding): a PASS requires the gripper VISIBLY
  OPEN / released at episode end. If the episode ends with the
  gripper still closed on / holding the object (no release shown),
  return INDETERMINATE regardless of how the object sits. State the
  final gripper state in your one-line evidence."""

def brief(name):
    t = open(os.path.join(BASE, name)).read().rstrip()
    return t.replace("contact sheet JPG", "contact sheet PNG").replace("ONE contact\nsheet JPG", "ONE contact\nsheet PNG")

def make_runs(run_prefix, brief_text, eps, sheet_prefix, size=25):
    runs = []
    for part in range(0, len(eps), size):
        chunk = eps[part:part + size]
        run = f"{run_prefix}-{part // size + 1:02d}"
        lines = [brief_text, "", OVERRIDE, "", "",
                 f"Judge exactly these {len(chunk)} episodes. For each one, open the contact-sheet "
                 f"image with your image/file reading tool and judge from the sheet only. "
                 f"Working directory: {WD}", ""]
        for idx, tstr in chunk:
            lines.append(f"- ep{idx} — gemini/sheets-dense/{sheet_prefix}__ep{idx}.png — task: \"{tstr}\"")
        lines += ["", f"Output ONLY the {len(chunk)} verdict lines in the format specified above, one per episode, in the order listed. No preamble, no summary, no other text."]
        with open(os.path.join(OUT, run + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        runs.append(run)
    return runs

rows = list(csv.DictReader(open(os.path.join(BASE, "register.csv"))))
def eps_for(dataset, lo=None, hi=None):
    sel = [r for r in rows if r["source_dataset"] == dataset and r["primary_verdict"]]
    sel.sort(key=lambda r: int(r["episode_index"]))
    out = [(int(r["episode_index"]), r["task_string"]) for r in sel]
    if lo is not None:
        out = [e for e in out if lo <= e[0] <= hi]
    return out

all_runs = {}
all_runs["gemini_spa-dense"] = make_runs("gemini_spa-dense", brief("brief-spa.md"),
    eps_for("Spa-Bench/spa-bench-training-teleoperation-1200", 601, 1200), "spa-bench")
all_runs["gemini_rc-cube-drawer-dense"] = make_runs("gemini_rc-cube-drawer-dense", brief("brief-rc.md"),
    eps_for("RoboColosseum/so101-cube-drawer"), "rc-cube-drawer")
all_runs["gemini_rc-screwdriver-dense"] = make_runs("gemini_rc-screwdriver-dense", brief("brief-rc.md"),
    eps_for("RoboColosseum/so101-remove-screwdriver"), "rc-screwdriver")
all_runs["gemini_rc-stack-bowls-dense"] = make_runs("gemini_rc-stack-bowls-dense", brief("brief-rc-stack.md"),
    eps_for("RoboColosseum/so101-stack-bowls"), "rc-stack-bowls")
all_runs["gemini_rc-stack-cubes-dense"] = make_runs("gemini_rc-stack-cubes-dense", brief("brief-rc-stack.md"),
    eps_for("RoboColosseum/so101-stack-cubes"), "rc-stack-cubes")
all_runs["gemini_bb-push-green-dense"] = make_runs("gemini_bb-push-green-dense", brief("brief-busybox-singles.md"),
    eps_for("armnet/busybox_push_green_button"), "bb-push-green")
all_runs["gemini_bb-flip-left-dense"] = make_runs("gemini_bb-flip-left-dense", brief("brief-busybox-singles.md"),
    eps_for("armnet/busybox_flip_left_switch_off"), "bb-flip-left")

with open(os.path.join(BASE, "gemini-staging", "runs-dense.txt"), "w") as f:
    for group in all_runs.values():
        for r in group:
            f.write(r + "\n")
print({k: len(v) for k, v in all_runs.items()})
print("total runs:", sum(len(v) for v in all_runs.values()))
