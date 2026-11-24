#!/usr/bin/env python3
"""Generate Gemini prompts for stage 4 (Spa-Bench + RoboColosseum
cube-drawer/screwdriver) and stage 5 (rc2 sources: stack-bowls,
stack-cubes, bb-push-green, bb-flip-left), mirroring
gen_gemini_train250.py's format. Sheets ship to the DGX as PNG under
gemini/sheets-spa/, gemini/sheets-rc/, gemini/sheets-rc2/ before the
runs execute. Episode/task data comes from register.csv (only rows
with a primary verdict and no Gemini verdict yet are included).
"""
import csv, os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "gemini-staging", "prompts")
os.makedirs(OUT, exist_ok=True)
WD = "/home/openclaw/muse-work/zk0-tier10h-cert-20261123"

def brief(name):
    t = open(os.path.join(BASE, name)).read().rstrip()
    return t.replace("contact sheet JPG", "contact sheet PNG").replace("ONE contact\nsheet JPG", "ONE contact\nsheet PNG")

def make_runs(run_prefix, brief_text, eps, sheet_dir, sheet_prefix, size=25):
    """eps: list of (episode_index, task_string). Returns run names."""
    runs = []
    for part in range(0, len(eps), size):
        chunk = eps[part:part + size]
        run = f"{run_prefix}-{part // size + 1:02d}"
        lines = [brief_text, "", "",
                 f"Judge exactly these {len(chunk)} episodes. For each one, open the contact-sheet "
                 f"image with your image/file reading tool and judge from the sheet only. "
                 f"Working directory: {WD}", ""]
        for idx, tstr in chunk:
            lines.append(f"- ep{idx} — gemini/{sheet_dir}/{sheet_prefix}__ep{idx}.png — task: \"{tstr}\"")
        lines += ["", f"Output ONLY the {len(chunk)} verdict lines in the format specified above, one per episode, in the order listed. No preamble, no summary, no other text."]
        with open(os.path.join(OUT, run + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        runs.append(run)
    return runs

rows = list(csv.DictReader(open(os.path.join(BASE, "register.csv"))))
def eps_for(dataset):
    sel = [r for r in rows if r["source_dataset"] == dataset and r["primary_verdict"] and not r["gemini_verdict"]]
    sel.sort(key=lambda r: int(r["episode_index"]))
    return [(int(r["episode_index"]), r["task_string"]) for r in sel]

all_runs = {}
# Stage 4
all_runs["gemini_spa"] = make_runs("gemini_spa", brief("brief-spa.md"),
    eps_for("Spa-Bench/spa-bench-training-teleoperation-1200"), "sheets-spa", "spa")
all_runs["gemini_rc-cube-drawer"] = make_runs("gemini_rc-cube-drawer", brief("brief-rc.md"),
    eps_for("RoboColosseum/so101-cube-drawer"), "sheets-rc", "rc-cube-drawer")
all_runs["gemini_rc-screwdriver"] = make_runs("gemini_rc-screwdriver", brief("brief-rc.md"),
    eps_for("RoboColosseum/so101-remove-screwdriver"), "sheets-rc", "rc-screwdriver")
# Stage 5
all_runs["gemini_rc-stack-bowls"] = make_runs("gemini_rc-stack-bowls", brief("brief-rc-stack.md"),
    eps_for("RoboColosseum/so101-stack-bowls"), "sheets-rc2", "rc-stack-bowls")
all_runs["gemini_rc-stack-cubes"] = make_runs("gemini_rc-stack-cubes", brief("brief-rc-stack.md"),
    eps_for("RoboColosseum/so101-stack-cubes"), "sheets-rc2", "rc-stack-cubes")
all_runs["gemini_bb-push-green"] = make_runs("gemini_bb-push-green", brief("brief-busybox-singles.md"),
    eps_for("armnet/busybox_push_green_button"), "sheets-rc2", "bb-push-green")
all_runs["gemini_bb-flip-left"] = make_runs("gemini_bb-flip-left", brief("brief-busybox-singles.md"),
    eps_for("armnet/busybox_flip_left_switch_off"), "sheets-rc2", "bb-flip-left")

for k, v in all_runs.items():
    print(k, len(v), "runs")
    with open(os.path.join(BASE, "gemini-staging", f"runs-{k}.txt"), "w") as f:
        f.write("\n".join(v) + "\n")
