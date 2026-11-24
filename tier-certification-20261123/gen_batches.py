#!/usr/bin/env python3
"""Generate judging batches JSON + Grok prompts for a new source.

Usage:
  gen_batches.py --episodes-csv <prefix>_episodes.csv \
      --sheet-prefix <prefix> --batch-prefix <bp> \
      --group-by-task <task1,task2,...> (optional; default: sequential) \
      --task-names <name1,name2,...> (parallel to groups when grouping) \
      --brief <brief.md> --out-batches batches/<file>.json \
      --grok-prefix grok_<bp> --batch-size 25

Batch JSON format matches batches/rc_batches.json:
  [{"batch_id": ..., "dataset": <group name>, "episodes":
    [{"episode_index": n, "task_string": s, "sheet": "sheets/<prefix>__ep<n>.jpg"}]}]

Grok prompts are written to grok-staging/prompts/<grok_prefix>-NN.txt in
the established format (brief text + episode list) and the run names are
appended to grok-staging/runs-<bp>.txt.
"""
import argparse, csv, json, os

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-csv", required=True)
    ap.add_argument("--sheet-prefix", required=True)
    ap.add_argument("--batch-prefix", required=True)
    ap.add_argument("--group-tasks", default=None, help="comma list of task_strings to group by, in order")
    ap.add_argument("--group-names", default=None, help="comma list of group names parallel to --group-tasks")
    ap.add_argument("--brief", required=True)
    ap.add_argument("--out-batches", required=True)
    ap.add_argument("--grok-prefix", required=True)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--source-label", default=None, help="label used in the grok prompt header line")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.episodes_csv)))
    for r in rows:
        r["episode_index"] = int(r["episode_index"])
    rows.sort(key=lambda r: r["episode_index"])

    groups = []  # (group_name, [rows])
    if a.group_tasks:
        tasks = a.group_tasks.split(",")
        names = a.group_names.split(",") if a.group_names else tasks
        for tstr, name in zip(tasks, names):
            groups.append((name, [r for r in rows if r["task_string"] == tstr]))
        grouped = {id(r) for _, rs in groups for r in rs}
        rest = [r for r in rows if id(r) not in grouped]
        if rest:
            groups.append(("other", rest))
    else:
        groups.append((a.batch_prefix, rows))

    brief = open(a.brief).read().rstrip()
    batches = []
    run_names = []
    for gname, grows in groups:
        for i in range(0, len(grows), a.batch_size):
            chunk = grows[i:i + a.batch_size]
            n = i // a.batch_size + 1
            bid = f"{a.batch_prefix}-{gname}-{n:02d}" if a.group_tasks else f"{a.batch_prefix}-{n:02d}"
            eps = [{"episode_index": r["episode_index"], "task_string": r["task_string"],
                    "sheet": f"sheets/{a.sheet_prefix}__ep{r['episode_index']}.jpg"} for r in chunk]
            batches.append({"batch_id": bid, "dataset": gname, "episodes": eps})
            # grok prompt
            run = f"{a.grok_prefix}-{gname}-{n:02d}" if a.group_tasks else f"{a.grok_prefix}-{n:02d}"
            label = a.source_label or a.batch_prefix
            lines = [brief, "", "",
                     f"Judge exactly these {len(chunk)} episodes of {label} (group: {gname}). "
                     f"For each one, open the contact-sheet image with your image/file reading tool "
                     f"and judge from the sheet only, against that episode's own task string. "
                     f"Working directory: /home/openclaw/muse-work/zk0-tier10h-cert-20261123", ""]
            for r in chunk:
                lines.append(f"- ep{r['episode_index']} — sheets/{a.sheet_prefix}__ep{r['episode_index']}.jpg — task: \"{r['task_string']}\"")
            lines += ["", f"Output ONLY the {len(chunk)} verdict lines in the format specified above, one per episode, in the order listed. No preamble, no summary, no other text."]
            pdir = os.path.join(BASE, "grok-staging", "prompts")
            os.makedirs(pdir, exist_ok=True)
            with open(os.path.join(pdir, run + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            run_names.append(run)

    with open(a.out_batches, "w") as f:
        json.dump(batches, f, indent=1)
    with open(os.path.join(BASE, "grok-staging", f"runs-{a.batch_prefix}.txt"), "w") as f:
        f.write("\n".join(run_names) + "\n")
    print(json.dumps({"batches": len(batches), "episodes": len(rows), "runs": run_names}))

if __name__ == "__main__":
    main()
