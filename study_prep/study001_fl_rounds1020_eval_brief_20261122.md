# Final eval batch — registration of FL rounds 10/20 (diagnostic)

Registered 2026-11-22 by the rounds-10/20 prep (founder approval 2026-11-22 15:40 CT).
Purpose: FL learning-curve diagnostic — did FL learn and get averaged away, or never
learn? Judge FL rounds 10 and 20 on the SAME frozen 100 held-out episodes, SAME rubric,
as every other arm, inside tonight's final eval batch (Step 4 of the founder sequence).
All completion figures are **JUDGED completion (offline replay)**.

## New checkpoints in the eval set

| Eval name | Path |
|---|---|
| `fl_r10_3cam` | `/home/openclaw/muse-work/zk0-study001-fl-20261121/outputs/2026-11-21_14-56-47/models/checkpoint_round_10_3cam` |
| `fl_r20_3cam` | `/home/openclaw/muse-work/zk0-study001-fl-20261121/outputs/2026-11-21_14-56-47/models/checkpoint_round_20_3cam` |

These are repaired copies of the as-saved round checkpoints (originals untouched):
D-9 pooled packaging + config `input_features` = state[6] + camera1/camera2/camera3
[3,256,256] (identical to C-exp's declared inputs). Weights byte-identical to the
as-saved rounds (SHA-256 r10 `aa2e02e0…1089`, r20 `6b472ba9…ac0a`); packaged processor
files byte-identical to the accepted `checkpoint_round_30_3cam`. Each copy carries a
`repair_note.json`. Prep record: `rounds1020/` in this dir (hashes, diffs, proofs,
package logs) and the prep report committed on branch `study001/fl-substrate-20261121`.

## Step A — re-run the eval gate over the FULL final set (mandatory)

The gate report at `run/eval-traj-20261122/eval_gate_report.json` currently covers the
10 checkpoints that exist now (verdict PASS, 37 checks, 2026-11-22 prep). Tonight's set
adds `sequential` and the five `spec_exp_*` checkpoints, whose config hashes are NOT in
that report — `eval_traj.py` and `judge_packets.py` will refuse them. Before any
trajectory pass, re-run:

```
cd /home/openclaw/muse-work/zk0-study001-fl-20261121/study_prep
W=/home/openclaw/muse-work/zk0-study001-20261119
M=/home/openclaw/muse-work/zk0-study001-fl-20261121/outputs/2026-11-21_14-56-47/models
../venv/bin/python eval_gate_check.py \
  --manifest fl_rerun_study_manifest.json \
  --split $W/run/eval-traj-20261122/split/split.json \
  --ckpt base=/home/openclaw/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --ckpt cexp=$W/run/train/centralized/out/checkpoints/015000/pretrained_model \
  --ckpt spec_block_stack=$W/run/train/spec_block_stack/out/checkpoints/003000/pretrained_model \
  --ckpt spec_eye_drops=$W/run/train/spec_eye_drops/out/checkpoints/003000/pretrained_model \
  --ckpt spec_ring_insert=$W/run/train/spec_ring_insert/out/checkpoints/003000/pretrained_model \
  --ckpt spec_tool_insert=$W/run/train/spec_tool_insert/out/checkpoints/003000/pretrained_model \
  --ckpt spec_cable_clip=$W/run/train/spec_cable_clip/out/checkpoints/003000/pretrained_model \
  --ckpt fl_r10_3cam=$M/checkpoint_round_10_3cam \
  --ckpt fl_r20_3cam=$M/checkpoint_round_20_3cam \
  --ckpt fl_r30_3cam=$M/checkpoint_round_30_3cam \
  --ckpt sequential=$W/run/train/seq_cable_clip/out/checkpoints/003000/pretrained_model \
  --ckpt spec_exp_block_stack=$W/run/train/spec_exp_block_stack/out/checkpoints/003000/pretrained_model \
  --ckpt spec_exp_eye_drops=$W/run/train/spec_exp_eye_drops/out/checkpoints/003000/pretrained_model \
  --ckpt spec_exp_ring_insert=$W/run/train/spec_exp_ring_insert/out/checkpoints/003000/pretrained_model \
  --ckpt spec_exp_tool_insert=$W/run/train/spec_exp_tool_insert/out/checkpoints/003000/pretrained_model \
  --ckpt spec_exp_cable_clip=$W/run/train/spec_exp_cable_clip/out/checkpoints/003000/pretrained_model \
  --out $W/run/eval-traj-20261122/eval_gate_report.json
```

(Adjust the spec_exp names/paths to the ones the spec_exp completion record reports, if
they differ.) Require `GATE VERDICT: PASS` before proceeding. Do NOT use
`EVAL_GATE_OVERRIDE`.

## Step B — trajectory passes (one checkpoint at a time, via the launcher)

Same pattern as the fl_r30_3cam pass (guard-wrapped, `launch_dgx_job.sh` study mode;
check fleet-lock state at run time — after the chain and spec_exp finish, the lock is
free and JOB_LOCK stays at its default). Bench venv python, frozen split, records out:

```
PY=/home/openclaw/muse-work/zk0-bench-20261117/venv/bin/python
E=/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122
$PY $E/scripts/eval_traj.py --ckpt-name fl_r10_3cam \
    --ckpt-path $M/checkpoint_round_10_3cam \
    --split $E/split/split.json --out $E/records
$PY $E/scripts/eval_traj.py --ckpt-name fl_r20_3cam \
    --ckpt-path $M/checkpoint_round_20_3cam \
    --split $E/split/split.json --out $E/records
```

Each must print `image_key_map` = front→camera1, top→camera2, wrist→camera3 and finish
with 100 records + `summary__<name>.json`. (Same commands for `sequential` and the five
`spec_exp_*` checkpoints with their own names/paths.)

## Step C — judging (scoped, do not touch existing FINAL judgments)

The stock `judge_packets.py` rebuilds packets for EVERY checkpoint in `records/` and
overwrites judgment files with provisionals. The 900 existing FINAL judgments (base,
spec×5, cexp, fl_r30_3cam) must not be re-provisionalled. Follow the repair3cam
precedent: scoped copies of `judge_packets.py` / `review_sheets.py` / `finalize.py`
glob-filtered to the NEW checkpoint names only (fl_r10_3cam, fl_r20_3cam, sequential,
spec_exp_*), rubric = eval-plan.md incl. Amendments A1/A6 gates and the A7-registered
borderline sub-rules (R1/R2/R4/R5 as documented), borderline packets visually reviewed,
decisions recorded in a decisions.csv beside the review sheets. Label every figure
"JUDGED completion (offline replay)".

## Step D — table surgery (build_table.py)

Current ORDER still names the superseded `fl_r30` and lacks the corrected/new rows.
Set ORDER to place the FL curve together, e.g.:
`[..., "cexp", "fl_r10_3cam", "fl_r20_3cam", "fl_r30_3cam", "sequential", "spec_exp_block_stack", ...]`
(choose the final layout in Step 4; the requirement is that fl_r10_3cam and fl_r20_3cam
rows appear next to fl_r30_3cam). Preserve the hand-appended Notes section of
`results/completion_table.md` — the script regenerates the table without it (see the
3-camera re-eval report). The superseded two-view records already live in
`superseded_20261122/fl_r30_2view/` — keep them out of `records/`.

## Step E — diagnostic readout

Read the three judged pooled scores (r10 / r20 / r30 = 26) against the loss curve
(`study001_fl_loss_curve_20261122.csv`, committed on the study branch and copied in
`rounds1020/`): median client policy loss falls monotonically 0.391 → 0.246 → 0.222
(rounds 1 → 20 → 30) with no mid-run peak in training loss. If judged completion is
flat (~25) at all three rounds, the story is "never learned to complete", not
"learned then averaged away"; if r10/r20 score materially above r30, averaging is
implicated. Report per-task rows alongside pooled.
