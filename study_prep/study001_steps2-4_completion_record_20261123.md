# Study 001 — Founder Sequence Steps 2–4 Completion Record (2026-11-23)

Founder sequence ordered 2026-11-22 13:03 CT; Steps 2–4 released ~13:29 CT
after the pre-flight gate fixes (F1/F2/F3) were independently re-audited
PASS-WITH-FINDINGS (re-audit §5 release path). Step 1 (FL repair + 3-camera
re-eval) has its own record:
`study001-fl-reeval-3cam-results-20261122.md`. All judging is **JUDGED
completion (offline replay)** under the registered rubric incl. Amendment
A7 — not physical completion, not sim rollout success. Numbers below are
internally verified (recomputed from the FINAL judgment files) but are NOT
independently certified; independent verification is a separate step before
the founder is quoted them as certified.

DGX workdir: `W=/home/openclaw/muse-work/zk0-study001-20261119`
Eval dir: `E=$W/run/eval-traj-20261122`

## Step 2 — Sequential chain resumed and completed

- Chain was SIGSTOPped inside its existing guard at seq_ring_insert step
  2319/3000 (Step 0 pause, 2026-11-22T19:10:12Z; fallback checkpoints 001875
  and 002250 verified RESUME READY).
- Resumed 2026-11-22 ~19:29Z via SIGCONT to process group 2685423 inside the
  existing guard (no wrapper invocation; re-audit confirmed this path is not
  gate-blocked). Continuity verified live: step 2382/3000 observed, past the
  2319 pause point, ~1.34 s/step.
- Blocks completed in order: ring_insert (~19:50Z), tool_insert (~21:27Z),
  cable_clip. **"SEQ CHAIN COMPLETE" logged 2026-11-22T23:17:57Z**; all five
  `seq_*.done` markers present in `$W/run/train/`.
- Sequential final checkpoint:
  `$W/run/train/seq_cable_clip/out/checkpoints/003000/pretrained_model`
  (config + model.safetensors + pre/post processors + tokenizer present;
  config declares state[6] + camera1/2/3 [3,256,256]).

## Step 3 — Expert-only specialists (spec_exp) completed

- The spec_exp waiter (guard PID 1954470 / driver PID 1954617, armed before
  the pause) fired on its own at 2026-11-22T23:17:59Z on the anchored
  chain-completion line — no intervention needed.
- 8-step smoke PASSED (train + save + fresh-process 500-tensor reload
  assert). Five expert-only blocks then ran serially from the pinned base,
  3,000 steps each, independently (no chaining): block_stack done ~01:00Z,
  eye_drops ~02:41Z, ring_insert ~04:26Z, tool_insert ~06:07Z, cable_clip
  last. **"SPEC EXP CHAIN COMPLETE" 2026-11-23T07:50:57Z**; all five
  `spec_exp_<task>.done` markers + `spec_exp_chain.done` present; waiter
  exited; GPU idle afterwards.
- Checkpoints: `$W/run/train/spec_exp_<task>/out/checkpoints/003000/pretrained_model`
  — all five declare state[6] + camera1/2/3 [3,256,256] in config.json.

## Step 4 — Re-eval and table update

### R-2 resolution (superseded two-view records)
- The original two-view fl_r30 records (100 `.npz` + `summary__fl_r30.json`)
  and its 100 judgment JSONs were MOVED (preserved, not deleted) to
  `$E/superseded_20261122/fl_r30_2view/{records,judgments}/` with a dated
  README stating the supersession reason (unequal eval inputs, defects
  D1/D9; replaced by fl_r30_3cam, 26/100).

### Eval gate
- `study_prep/eval_gate_check.py` (in the FL clone
  `/home/openclaw/muse-work/zk0-study001-fl-20261121`) over the full
  14-checkpoint set (base, spec×5 full-FT, cexp, fl_r30_3cam, sequential,
  spec_exp×5) against the frozen split: **GATE VERDICT: PASS (49 PASS,
  0 DIFF-WAIVED, 0 DIFF-FAIL)**, exit 0. Report: `$E/eval_gate_report.json`.
  Split checks: 100 episodes, seed 20261122, canonical sha256
  `24b0b99e3053…`, disjoint from the 250 training indices.

### Trajectory evals
- Driver `$E/step4_eval_driver.sh` (log `$E/step4_eval_driver.log`, per-run
  log `$E/step4_eval.log`) ran six sequential `launch_dgx_job.sh`
  invocations (guard dirs `$E/guard-step4-<name>`, mirror target
  `$E/mirror`); the wrapper printed "study EVAL gate PASS verified" for each
  and every `eval_traj.py` run exited 0 (~2.2 min each, 08:07–08:21Z
  2026-11-23). Records dir now holds 14 checkpoints × 100 `.npz` + 14
  summaries. Only the gated canonical harness was used (residual R-1: the
  ungated repair3cam script copies were NOT reused).

### Judging
- Scoped Step-4 scripts were built FRESH from the current gated canonical
  scripts (`$E/scripts/step4/{judge_packets_step4,review_sheets_step4,finalize_step4}.py`);
  the diff vs canonical `judge_packets.py` is exactly 2 lines (docstring +
  one scope-filter line); the gate block and all rubric code are
  byte-identical. Packets: 600 provisional, 113 borderline.
- Visual review of all 113 borderline packets (19 sheets,
  `$E/judgments/review_step4/`), blind to other checkpoints' verdicts on the
  same episodes, under Amendment A7 sub-rules R1/R2/R4 (R5 not used):
  **21 flips False→True (R1 ×6, R2 ×6, R4 ×9); 9 provisional-completed
  confirmed; 83 confirmed not completed; no provisional completed was
  overturned.** Decisions: `$E/judgments/review_step4/decisions.csv`.
  Finalize output: "finalized 600 step4 packets; borderline=113; flips=21".
- INCIDENT (same class as Step 1's): the scoped review-sheets script
  inherited the canonical script's hardcoded write path and briefly
  overwrote the canonical `judgments/review/borderline_list.json` (142
  entries) with the Step-4 list (113). The canonical list was regenerated
  from the FINAL judgment files' borderline flags (original 8 checkpoints,
  including the archived fl_r30 judgments) and verified at exactly 142
  entries; the scoped script's path was patched to `review_step4/` so the
  defect cannot recur. No judgment data was affected at any point.

### Table update
- `scripts/build_table.py` ORDER updated: `fl_r30` replaced by
  `fl_r30_3cam`; `sequential` and the five `spec_exp_*` rows added. Notes
  section rewritten: two-view fl_r30 row marked SUPERSEDED with a pointer to
  the archive; corrected FL row 26/100 with Φ = −44 pp and the ep859
  strict-reading bound Φ = −45 pp; Step 4 borderline/flip disclosure added;
  the original 142-packet/33-flip disclosure retained.
- Canonical tables: `$E/results/completion_table.md` / `.csv`. Local mirrors
  (byte-identical, md5 `4181a8013c7bb4818407aaedf44e4bd9` /
  `68beab2cd7dba321b98ae8ba78863895`):
  `study001-eval-completion_table.md` / `.csv` beside this record.
- Internal verification: all 14 rows recomputed independently from the
  1,400 FINAL judgment JSONs (100 per checkpoint, all judge_status FINAL)
  match the published table exactly.

## Updated comparison table (JUDGED completion, offline replay)

| checkpoint | block_stack | eye_drops | ring_insert | tool_insert | cable_clip | pooled |
|---|---|---|---|---|---|---|
| base | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/100 (0%) |
| spec_block_stack (full-FT) | 13/20 | 6/20 | 2/20 | 5/20 | 0/20 | 26/100 (26%) |
| spec_eye_drops (full-FT) | 1/20 | 19/20 | 0/20 | 12/20 | 0/20 | 32/100 (32%) |
| spec_ring_insert (full-FT) | 0/20 | 4/20 | 19/20 | 4/20 | 0/20 | 27/100 (27%) |
| spec_tool_insert (full-FT) | 0/20 | 7/20 | 3/20 | 18/20 | 0/20 | 28/100 (28%) |
| spec_cable_clip (full-FT) | 0/20 | 6/20 | 0/20 | 8/20 | 0/20 | 14/100 (14%) |
| cexp (expert-only, centralized) | 17/20 | 18/20 | 19/20 | 16/20 | 0/20 | 70/100 (70%) |
| fl_r30_3cam (expert-only, federated, corrected) | 0/20 | 7/20 | 10/20 | 9/20 | 0/20 | 26/100 (26%) |
| sequential (chained specialists) | 1/20 | 9/20 | 10/20 | 12/20 | 0/20 | 32/100 (32%) |
| spec_exp_block_stack | 12/20 | 4/20 | 4/20 | 11/20 | 0/20 | 31/100 (31%) |
| spec_exp_eye_drops | 1/20 | 18/20 | 0/20 | 6/20 | 0/20 | 25/100 (25%) |
| spec_exp_ring_insert | 0/20 | 5/20 | 17/20 | 9/20 | 0/20 | 31/100 (31%) |
| spec_exp_tool_insert | 0/20 | 4/20 | 1/20 | 15/20 | 0/20 | 20/100 (20%) |
| spec_exp_cable_clip | 0/20 | 9/20 | 3/20 | 8/20 | 0/20 | 20/100 (20%) |

cable_clip is 0/20 for every checkpoint (finding D10: its held-out
references are failed executions). Mean trajectory error (pooled, excl.
cable_clip): cexp 18.98 · fl_r30_3cam 36.49 · sequential 47.21 ·
spec_exp rows 40.74–58.23 · full-FT specialists 40.69–51.15 · base 99.99.

## Where every record lives

- DGX eval dir: `/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/`
  — `records/` (1,400 `.npz` + 14 summaries), `judgments/` (1,400 FINAL
  JSONs), `judgments/review_step4/` (19 sheets, borderline_list.json,
  decisions.csv), `judgments/review/` (canonical 142-entry borderline list,
  restored), `superseded_20261122/fl_r30_2view/` (archived two-view records
  + judgments + README), `eval_gate_report.json`, `results/completion_table.{md,csv}`,
  `scripts/step4/`, `step4_eval_driver.{sh,log}`, `step4_eval.log`,
  `guard-step4-*/`, `mirror/`.
- Training records: `$W/run/train/` (sequence.log, spec_exp logs, `.done`
  markers, checkpoints).
- Local: this record + byte-identical table mirrors in
  `~/workspace/research_notes/zk0-bench003-design-20261119/`; Step-4 review
  working copies (sheets, borderline_features.json, decisions.csv) in its
  `step4-review/` subfolder.
