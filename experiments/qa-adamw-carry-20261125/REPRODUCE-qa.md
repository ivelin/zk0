# REPRODUCE — Experiment Q-A (AdamW-carry FL rerun on certified data)

A guest with only public assets (this branch + the Hugging Face repos named
below) can reproduce the Q-A training run and its eval end to end. The
reproduction instructions and the guest-facing scripts in `eval/` use only
public assets and parameterized paths. Preserved execution records
(`execution/`, `RUN_LOG.md`, the gate report, the payload hash list) retain
the build machine's paths as as-run evidence — they are records, not
instructions.

## What Q-A is

Study 001's federated arm scored 26/100 vs the centralized arm's 70/100
(judged completion, offline replay) on data whose completion axis had never
been certified. Q-A reruns the comparison on the fully certified run set with
**AdamW optimizer state carried across FL rounds**:

- **Federated (Q-A):** 5 clients (one per task), Flower, 30 server rounds,
  expert-only fine-tune, AdamW state carried across rounds.
- **Centralized pooled:** one model on all 239 episodes, expert-only, 15,000 steps.
- **Five specialists:** one per task, full fine-tune, 3,000 steps each.

All three arms start from the same base (`lerobot/smolvla_base`) and the same
pooled preprocessor (D-9 union statistics).

**Two variables differ from Study 001's FL arm — AdamW carry AND the dataset
(250 → 239 certified episodes). The headline must never claim carry alone was
isolated.**

**Registered deviation (DECISIONS #11):** the run was interrupted and resumed
from the round-11 checkpoint. On the resume, the re-fit of round 12 gave the
optimizer a one-round AdamW moment lead; the post-round-12 states were kept
as-is and the chain completed rounds 12–30 from them. Optimizer step counters
were verified advancing (1200 → 1220 across the resume smoke) — the carry
itself loaded and continued.

## Assets (all public)

| Asset | Location |
|---|---|
| Base dataset | `zk0bot/armnetbench_v01_lerobot_so101` @ `89eef614` |
| Certified manifest (329 rows: 239 train + 90 held-out) | `zk0bot/certified-study001-five-tasks` |
| Q-A arm checkpoints (eval form) | `zk0bot/smolvla-qa-20261127` (see `checkpoints_manifest.json` in this folder for sha256 per arm) |
| Run set (239) | `run_set_manifest.json` + `episode_lists/` in this folder |
| Eval split (90) + canary (40) | `eval/split.json`, `eval/canary.json` in this folder |

Held-out denominator is **90** (20 per task × 4 + cable 10 — the founder's
2026-11-23 re-split replaced the original 20 held-out cable episodes, which
had all failed certification, with 10 certified ones; the eval gate carries
this as registered waiver `W-cable-resplit-counts`).

## Environment

Python 3.12 venvs; exact freezes in `env_lock/`:

- `env_lock/requirements-fl-train-venv.txt` — FL substrate + lerobot training.
- `env_lock/requirements-eval-venv.txt` — eval harness (trajectory generation + judging).

One GPU with ≥24 GB is enough; the FL clients train **serially** on one
machine (one aggregator, clients in sequence — never simultaneously).

## Train

1. Build the run set from the certified manifest (rows pinned in
   `run_set_manifest.json`) and the pooled preprocessor
   (`study_prep/build_pooled_preprocessor.py` on the branch).
2. FL: the substrate entry points are on this branch (`src/` +
   `study_prep/run_train.py` drivers; the executed chain driver is preserved
   at `execution/qa_chain_driver3.sh` — an as-run record carrying the build
   machine's paths — with `execution/chain_done.json` the artifact-gated
   completion record: rounds 1–30 gapless, 5/5 clients per round, 500-tensor
   census = 450,046,176 elements).
3. Centralized: pooled 239, expert-only, 15,000 steps from the shared base.
4. Specialists: per task, full fine-tune, 3,000 steps each.

## Eval (the Study 001 pipeline, unchanged)

1. **Package** the FL round checkpoints (`study_prep/package_final_checkpoint.py`)
   — pooled preprocessor/postprocessor + tokenizer are copied in and verified
   against the registered statistics hash. **A5 note:** the substrate saves
   FL checkpoints with a dataset-space `config.json` (front/top/wrist at
   native resolutions). The packaged eval copies carry the A5 metadata
   correction — `input_features` rewritten to the registered declaration
   (camera1/2/3 [3,256,256] + state [6]), original preserved beside it as
   `config.defective-orig.json` with a `repair_note_a5.json`. Weights are
   byte-identical (sha256 in the manifest). This is the same correction
   Study 001 applied (its `a5repair` record).
2. **Gate** all 9 checkpoints together:
   `study_prep/eval_gate_check.py --manifest eval/qa_eval_manifest_90.json
   --split eval/split.json --waive eval.split.per_task_counts=W-cable-resplit-counts`
   → the Q-A run's report (`eval/eval_gate_report.json`): **PASS** (43 PASS,
   1 DIFF-WAIVED, 0 DIFF-FAIL). Launches under the guard then need
   `STUDY_GATE_ALLOW_OVERRIDES=1` because the PASS carries the registered
   waiver — the same deliberate, logged pattern as the training launch.
3. **Generate** open-loop trajectories for all 90 held-out episodes × 9
   checkpoints (`eval/eval_traj.py` in this folder; download the dataset
   snapshot `zk0bot/armnetbench_v01_lerobot_so101` @ `89eef614` and point
   `QA_DATASET_ROOT` at it; deterministic per-episode/chunk noise seeds).
   `eval/generation_driver.sh` is the as-run driver with its paths
   parameterized the same way.
4. **Judge** with the registered rubric (`eval/JUDGE_BRIEF.md`): three gates
   per packet (end-state cluster, gripper band, within-chunk jump pathology),
   borderline packets decided by trajectory-sheet review, **two independent
   judge legs (different model families), unanimity certifies**.

## Results

Metric: **JUDGED completion (offline replay)** — a proxy for task completion,
not physical completion. Full per-packet verdicts: `eval/judgments_index_qa.json`;
table: `eval/completion_table_qa.md` (+ `.csv`).

| checkpoint | scope | certified | single-leg completed | disputes | n |
|---|---|---|---|---|---|
| centralized_qa (all 90) | | 0 | 71 | 0 | 90 |
| centralized_qa | block_stack | 0 | 16 | 0 | 20 |
| centralized_qa | eye_drops_to_basket | 0 | 15 | 0 | 20 |
| centralized_qa | ring_insert | 0 | 19 | 0 | 20 |
| centralized_qa | tool_insert | 0 | 12 | 0 | 20 |
| centralized_qa | cable_clip | 0 | 9 | 0 | 10 |
| fl_qa_r30 (all 90) | | 0 | 27 | 0 | 90 |
| fl_qa_r30 | block_stack | 0 | 3 | 0 | 20 |
| fl_qa_r30 | eye_drops_to_basket | 0 | 7 | 0 | 20 |
| fl_qa_r30 | ring_insert | 0 | 10 | 0 | 20 |
| fl_qa_r30 | tool_insert | 0 | 7 | 0 | 20 |
| fl_qa_r30 | cable_clip | 0 | 0 | 0 | 10 |
| fl_qa_r20 (all 90) | | 0 | 15 | 0 | 90 |
| fl_qa_r20 | block_stack | 0 | 2 | 0 | 20 |
| fl_qa_r20 | eye_drops_to_basket | 0 | 3 | 0 | 20 |
| fl_qa_r20 | ring_insert | 0 | 5 | 0 | 20 |
| fl_qa_r20 | tool_insert | 0 | 3 | 0 | 20 |
| fl_qa_r20 | cable_clip | 0 | 2 | 0 | 10 |
| fl_qa_r10 (all 90) | | 0 | 6 | 0 | 90 |
| fl_qa_r10 | block_stack | 0 | 0 | 0 | 20 |
| fl_qa_r10 | eye_drops_to_basket | 0 | 3 | 0 | 20 |
| fl_qa_r10 | ring_insert | 0 | 1 | 0 | 20 |
| fl_qa_r10 | tool_insert | 0 | 2 | 0 | 20 |
| fl_qa_r10 | cable_clip | 0 | 0 | 0 | 10 |
| specialists (diagonal pooled) | | 0 | 71 | 0 | 90 |
| spec_block_stack | diagonal (block_stack) | 0 | 13 | 0 | 20 |
| spec_eye_drops_to_basket | diagonal (eye_drops_to_basket) | 0 | 18 | 0 | 20 |
| spec_ring_insert | diagonal (ring_insert) | 0 | 18 | 0 | 20 |
| spec_tool_insert | diagonal (tool_insert) | 0 | 18 | 0 | 20 |
| spec_cable_clip | diagonal (cable_clip) | 0 | 4 | 0 | 10 |

**Reading the table:** counts are the Muse judge leg only (SINGLE-LEG, labeled, never final) — the Gemini second leg is pending and certification (two-judge unanimity) completes when it lands. For fl_qa_r10/r20 only the fixed 40-packet canary subset was judged (r10: 6/40, r20: 15/40, r30 canary: 12/40); their remaining 50 packets per arm are UNJUDGED in the index. Headline arms (centralized_qa, fl_qa_r30) and the specialist diagonal were judged on the full 90.

## Verification status

- Training completion is artifact-gated (chain_done.json + per-round
  aggregation records + checkpoint census), never exit-code based.
- Every eval checkpoint passed `checkpoint_verify.sh` (full state + reload;
  the script is `eval/checkpoint_verify.sh` in this folder) before the eval gate.
- An independent outside-context reproduction pass over this package is
  recorded in `VERIFICATION.md` in this folder when it completes.
