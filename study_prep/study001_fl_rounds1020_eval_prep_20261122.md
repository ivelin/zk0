# Study 001 — FL rounds 10/20 diagnostic: eval prep (2026-11-22)

Status: **PREP COMPLETE, staged for tonight's final eval batch.** Founder approval 2026-11-22 15:40 CT. All figures to come from this batch are **JUDGED completion (offline replay)**. Prep was file/CPU work only — the sequential chain (seq_tool_insert) and the spec_exp waiter were not disturbed (GPU 96% chain training throughout; no GPU job launched).

## 1. Checkpoints located; defect confirmed

As-saved FL round checkpoints live in `/home/openclaw/muse-work/zk0-study001-fl-20261121/outputs/2026-11-21_14-56-47/models/checkpoint_round_<N>` (rounds 1–30). Read directly this session: **rounds 10 and 20 carry the same defect as rounds 1/15/29** — `config.json` `input_features` declares only `observation.images.front` + `observation.images.top`, each [3,240,320], no wrist. They are also *unpackaged* raw server saves: `policy_preprocessor.json` has an empty rename map, and there is no normalizer state file and no `tokenizer/` dir (round 30 received the D-9 packaging at packaging time; rounds 1–29 never did).

## 2. Repaired copies (originals untouched)

| Eval name | Repaired copy |
|---|---|
| `fl_r10_3cam` | `…/models/checkpoint_round_10_3cam` |
| `fl_r20_3cam` | `…/models/checkpoint_round_20_3cam` |

Each copy = as-saved round checkpoint + the round-30 packaging (D-9 pooled `policy_preprocessor.json` + normalizer state, postprocessor unnormalizer state, `tokenizer/`, all from `study_assets/pooled_preprocessor`) + `config.json` `input_features` replaced with state[6] + camera1/camera2/camera3 [3,256,256] (byte-identical declared inputs to C-exp) + a `repair_note.json` inside the copy.

- **Weights byte-identical** (SHA-256 of `model.safetensors`, source before copy == copy after):
  - round 10: `aa2e02e05fdad14a5b1063c7f0a601dce7c5c7deeef0142924091f3f236c1089`
  - round 20: `6b472ba91b661bd915b6a9c5c368784030fe0f10f9e9afb79af168cb424dac0a`
- **Packaging equivalence:** the packaged files in both copies are md5-identical to the accepted `checkpoint_round_30_3cam`'s (preprocessor json `d65a5279…`, normalizer safetensors `630e86f1…`, postprocessor state `e2d28f43…`, postprocessor json `473056e3…`, tokenizer tree identical by `diff -rq`).
- **Diff vs as-saved originals** (per round, recorded in `rounds1020/r<N>_diff.txt`): `config.json` and `policy_preprocessor.json` differ; added: normalizer safetensors, postprocessor safetensors, `tokenizer/`, `repair_note.json`. Nothing else.
- One process note: `package_final_checkpoint.py` itself cannot run while the GPU is busy — it instantiates the preprocessor's `cuda` device step during its verification load and hard-fails with CUDA hidden. The file copies it performs were applied from the same pooled source instead (fallback path, logged in `rounds1020/r<N>_package.log`), and its stats verification is substituted by the md5 equivalence above plus the proofs below. See gap G5.

## 3. Proofs (CPU-only, harness device forced to CPU per the A5 precedent)

- **D9 assert: PASS** for both copies — repaired `input_features` == C-exp declared inputs, exact dict equality (`rounds1020/proofs1020.log`).
- **Standard-load test: PASS** for both copies — fresh process, naive `SmolVLAPolicy.from_pretrained` + packaged processors, dummy batch keyed front/top/wrist: want == got == {camera1, camera2, camera3}, `all_features_present=True` (`rounds1020/proofs1020c.log`; the test ran on a temp copy whose only change was the processor device step set to CPU in the harness — the artifacts of record keep `cuda`, correct for tonight's GPU eval; temp dirs deleted). Note: `proofs1020b.log` is empty — the first launch attempt ran an incompletely transferred script; `proofs1020c.log` is the valid record.

## 4. Eval gate verdict

`eval_gate_check.py` (manifest `fl_rerun_study_manifest.json`, frozen split `run/eval-traj-20261122/split/split.json`) over the enlarged 10-checkpoint set — base, cexp, spec×5 (full-FT), fl_r30_3cam, **fl_r10_3cam, fl_r20_3cam**:

**GATE VERDICT: PASS (37 PASS, 0 DIFF-WAIVED, 0 DIFF-FAIL), exit 0.** Report written to the canonical path `run/eval-traj-20261122/eval_gate_report.json` (the path `eval_traj.py` and `judge_packets.py` read). Frozen-split checks pass (100 episodes, seed 20261122, 20/task, disjoint from the 250 training episodes).

**Negative control:** the same gate run over the raw `checkpoint_round_10` FAILs it by name (`fl_r10_raw config.input_features` DIFF-FAIL, verdict FAIL) — the gate discriminates repaired from defective.

## 5. Registration into tonight's batch — staged, one trigger remains

There is **no automated mechanism that runs the final eval batch**: the seqchain monitor cron only reports chain completion; the spec_exp completion cron does verification/HF-backup/git/report for the specialists and never invokes the eval. So the batch fires when the parent dispatches Step 4 of the founder sequence (after the spec_exp completion report, ~02:00 CT). Everything that dispatch needs is staged:

- Repaired checkpoints (§2) at the paths the brief names.
- Gate report at the canonical path covering both new checkpoints (§4). The Step-4 executor must **re-run the gate over the full final set** (adding `sequential` + the five `spec_exp_*`) before judging — the exact command is Step A of the brief; the current report does not cover those six config hashes and the harness will refuse them until it does.
- **Ready-to-run brief:** `run/eval-traj-20261122/rounds1020/EVAL_BRIEF.md` on the DGX (also committed to the study branch, §7) — gate re-run command, trajectory-pass commands (same launcher pattern as the fl_r30_3cam pass), scoped judging per the repair3cam precedent (the 900 existing FINAL judgments stay untouched), `build_table.py` ORDER surgery (add `fl_r10_3cam`, `fl_r20_3cam` beside `fl_r30_3cam`; preserve the hand-appended Notes), and the readout step. Stale two-view records are already out of `records/` (`superseded_20261122/fl_r30_2view/`), so re-audit residual R-2 does not block judging.

**Parent trigger:** dispatch Step 4 with the brief above once the spec_exp completion report lands. Nothing else is owed for rounds 10/20.

## 6. Loss curve (diagnostic companion)

File: `study001_fl_loss_curve_20261122.csv` — local notes folder, DGX `rounds1020/`, and committed to the study branch (§7). Sources: per-round server aggregation records (`outputs/…/server/aggregation_round_<N>.json`: median + per-client policy loss under the D-11 dynamic weighting) and each round checkpoint's `metrics.json` (`avg_client_loss`). Client→task: client_id 0=block_stack, 1=eye_drops_to_basket, 2=ring_insert, 3=tool_insert, 4=cable_clip (reconstructed from client construction lines in `clients/…/client.log` — "Client N: P0 episode restriction from subsets/<task>" — see gap G4).

Reading: median client policy loss falls monotonically after a round-2 bump (0.391 → 0.474 → … → **0.322 @ r10 → 0.246 @ r20 → 0.222 @ r30**); avg client loss likewise (0.455 → 0.420 → 0.269 → 0.261). No mid-run peak or late collapse in the training losses — so if tonight's judged scores for r10/r20 come back near r30's 26/100, the loss curve and the judging agree that FL kept "improving" on its training objective without learning to complete; if r10/r20 judge materially higher, aggregation is implicated despite the smooth curve. One per-client divergence worth watching: the eye_drops client's policy loss *rises* from round 21 on (0.330 → 0.422) while all others fall or hold.

Caution recorded for any reader of the raw artifacts: the `server_composite_eval_loss` inside every round checkpoint's `metrics.json` is the **round-0** value (0.5773) repeated verbatim — per-round server evals were off (D-11), so that field is not a per-round curve. See gap G1.

## 7. Committed record (founder addendum 2026-11-22 15:43 CT)

Committed and pushed to branch `study001/fl-substrate-20261121` (DGX clone `study_prep/`): this prep report, the loss-curve CSV, the eval-brief copy, and the eval gate report copy (`study001_eval_gate_report_rounds1020_20261122.json`). The repaired checkpoint copies themselves stay on the DGX (checkpoints never go in git); their hashes and paths are recorded here and in their `repair_note.json` files. Commit hash in the delivery message.

## 8. Gaps found — candidate regression tests (not silently worked around)

- **G1 — stale eval field in per-round artifacts.** Every FL round checkpoint's `metrics.json` carries the round-0 `server_composite_eval_loss` (0.577289…) as if it were that round's measurement. Candidate test: per-round artifacts must not carry an eval field that was not computed in that round (assert the field is absent, labeled with its round, or varies with the round's own eval record).
- **G2 — no canonical eval gate report existed before this prep.** The gate checker existed and the re-audit had obtained a PASS in /tmp, but `run/eval-traj-20261122/eval_gate_report.json` — the artifact the harness actually reads — was never produced as part of eval prep. Candidate test: a batch pre-flight that asserts the canonical report exists, is PASS, and covers every checkpoint name scheduled in the batch, run before the first trajectory pass is launched.
- **G3 — eval gate's preprocessor check is existence-only.** A raw round checkpoint *has* a `policy_preprocessor.json` (empty rename map, no normalizer state, no tokenizer) and passes that check; only its config defect made the gate fail it. A config-correct but unpackaged checkpoint would pass the gate and fail at load time. Candidate test: eval-set admission includes a standard-load proof per checkpoint (from_pretrained + packaged processors; all declared image features present after preprocessing) — the proofs pattern used in §3 — as a gate check.
- **G4 — client→task assignment not recorded in run artifacts.** The FL aggregation records key losses by client_id only; which client held which task exists only in client construction log lines. Candidate: the run manifest must record the client→task table, and a test asserts every aggregation record's client set matches it.
- **G5 — packaging verification cannot run on a busy box.** `package_final_checkpoint.py` hard-fails when CUDA is unavailable because processor construction instantiates the `cuda` device step; on a training box there is no supported way to run its verification half. Candidate: a `--cpu-verify` mode (or device override) for the packaging script, with a test that packaging verification passes with CUDA hidden.

## Audit trail (DGX)

`run/eval-traj-20261122/rounds1020/`: `EVAL_BRIEF.md`, `study001_fl_loss_curve_20261122.csv`, `r10/r20_weights_sha.txt`, `r10/r20_diff.txt`, `r10/r20_package.log`, `proofs1020.log` (D9), `proofs1020c.log` (load tests). Gate report: `run/eval-traj-20261122/eval_gate_report.json`. Repaired copies: §2 paths, each with `repair_note.json`.
