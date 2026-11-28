# VERIFICATION — Q-A Phase 1 reproducibility package

Independent verification record for `experiments/qa-adamw-carry-20261125/`
(branch `experiment/qa-adamw-carry-20261125`). Three rounds, each by a
fresh-context verifier who did not build the package, against the guest
standard: reproducible from the branch + public Hugging Face alone.

- **Round 1** @ commit `3c24795` — verdict **NOT CERTIFIED** (report below).
- Builder fix @ commit `a8f0416` — eval_traj.py + checkpoint_verify.sh
  published; guest-facing script paths parameterized; reference inputs
  shipped in-package; REPRODUCE-qa.md public-assets claim scoped to the
  instructions (preserved records stay verbatim as evidence).
- **Round 2 (delta)** @ commit `a8f0416` — verdict **CERTIFIED WITH
  FINDINGS** (report below). All six round-1 defects resolved; two new
  path-layout defects found in the helper scripts (N1, N2).
- Builder fix @ commit `5e76769` — N1: build_qa_split.py reads
  run_set_manifest.json from the package root (was: inside episode_lists/)
  and its split.json write is live code again (was: swallowed by a comment);
  N2: judge_packets_qa.py reads eval/split.json (package layout). Functional
  smoke by the builder: the fixed build_qa_split.py, run against the dataset
  with QA_EVAL_TREE pointed at a scratch dir, regenerated split.json and
  canary.json **identical** to the shipped files (canonical rows sha256
  `c09a39a0914e3290…` reproduced) and qa_eval_manifest_90.json identical
  except the note wording the fix itself updated (shipped file replaced by
  the regenerated one so script and artifact agree).
- **Round 3 (micro-check of N1/N2)** @ commit `5e76769` — appended below.

---

## Round 1 report (commit 3c24795)

# Q-A Phase 1 — Independent Verification (guest standard)

**Overall verdict: NOT CERTIFIED**

Reason in one line: every quantitative claim, checkpoint hash, and disclosure checks out, but a guest cannot reproduce the eval end-to-end from public assets alone — the trajectory-generation script `eval_traj.py` referenced in REPRODUCE-qa.md is not on the branch, and several guest-facing scripts hard-code private `/home/openclaw/...` paths. That fails the stated guest standard ("can reproduce the Q-A training run and its eval end to end. Nothing here points at a private file.").

Verifier context: independent, fresh context; builder's assertions were not trusted — each was re-derived from the artifacts below.
Sources checked 2026-11-28: anonymous `git clone --depth 1 --branch experiment/qa-adamw-carry-20261125 https://github.com/ivelin/zk0.git` succeeded; HEAD = `3c247950ebd909d905b882f558eed6b14a3ebf16` (matches the stated commit 3c24795). HF checked via `HfApi.repo_info(..., files_metadata=True)` on `zk0bot/smolvla-qa-20261127`.

---

## Check 1 — REPRODUCE-qa.md completeness: **FAIL (partial)**

Present (verified on the branch, at branch root where REPRODUCE-qa.md says "on the branch"):
- `study_prep/build_pooled_preprocessor.py` ✓, `study_prep/package_final_checkpoint.py` ✓, `study_prep/eval_gate_check.py` ✓, `study_prep/run_train.py` ✓, `src/` ✓ (client/common/configs/server/training)
- In the package folder: `run_set_manifest.json` ✓, `episode_lists/` (5 files) ✓, `eval/*` ✓ (split.json, canary.json, qa_eval_manifest_90.json, eval_gate_report.json, JUDGE_BRIEF.md, judgments_index_qa.json, completion_table_qa.md/.csv, build_qa_split.py, judge_packets_qa.py, review_sheets_qa.py, generation_driver.sh), `env_lock/*` ✓, `execution/*` ✓ (qa_chain_driver3.sh, chain_done.json, chain_status.json), `checkpoints_manifest.json` ✓

Missing references:
- **`eval_traj.py` — MISSING everywhere on the branch** (`find` over the whole clone: no match). REPRODUCE-qa.md step Eval-3 says "Generate open-loop trajectories for all 90 held-out episodes × 9 checkpoints (`eval_traj.py`; …)". The only pointer to it is a private absolute path in `eval/generation_driver.sh:10`: `TRAJ=/home/openclaw/muse-work/zk0-study001-20261119/run/eval-traj-20261122/scripts/eval_traj.py` — not a public asset. A guest cannot run the generation step.
- **`checkpoint_verify.sh` — MISSING on the branch** (no match in clone). REPRODUCE-qa.md "Verification status" claims "Every eval checkpoint passed `checkpoint_verify.sh`"; a guest cannot re-run that check from public assets.
- **`VERIFICATION.md` — absent** from the package folder. REPRODUCE-qa.md phrases it conditionally ("recorded in `VERIFICATION.md` in this folder when it completes"), so this is an open item rather than a false claim, but at verification time it does not exist.

## Check 2 — checkpoints_manifest.json vs Hugging Face: **PASS**

- Repo `zk0bot/smolvla-qa-20261127`: **public** (`private=False`), head sha `a871654ff03dd28855863143ae388274cd0c35af` — matches the manifest's `verified_head_sha`. Card data license = **apache-2.0**; README present and its front-matter says `license: apache-2.0`.
- All **9/9 arms MATCH** — HF LFS sha256 of `<arm>/model.safetensors` vs `checkpoints_manifest.json`:

| Arm | sha256 (manifest = HF) |
|---|---|
| centralized | a1933bd27bf389bcc16a9ab2a4d71838c3efb55bc00a66d0ab7b67bf0d302e96 |
| fl_r10 | a97f06caa97c16e5e126a4bd5255eaa9c1a0e9d0555860bc3485ddb3ea95d262 |
| fl_r20 | edc879a6c152dd1ff11ac298ce20e954b1b8def5a8f596013d58242b058ca0f2 |
| fl_r30 | ac36f73fc29d4d0bff472229be42f01fa68f326e4f84f51431d5e16767315d38 |
| specialists/block_stack | 7638a8506a39643dde2b2df845ef50b6fb09832b96b6baf77057992b9d37b566 |
| specialists/eye_drops_to_basket | c4d930a447cc821fd9ea959e26d58eb3322fa871a6fbf0cc6e8338458a1eb3d9 |
| specialists/ring_insert | a5c2aa1040a7c5bf0966a8b9a6b8bf575dabecded206f3a7065cc2bb75369d39 |
| specialists/tool_insert | 4cc05ebd362e3fbb85c564ab7b74cde8506382f1b19349fcb12e5f02c1b0ddcc |
| specialists/cable_clip | cddc79f8a4829b13ba8e60feb7eada9fd7741f12d8bf0bb21e56141bf75ffc94 |

- Supporting public assets also verified public: `zk0bot/armnetbench_v01_lerobot_so101` (dataset, sha 89eef614… — matches the pinned revision in REPRODUCE-qa.md), `zk0bot/certified-study001-five-tasks` (dataset, public), `lerobot/smolvla_base` (model, public).

## Check 3 — Internal consistency: **PASS**

(a) `eval/split.json`: **90 episodes** total; per-task **20 / 20 / 20 / 20 / 10** (block_stack / eye_drops_to_basket / ring_insert / tool_insert / cable_clip), all unique. Training set re-derived from `episode_lists/*.json`: 49 + 50 + 50 + 50 + 40 = **239** (matches `run_set_manifest.json` total=239 and its per-task counts). **Eval ∩ training = ∅** (empty intersection, computed set-wise). **90 + 239 = 329** ✓.
(b) `eval/canary.json`: **40 packets** (8 per task × 5 tasks), **all 40 inside the split** (0 outside, checked per task).
(c) `eval/eval_gate_report.json`: verdict **PASS**; 44 checks = **43 PASS + 1 DIFF-WAIVED + 0 FAIL**. The single waived check is `eval.split.per_task_counts` with detail `[waiver W-cable-resplit-counts]` ✓.
(d) `execution/chain_done.json`: FL gate records rounds **1–30, gapless**, `aggregation_per_client_counts` = **5 for every round 1–30**, `aggregation_5_of_5_no_gaps: true`, census **500 tensors / 450,046,176 elements** ✓, and it itself records the DECISIONS #11 deviation.
(e) `eval/completion_table_qa.md` vs independent recount of `eval/judgments_index_qa.json` (450 rows; counting the Muse leg, `gemini=null` throughout):
- centralized_qa: **71/90** ✓ (per-task 16/15/19/12/9 — all match the table)
- fl_qa_r30: **27/90** ✓ (per-task 3/7/10/7/0 — match)
- specialists diagonal pooled: **71/90** ✓ (13 + 18 + 18 + 18 + 4)
- fl_qa_r10 / fl_qa_r20: exactly **40 SINGLE-LEG + 50 UNJUDGED** rows each — canary-only judging confirmed; canary recounts r10 **6/40**, r20 **15/40**, r30 canary **12/40**, all matching the table's reading note ✓.

## Check 4 — Disclosure integrity: **PASS** (all four, in both documents)

REPRODUCE-qa.md:
- (i) AdamW deviation: "**Registered deviation (DECISIONS #11):** the run was interrupted and resumed from the round-11 checkpoint. On the resume, the re-fit of round 12 gave the optimizer a one-round AdamW moment lead; the post-round-12 states were kept as-is…"
- (ii) Two variables: "**Two variables differ from Study 001's FL arm — AdamW carry AND the dataset (250 → 239 certified episodes). The headline must never claim carry alone was isolated.**"
- (iii) Proxy metric: "Metric: **JUDGED completion (offline replay)** — a proxy for task completion, not physical completion."
- (iv) Single-leg: "**Reading the table:** counts are the Muse judge leg only (SINGLE-LEG, labeled, never final) — the Gemini second leg is pending and certification (two-judge unanimity) completes when it lands."

HF README (`zk0bot/smolvla-qa-20261127`):
- (i) "**Registered deviation (DECISIONS #11):** the FL run was interrupted and resumed from round 11; the round-12 re-fit gave the optimizer a one-round AdamW moment lead. Post-round-12 states were kept as-is; rounds 12–30 completed from them."
- (ii) "**Two variables differ from Study 001's FL arm — AdamW carry AND the dataset (250 → 239 certified episodes). No result from this repo isolates carry alone.**"
- (iii) "JUDGED completion (offline replay) — a proxy metric, not physical completion."
- (iv) Section headed "## Results (preliminary, single-leg)" and "The Gemini second leg is pending; two-judge certification completes when it lands."

## Check 5 — Public hygiene: **FAIL**

Grep over the whole package folder `experiments/qa-adamw-carry-20261125/`:
- `drive.google`: 0 hits · `docs.google`: 0 hits · `100.1`: 0 hits · `ts.net`: 0 hits
- **`tailnet`: 1 hit in 1 file** — `RUN_LOG.md` ("dual tailnet + Funnel probes")
- **`/home/openclaw`: 52 hits in 9 files** — `RUN_LOG.md`, `payload_sha256.txt`, `eval/build_qa_split.py`, `eval/eval_gate_report.json`, `eval/generation_driver.sh`, `eval/judge_packets_qa.py`, `eval/review_sheets_qa.py`, `execution/chain_done.json`, `execution/qa_chain_driver3.sh`

This is not merely cosmetic in the scripts: `eval/build_qa_split.py` lines 9–13 hard-code five `/home/openclaw/muse-work/...` input paths, `eval/judge_packets_qa.py` hard-codes dataset/manifest paths under `/home/openclaw/...`, and `eval/generation_driver.sh` / `execution/qa_chain_driver3.sh` are built entirely on those paths. A guest running them as-is fails immediately and must rewrite the paths by hand. It also contradicts REPRODUCE-qa.md's own opening claim: "Nothing here points at a private file."

## Check 6 — Env lockfiles: **PASS**

- `env_lock/requirements-fl-train-venv.txt`: **186 lines, 186 pinned with `==`**, 0 unpinned.
- `env_lock/requirements-eval-venv.txt`: **160 lines, 160 pinned with `==`**, 0 unpinned.
Both non-trivial (>100 lines) and fully version-pinned.

---

## Defects a guest would hit, ordered by severity

1. **BLOCKER — `eval_traj.py` is not public.** REPRODUCE-qa.md's Eval step 3 depends on it; it exists only at a private `/home/openclaw/...` path (per `eval/generation_driver.sh`). The eval cannot be reproduced end-to-end from public assets. Fix: publish the script (branch or HF) and reference that location.
2. **HIGH — Guest-facing scripts hard-code private paths.** `eval/build_qa_split.py`, `eval/judge_packets_qa.py`, `eval/generation_driver.sh`, `execution/qa_chain_driver3.sh` (and defaults in `review_sheets_qa.py`) use `/home/openclaw/muse-work/...` paths; they fail as-is for any guest and contradict the "nothing points at a private file" claim. Fix: parameterise paths (env var / relative-to-repo defaults, as `judge_packets_qa.py` partially does with `QA_EVAL_TREE`).
3. **MEDIUM — Hygiene hits in preserved records.** 52× `/home/openclaw` across 9 files and 1× `tailnet` in `RUN_LOG.md`. Historical records (chain_done.json, eval_gate_report.json, payload_sha256.txt) are evidence and arguably should stay verbatim — but then REPRODUCE-qa.md's blanket "Nothing here points at a private file" should be scoped to the reproduction instructions, not the whole folder.
4. **LOW — `checkpoint_verify.sh` not on the branch**, though REPRODUCE-qa.md cites it as the pre-eval verification step; a guest cannot re-run that verification.
5. **LOW — `VERIFICATION.md` absent** from the package folder at verification time (REPRODUCE-qa.md promises it "when it completes").
6. **LOW — Table presentation for r10/r20:** `completion_table_qa.md` per-task rows list n=20/10 for fl_qa_r10/r20 although only 8 packets per task were judged; the totals and the reading note are correct, but a guest skimming only the per-task rows could misread the denominators.

## What would change the verdict

Publish `eval_traj.py` publicly, parameterise the `/home/openclaw` paths in the eval/execution scripts, and scope or fix the "nothing points at a private file" claim. Checks 2, 3, 4 and 6 already pass in full and would not need re-verification beyond the changed files; the numeric/hash core of the package is sound.
---

## Round 2 report — delta re-verification (commit a8f0416)

CERTIFIED WITH FINDINGS

# Q-A Phase 1 — Delta Re-Verification (independent, fresh context)

- Repo/branch: ivelin/zk0 @ `experiment/qa-adamw-carry-20261125`
- HEAD verified: `a8f0416a060494ae2308d8fa7becda75dcf32f0a` (matches claimed a8f0416)
- Package: `experiments/qa-adamw-carry-20261125/`
- Method note: the cloned content triggered an execution-risk flag, so verification was by plain text reading / read-only grep and JSON inspection only. The builder-requested `python3 -m py_compile` and `bash -n` runs were NOT executed (running/compiling cloned scripts was prohibited by that flag). No obvious syntax defects were seen on visual inspection, but a compile pass is not claimed. This is a stated limitation, not a pass.

## Per-defect results

1. **BLOCKER — eval_traj.py not public: RESOLVED**
   - `eval/eval_traj.py` exists in the package.
   - Dataset root parameterized: `DS_ROOT = os.environ.get("QA_DATASET_ROOT", "armnetbench_v01_lerobot_so101")`, with a header comment naming HF `zk0bot/armnetbench_v01_lerobot_so101 @ 89eef614`. No `/home/openclaw` (no `/home/` at all) in the file.
   - REPRODUCE-qa.md Eval step 3 points at `eval/eval_traj.py` and tells the guest to download the dataset snapshot `zk0bot/armnetbench_v01_lerobot_so101 @ 89eef614` and point `QA_DATASET_ROOT` at it.

2. **HIGH — guest scripts hard-code /home/openclaw: RESOLVED (on its defined criteria; see New Defects for two path-layout bugs found in the same scripts)**
   - `grep "/home/openclaw"` in `eval/build_qa_split.py`, `eval/judge_packets_qa.py`, `eval/review_sheets_qa.py`, `eval/generation_driver.sh` (also checked `eval/eval_traj.py`, `eval/checkpoint_verify.sh`): **ZERO hits**.
   - Shipped reference files exist and are the files the scripts read: `eval/study001_split.json` (build_qa_split via `Path(__file__).with_name(...)`), `eval/qa_gate_manifest_239_20261125.json` (same), `eval/subsets_manifest.json` (judge_packets_qa via `with_name`).
   - Training lists referenced package-relatively: build_qa_split uses `Path(__file__).resolve().parent.parent / "episode_lists"`; generation_driver.sh defaults `E` to its own folder, `W3` to `$HOME/zk0-qa-run`, python/trajectory/split all env-overridable (`QA_EVAL_TREE`, `QA_RUN_HOME`, `QA_EVAL_PYTHON`, `QA_TRAJ_SCRIPT`, `QA_SPLIT`).
   - Compile/syntax execution checks not performed — see Method note.

3. **MEDIUM — hygiene / blanket "Nothing here points at a private file" claim: RESOLVED**
   - The blanket claim is gone (no occurrence package-wide). REPRODUCE-qa.md now opens with the scoped claim: reproduction instructions and guest-facing scripts in `eval/` use only public assets and parameterized paths; preserved execution records retain the build machine's paths as as-run evidence — records, not instructions. That scoping is honest and sufficient.
   - Remaining `/home/openclaw` hits are ONLY in preserved records: `RUN_LOG.md` (2), `payload_sha256.txt` (5), `eval/eval_gate_report.json` (19), `execution/qa_chain_driver3.sh` (4), `execution/chain_done.json` (8, under `execution/`). Zero hits anywhere else.
   - `execution/qa_chain_driver3.sh` is described in REPRODUCE-qa.md (Train step 2) as "preserved … an as-run record carrying the build machine's paths".

4. **LOW — checkpoint_verify.sh not on branch: RESOLVED**
   - `eval/checkpoint_verify.sh` exists (full-state verifier: weights/optimizer/scheduler/RNG/step, hashing, integrity vs prior manifest, reload attempt).
   - REPRODUCE-qa.md Verification status references exactly that location: "the script is `eval/checkpoint_verify.sh` in this folder".

5. **LOW — VERIFICATION.md absent: OPEN ITEM (by design, not a failure this round)**
   - Still absent, as expected; it is to be filed after this verdict and must contain it. This is the one remaining open item for package completeness.

6. **LOW — completion_table_qa.md r10/r20 per-task rows misleading: RESOLVED**
   - Footnote present verbatim in `eval/completion_table_qa.md`: "Per-task rows for fl_qa_r10 / fl_qa_r20: n is the split size (20 or 10); judged packets per task = 8 (the canary), the rest UNJUDGED — see judgments_index_qa.json for per-packet status." The "Reading the table" note above it likewise states the remaining 50 packets per arm are UNJUDGED.

## Regression guard — PASS (core untouched)

- `checkpoints_manifest.json`: 9 arms; fl_r30 sha256 = `ac36f73fc29d4d0bff472229be42f01fa68f326e4f84f51431d5e16767315d38` ✓; centralized = `a1933bd27bf389bcc16a9ab2a4d71838c3efb55bc00a66d0ab7b67bf0d302e96` ✓ (both match the sealed values exactly).
- `eval/split.json`: 90 episodes (20/20/20/20/10 per task), `n_episodes` field = 90 ✓.
- `eval/eval_gate_report.json`: verdict **PASS**, waiver `W-cable-resplit-counts` present ✓.
- DECISIONS #11 deviation disclosure present in REPRODUCE-qa.md ("Registered deviation (DECISIONS #11)" — round-11 resume, one-round AdamW moment lead, states kept as-is) ✓.

## NEW defects found (do not re-open the six, but should be fixed before a guest end-to-end run of the helper scripts)

- **N1 (MEDIUM) — `eval/build_qa_split.py` cannot run as shipped.** It sets `TRAIN = …/episode_lists` and then reads `TRAIN / "run_set_manifest.json"`, but `run_set_manifest.json` lives at the package root, not inside `episode_lists/` — the read would fail with FileNotFoundError. Additionally, its split-write line is a no-op: `(EVAL / "split.json")  # package layout (as-run tree used split/split.json).write_text(...)` — the `.write_text(...)` is inside the comment, so `split.json` is never written by the script.
- **N2 (LOW) — `eval/judge_packets_qa.py` split path mismatch.** It reads `E / "split/split.json"`, but the package ships `eval/split.json` (and `generation_driver.sh` in the same folder uses `$E/split.json`). With the default `QA_EVAL_TREE` (the script's own folder) the path does not exist; a guest must hand-create the `split/` sub-layout the as-run tree used.

## Verdict rationale

All six original defects are fixed at the artifacts, and the previously passing core is byte-for-byte intact on every spot-checked value. The package is therefore CERTIFIED WITH FINDINGS rather than CERTIFIED: findings = (a) VERIFICATION.md still to be filed containing this verdict (Defect 5, by design), and (b) new defects N1/N2 in helper-script path layout, plus the disclosed limitation that compile/syntax execution checks were not run under the execution-risk restriction.

---

## Round 3 — micro-verification of the N1/N2 fixes (commit 5e76769)

A third fresh-context verifier checked only the two delta findings against
the fixed tree at `5e76769` (its full report is held in the lane's working
records; summarized here):

- **N1 RESOLVED** — `build_qa_split.py` now reads
  `(TRAIN.parent / "run_set_manifest.json")`, which resolves to the package
  root where the manifest sits beside `episode_lists/`; the
  `(EVAL / "split.json").write_text(...)` call is live code on its own line;
  and the note string the script writes matches the shipped
  `eval/qa_eval_manifest_90.json` note character for character.
- **N2 RESOLVED** — `judge_packets_qa.py` loads `(E / "split.json")` directly
  under the eval tree, and `eval/split.json` is present in the package.
- **Diff scope** — the micro-verifier's environment barred git operations
  inside its clone, so it could not independently run the
  a8f0416..5e76769 diff. From the worktree's git record (the lane's own
  repository copy), `git diff --stat a8f0416..5e76769` shows exactly the
  three expected files and nothing else:
  `eval/build_qa_split.py` (7 lines), `eval/judge_packets_qa.py` (2),
  `eval/qa_eval_manifest_90.json` (2) — 6 insertions, 5 deletions.

## Final state

**CERTIFIED WITH FINDINGS — all findings resolved.** The package at
`5e76769` + this file: quantitative core verified twice independently
(HF hashes 9/9, split/canary/gate/chain consistency, table recount),
disclosures verified in both REPRODUCE-qa.md and the HF model card,
guest-facing scripts parameterized and published, and the split builder
proven to regenerate the shipped split + canary identically. Remaining
known limitation, stated plainly: the eval's second (Gemini) judge leg is
still pending, so all completion figures in the package are SINGLE-LEG
(Muse) and labeled as such everywhere they appear; certification of the
results completes when that leg lands, at which point the completion
table and this package's results section get a follow-up commit.
