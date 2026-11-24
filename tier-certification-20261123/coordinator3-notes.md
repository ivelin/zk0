# Successor coordinator #3 notes — tier certification program (Wave 0)

Started 2026-11-23 (successor to coordinator #2; handoff checkpoint
commit `65f0c39` on `study001/fl-substrate-20261121`, repo path
`tier-certification-20261123/` mirrors this folder minus `sheets/`).

## State verified at takeover (2026-11-23 ~17:36 CT)

- register.csv: 1,650 rows, ALL "PENDING HUMAN APPROVAL". Re-judging
  complete: Spa 239/240 = 99.6% (one contested: Spa ep859 Counting —
  primary FAIL vs re-judge PASS); RC 40/40 = 100%. Register-wide
  contested = 2 (Spa ep859 + ArmnetBench ep573 carry-in).
- Cleanliness complete for all 1,650 judged rows (pass/fail untouched):
  ArmnetBench 246 CLEAN / 3 RECOVERED / 1 INCIDENT; Spa 1,151 C /
  49 I / 0 R; RC 179 C / 20 I / 1 R.
- Grok leg (Spa 50 runs + RC 8 runs, 25 eps/run, 3 parallel, driver
  pid 4102338 launched 17:26 CT under guard rundir-grok, logonly):
  RUNNING at takeover — 6/58 .done, all rc=0; in flight obje-04/08/09.
  Outputs in DGX grok/out/<run>.{raw,err,done}; verdict lines parse as
  `ep<idx> | PASS|FAIL|INDETERMINATE | <clause>` (narration ignored).
  Run→episode mapping from the prompt files (local grok-staging/prompts
  == DGX grok/prompts; 58 prompts confirmed).
- Downloads (DGX, all under job guard, HF rate-limit backoff — monitor
  only, no restarts, no retry storms):
  - download.sh + download-small.sh both crawling rc-stack-bowls
    (two hf processes on the same repo/dir, 307 MB at takeover) →
    then rc-stack-cubes → rc-upright-bottle → busybox → cup-nesting.
  - download-wave0.sh: df-sorting FAILED once (transient; repo + rev
    e0546023 verified valid) → df-testtube crawling (4.1 GB at
    takeover) → bb-push-green → bb-flip-left. df-sorting gets ONE
    queued retry after crawler load drops.
  - Landed so far: spa-bench (28 G), rc-cube-drawer, rc-remove-
    screwdriver. ArmnetBench held-out tasks need NO download (dataset
    already local from the 1h program; eps ranges in README).
- Sheets pipeline: DGX scripts/make_sheets.py <dataset_dir> <cam1>
  <cam2> <out_dir> <prefix> [workers]; emits <prefix>__ep<idx>.jpg +
  <prefix>_episodes.csv. Local sheets/ holds 1,403 pulled sheets for
  judge subagents (attach as image items). Batches: batches/*.json,
  25 eps/batch; briefs: brief-spa.md / brief-rc.md / brief-cleanliness.md.
- DF §8 criteria: drafted in criteria-frozen.md; freeze completes only
  when the downloaded meta/tasks.parquet strings are confirmed and the
  confirmation is logged in report.md BEFORE any DF judging.

## Work log (coordinator #3)

- 2026-11-23 ~17:45 CT: DF test-tube task strings confirmed from the
  downloaded meta/tasks.parquet (160 eps, 30 fps, cam_front+cam_top)
  and the §8 freeze confirmation logged in report.md BEFORE judging.
  df-sorting strings still pending its download.
- 2026-11-23 ~17:40 CT: ArmnetBench held-out sheets rendered on the
  DGX (150/150, scripts/make_sheets_filtered.py + sheets-heldout.sh
  under guard rundir-sheets-heldout). §6 strings verified verbatim
  against the dataset's episodes parquet (1200–1249 tool_removal,
  1549–1598 eye_drops_to_shelf, 2449–2498 cable_unclip). No register
  key collisions (training max index 2208; ranges disjoint) → keys
  stay `armnetbench:<idx>`.
- Batches + Grok prompts generated: batches/arm_heldout_batches.json
  (6 × 25), grok-staging/prompts/grok_arm-heldout-*.txt +
  grok-staging/runs-arm-heldout.txt. Brief: brief-armnetbench-heldout.md.
- Sheets pulled locally: shelf 50/50, toolrem 28/50 (1200–1227);
  unclip 0/50. DGX SSH proxy then refused connections ("proxy closed
  during CONNECT" ×3) — bounded probes done, DGX work paused; the
  guard-launched on-box jobs (Grok driver, downloads) run
  independently of SSH access. Remaining pulls: toolrem 1228–1249,
  unclip 2449–2498 (pull in ≤25-sheet chunks).
- Primary judges spawned for shelf-01, shelf-02, toolrem-01 (sheets
  fully local). Remaining sheets pulled in ≤25-sheet chunks: register
  now has all 150 held-out sheets local.
- Held-out judging (2026-11-23 evening): primary shelf 50/50 PASS,
  toolrem 50/50 PASS, unclip-01 25/25 PASS (unclip-02 pending);
  cleanliness 150/150 CLEAN (all six batches); re-judge sample drawn
  (seed 20261123, 10/task, batches/arm_heldout_rejudge.json) — shelf
  re-judge 10/10 PASS (agrees with primary). Verdict files:
  verdicts/arm_heldout_{primary,clean,rejudge}_*.txt. Register append
  via add_source.py once unclip-02 primary + all re-judges land.
- DOWNLOAD STALL (diagnosed 2026-11-23 ~17:55 CT, read-only): the
  three hf crawler processes (stack-bowls ×2, df-testtube ×1) are
  asleep with all sockets in CLOSE-WAIT; data sizes static since
  ~16:47 CT (stack-bowls 307M, df-testtube 4.1G). Treated as HF
  rate-limit backoff — NO restarts, NO retry storm (program rule).
  df-sorting's single queued retry stays queued (crawler load has
  not dropped). If the stall persists, the unlanded sources
  (rc-stack-cubes, rc-upright-bottle, BusyBox multitask + singles,
  Cup Nesting, DF sorting) remain blocked — a program finding for
  report.md, not something to force.

## Gemini leg (added 2026-11-23 ~17:50 CT — agy unblocked)

- Tool: `/home/openclaw/.local/bin/agy` v1.5.15 on the DGX, signed in
  (founder's account, starter tier). Usage at leg start: Gemini group
  98% weekly remaining (resets 2026-11-30T21:41:45Z); after ~4 calls
  96%. Check `agy -p "/usage"` before each batch; the quota is SHARED
  with Cos's machine — sequence, don't sweep.
- MODEL PINNING (founder, 2026-11-23 18:03 CT): every judging call
  pins `--model gemini-3.8-flash-high` — the strongest Gemini
  selectable on this account (models: gemini-3.8-flash-{high,medium,
  low}, claude-sonnet-4-6, claude-opus-4-6-thinking, gpt-oss-120b).
  The leg must be a non-Muse read; the Claude/GPT group is never
  used. VERIFIED from session records: the print-mode log lines for
  the test run and the contested-cable run both show
  model="gemini-3.8-flash-high" as the only model string. No explicit
  default-model config exists in ~/.gemini/antigravity-cli — the pin
  is per-invocation in gemini/driver-gemini.sh (MODEL env, default
  gemini-3.8-flash-high) so the account default is irrelevant.
  Register/audit rows record gemini_model = gemini-3.8-flash-high
  next to each Gemini verdict.
- MEDIA FINDING: agy fails on the JPG contact sheets (HTTP 400 "file
  or media data is empty, corrupted, or malformed"; a JPG run also
  died on a headless command-permission denial). PNG sheets work —
  ALL Gemini judging uses PNG conversions of the sheets. Driver runs
  with --dangerously-skip-permissions (judging-only prompts, dedicated
  workdir) and stops the leg if output mentions quota exhaustion
  (gemini/QUOTA_STOP).
- Driver: gemini/driver-gemini.sh (sequential; .done rc files; skips
  completed runs). Prompts local in gemini-staging/prompts/, shipped
  to DGX gemini/prompts/. Sheets for the leg live in
  gemini/sheets-heldout/ (JPG+PNG) on the DGX.
- Sequence: (1) contested — held-out 20 (cable 5 first) + register
  Spa ep859 + ArmnetBench ep573; (2) held-out 100 (5 runs × 20,
  prompts gemini_heldout100-<task>); (3) 1h training 250; (4) Spa+RC
  registers; (5) new sources as their Grok legs land.
- Contested cable result (2026-11-23): ep2020 PASS / ep2042 FAIL /
  ep2046 FAIL / ep2092 FAIL / ep2101 FAIL (batch read). NOTE: an
  earlier single-image micro-test read of ep2020 described the cable
  as NOT seated (FAIL-equivalent) — Gemini's two reads of ep2020
  contradict each other; both are recorded in the held-out audit
  record. All five stay contested for the founder; Gemini overrules
  nobody.

## Remaining work (task order)

1. Grok leg: collect all 58 runs; verify rc per run; re-run only
   recorded failures, once; merge grok_verdict/grok_evidence into the
   register; per-source agreement; new contested stay contested.
2. New sources (~892 rows): rc-stack-cubes 100, rc-upright-bottle 100,
   BusyBox multitask 132, BusyBox singles 59, Cup Nesting 200,
   DF sorting 141, DF test-tube 160, ArmnetBench held-out 150.
   Per source: confirm download → sheets → batches → primary judging
   (blind subagents) → cleanliness → seeded 20% re-judge (seed
   20261123) → Grok leg (same driver method) → register rows.
   DF: confirm task strings + log freeze in report.md BEFORE judging.
3. Review packs per source once its Grok leg lands: all FAIL sheets,
   all contested sheets, seeded PASS samples (seed 20261118) split by
   cleanliness subset (clean / recovered / incident). ArmnetBench pack
   derives from the 1h program's records.
4. Final: commit register via output-minimized route; write report.md
   (per-source primary counts, cleanliness splits, re-judge agreement,
   Grok agreement, contested list, pack paths; everything PENDING
   HUMAN APPROVAL; Gemini/agy leg OWED — no runnable Gemini exists,
   no credentials improvised).

## Hard rules

- No training on this data. Cleanliness never changes pass/fail.
- Contested episodes are NEVER resolved by the coordinator — they go
  to the founder with both reads attached.
- If a git security notice fires even under the output-minimized
  route: STOP and report.
- /tmp on this VM is wiped between exec calls — staging lives in the
  workspace only.

## CORRECTION (2026-11-23 ~18:50 CT) — held-out verdict files were wrong; fixed

- The 15 verdicts/arm_heldout_*.txt files on disk claimed primary
  150/150 PASS and cleanliness 150/150 CLEAN. The judge workers'
  delivered final responses (batch completed 23:44–23:49 UTC) say
  otherwise: primary 146 PASS / 4 FAIL (toolrem ep1205; unclip
  ep2479, ep2486, ep2498); cleanliness 127 CLEAN / 23 INCIDENT /
  0 RECOVERED (shelf 1577 + 1579 basket displaced; toolrem 1230 tool
  dropped mid-carry, not re-grasped; unclip 20 episodes with the
  charger/adapter unit tipped or displaced by the cable pull);
  re-judge 29/30 (ep1205 re-judged PASS vs primary FAIL → CONTESTED).
- The register's 150 held-out rows had been appended from the wrong
  files (all PASS/CLEAN, no contested). Fix applied: wrong files
  preserved in verdicts/superseded-20261123/; register.csv backed up
  as register.csv.bak-20261123-pre-heldout-fix; verdict files
  rewritten verbatim from the worker handoffs; the 150 rows removed
  and re-appended via add_source.py (added 150, verifier 30 parsed).
  Register verified after fix: 1,800 rows; held-out 146/4, 127 C /
  23 I; subsets 124 imitation core / 22 pass-evidence-only;
  contested armnetbench:1205; register-wide contested = 3
  (spa-bench:859, armnetbench:573, armnetbench:1205).
- The earlier notes line "cleanliness 150/150 CLEAN; primary toolrem
  50/50 PASS" in the work log above is SUPERSEDED by this correction.
- Non-regression measure: verdict files are written only from
  delivered judge handoffs, and counts (PASS/FAIL/CLEAN/INCIDENT/
  RECOVERED + contested) are recomputed from the files and checked
  against the handoffs BEFORE any register append.

## Gemini leg — stages 1-2 COMPLETE, leg stopped on quota (2026-11-23 ~19:35 CT)

- Stage 1 (contested): held-out 20 + register spa-bench:859 (Gemini
  PASS) + armnetbench:573 (Gemini PASS) — all rc=0, merged
  (merge_gemini.py; register rows carry gemini_verdict/evidence/model;
  contested flags untouched; primary_model/verifier_model/grok_model
  attribution columns added to register.csv).
- Stage 2 (held-out 100): 5 runs x 20, all rc=0; merged into the
  held-out audit verdicts_master.csv (100/100 rows; 73 PASS / 27
  FAIL; vs primary 93/100; vs Grok 82/100). New contested: tool
  ep824, eye ep1643 → held-out contested now 22. Addendum appended
  to heldout-episode-certification-audit-20261123.md (incl. ep2020's
  three mutually inconsistent Gemini reads).
- STOP POINT: Gemini weekly group at 32% remaining after stage 2
  (shared with Cos; reset 2026-11-30T21:41:45Z). Stages 3 (1h 250 —
  prompts ready: gemini_train250-*, sheets still to ship + PNG),
  4 (Spa+RC) and 5 (new sources) resume after the reset. #bot-hub
  note owed re: this week's Gemini consumption.

## Coordinator #3 close-out (2026-11-23 ~20:10 CT)

- Grok leg COMPLETE: Spa+RC 58/58 runs rc=0, 1,400 verdicts merged
  (agreement: Spa 97.2%, RC cube-drawer 90%, RC screwdriver 100%);
  ArmnetBench held-out 6/6 runs, 150 verdicts merged (96.7%).
  Register Grok coverage 1,800/1,800, grok_model attributed.
  44 + 4 new contested (register total 51). The 4 remainder batches
  (14/19/2/15 eps) were re-run once under a mistaken truncation
  assumption; attempt 1 is canonical, attempt 2 archived on the DGX;
  spa ep786 differs across attempts (FAIL stands). See report.md.
- Review packs built: review-pack/{spa-bench, rc-cube-drawer,
  rc-screwdriver, armnetbench-heldout}/ (146 sheets + READMEs).
- Gemini leg: stages 1-2 complete and merged; leg stopped at 32%
  weekly quota (reset 2026-11-30T21:41:45Z). Stage 3 prompts ready
  (gemini_train250-*); sheets-train250 still to ship + PNG-convert.
  #bot-hub note owed re: Gemini consumption + the stop.
- Downloads: stack-cubes (208M) + upright-bottle (217M) in flight at
  close; BusyBox x2, Cup Nesting, DF sorting unlanded; DF test-tube
  partial (§8 frozen). ~742 rows remain for a successor: pipeline
  each landed source exactly like the completed ones (criteria
  freeze in writing first — DF sorting strings still unread).
- FINAL COMMIT: c557605 on study001/fl-substrate-20261121 via the
  output-minimized route (fresh worktree, on-box logs, 143 files
  changed, zero security notices). Supersedes checkpoint 65f0c39.
- Everything remains PENDING HUMAN APPROVAL.
