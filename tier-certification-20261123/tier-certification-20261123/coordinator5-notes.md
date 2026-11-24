# Successor coordinator #5 notes — tier certification program (Wave 0)

Started 2026-11-23 (successor to coordinator #4, who ended mid-flight:
"Upright-bottle is still downloading (two hf processes alive, 316 MB
so far) — monitor-only. Waiting on the Grok leg to finish before
merging and building the review packs.")

## State verified at takeover (2026-11-23 ~19:42 CT, all at source)

- register.csv: 2,059 data rows, ALL "PENDING HUMAN APPROVAL".
  Complete (primary + cleanliness + seeded re-judge + Grok):
  ArmnetBench 400 (250 train + 150 held-out), Spa-Bench 1,200,
  RC cube-drawer 100, RC remove-screwdriver 100.
- NON-REGRESSION CHECK PASSED for the four rc2 sources: counts
  recomputed from verdicts/ files match register.csv exactly,
  0 mismatches (primary, cleanliness, verifier columns):
  - rc-stack-bowls: 100 PASS; 100 CLEAN; re-judge 20/20 PASS.
  - rc-stack-cubes: 100 PASS; 86 CLEAN / 13 INCIDENT / 1 RECOVERED;
    re-judge 20/20 PASS; 2 contested in register.
  - bb-push-green: 39 PASS; 39 CLEAN; re-judge 8/8 PASS.
  - bb-flip-left: 20 primary PASS; re-judge FAIL 4/4 sampled; third
    read FAIL 20/20; fourth read FAIL 20/20 (both extra reads from
    cleanliness workers that returned pass/fail instead of grades —
    preserved as delivered in verdicts/bb_flip_third-read.txt and
    bb_flip_fourth-read.txt). ALL 20 flagged contested for the
    founder. Cleanliness UNGRADED after two attempts (register notes
    document this per-row). Coordinator does NOT resolve.
- Review packs present: armnetbench-heldout, rc-cube-drawer,
  rc-screwdriver, spa-bench. MISSING: rc-stack-bowls, rc-stack-cubes,
  bb-push-green, bb-flip-left (build after Grok merge).
- GROK LEG rc2: STILL RUNNING on the DGX at takeover — driver-rc2.sh
  under guard rundir-grok-rc2b (relaunched by #4 as rc2b after an
  rm of stale marker files), xargs -P 3 over grok/runs-rc2.txt
  (11 runs: bowls 01-04, cubes 01-04, push 01-02, flip 01). At
  takeover: bowls 01-03 in flight (.raw/.err present, no .done yet),
  GROK_RC2_DONE absent. Register grok columns for all four sources
  are empty (not yet merged).
- DOWNLOADS on the DGX (workdir
  /home/openclaw/muse-work/zk0-tier10h-cert-20261123):
  - rc-upright-bottle: 316 MB on disk, 12 .incomplete files, two hf
    processes alive (PIDs 57925/58012, started 19:12 CT) — one each
    from download.sh (guard rundir-download) and download-small.sh
    (guard rundir-download-small), same local-dir. MONITOR ONLY.
  - download-small.sh sequence after upright-bottle: BusyBox
    multitask (armnet/busybox_multitask, 132 eps) → Cup Nesting
    (DecisionFacts/Physical_AI_SO101_Cup_Nesting_Task, 200 eps).
  - df-testtube: 195 .mp4 on disk, 0 .incomplete — #4 counted 7
    cam_front files missing at ITS takeover; completeness vs the
    episode manifest still to be verified by #5.
  - DF sorting: absent on disk; download-wave0.sh has exited. Its
    ONE queued retry stays single — run only when crawler load
    drops; no parallel retries, no storms.
- Remaining pipeline per landed source: criteria freeze in writing
  BEFORE judging (DF sorting §8 waits on its task strings; DF
  test-tube §8 already FROZEN per #3) → sheets → batches of 25 →
  blind primary → cleanliness (separate context) → seeded 20%
  re-judge (seed 20261123) → Grok leg → register append via
  add_source.py → review pack (build_review_packs.py pattern,
  seed 20261123, cap 12/subset).
- Commits: output-minimized route only (fresh worktree, on-box
  logs, return only exit codes/hashes/counts). Security notice even
  under that route = STOP and report.
- Hard rules: no Gemini/agy work (stages 3-5 scheduled Nov 30,
  separate lane); no training on this data; cleanliness never
  changes pass/fail; contested never resolved by the coordinator;
  verdict files ONLY from delivered judge handoffs, counts
  recomputed before any register append; everything stays
  PENDING HUMAN APPROVAL.

## Work log (coordinator #5)

- Takeover: read coordinator4-notes.md (+ #3 state via #4's summary),
  register header/counts, verdicts/ inventory. Verified DGX state
  directly (grok driver alive, hf download processes alive).
- Non-regression recompute for rc2 sources: 0 mismatches (above).
- Briefs pre-written for the four unlanded sources (criteria verbatim
  from criteria-frozen.md): brief-rc-upright.md (§3),
  brief-busybox-multitask.md (§4), brief-cup-nesting.md (§5),
  brief-df-testtube.md (§8b, frozen strings).
- df-testtube completeness VERIFIED at the source (episodes parquet
  vs disk): 160 episodes in meta; 7 video files still missing —
  cam_front file-038/042/048/053/075/096 + cam_top file-013.
  Remainder fetch runs when crawler load drops (after
  upright-bottle), sequentially with DF sorting's single retry.
- GROK LEG rc2 COMPLETE for stack-bowls: 4/4 runs rc=0, 100 verdicts
  merged (merge_grok.py; grok-merge-summary3.json; register backed
  up pre-merge as register.csv.bak-pre-rc2-grok-merge; re-merge of
  the first leg's 1,550 verdicts idempotent — agreement numbers
  unchanged). Bowls agreement 50/100: runs 01/02 read PASS 50/50;
  runs 03/04 read FAIL on ALL of ep50-99 with evidence describing the
  SAME end state ("one empty bowl remains / one bowl left alone on
  the table, gripper empty") that runs 01/02 called a nested stack.
  Flagged as a probable batch-level Grok artifact (cf. the Spa
  ep204-224 block); the reads STAND as recorded; all 50 episodes
  (rc-stack-bowls:50-99) are now contested for the founder.
  Register-wide contested is now 123.
- GROK LEG BLOCKED for rc-stack-cubes (100), bb-push-green (39),
  bb-flip-left (20): all 7 runs failed with HTTP 402 "Grok Build
  usage balance exhausted" — first attempt under #4's driver AND
  the one authorized re-run under coordinator #5 (guard
  rundir-grok-rc2c, 2026-11-23 19:49 CT), same 402. Failed-attempt
  outputs archived on the DGX (grok/out/attempt1-402/). No further
  retries: the Grok Build balance is a founder/fleet-side resource.
  Review packs for these three sources wait on their Grok leg
  (program rule: pack once the source's Grok leg lands).
- REVIEW PACK built: review-pack/rc-stack-bowls/ (0 FAIL, 50
  contested, 12 seeded PASS+CLEAN samples; seed 20261123 per
  build_review_packs.py — the prose notes' "seed 20261118" is
  superseded by the script the existing packs were built with).
- Downloads at last check: rc-upright-bottle still 316 MB / 12
  .incomplete, both hf processes alive (HF backoff; monitor only).
  BusyBox multitask + Cup Nesting queue behind it in both download
  scripts.
- CHECKPOINT COMMITTED: a7fb537 on study001/fl-substrate-20261121
  via tiercert_commit5.sh (output-minimized; 66 files changed;
  staging refreshed to commit-staging/program5/ on the DGX; repo-vs-
  staging diff checked first: 0 deletions, 61 new files).
- GEMINI STAGES RUN (founder order 2026-11-23 ~20:05 CT superseded
  the Nov 30 wait; coordinator #5 executed): /usage at leg start =
  Gemini group 33% remaining. Stage 3 sheets (250 audit JPGs)
  shipped in per-task chunks and PNG-converted on the DGX
  (gemini/sheets-train250/, 250 PNGs, size-verified vs the JPGs);
  train250 prompts were already on the DGX (md5-verified vs local).
  Driver ran chunk A under guard rundir-gemini-train250a.
  **RESULT: 3/10 stage-3 runs completed, all rc=0** —
  gemini_train250-block_stack-1, block_stack-2, cable_clip-1 =
  75 episodes merged into the register (merge_gemini.py,
  key-explicit; backup register.csv.bak-pre-gemini-stage3).
  Gemini verdicts: 74 PASS / 1 FAIL — the FAIL is ep604, concurring
  with primary + Grok. Agreement: vs primary 75/75; vs Grok 74/75
  (ep573: Gemini PASS, siding with primary + the founder's
  adjudication over Grok's FAIL). Register Gemini coverage now
  76/2,059. Note: block_stack-1's raw re-emitted verdict lines for
  ep580-584 (both emissions PASS; the merge took the last, per
  parse order).
  **LEG STOPPED ON THE QUOTA FLOOR**: after the 3 runs /usage read
  5% remaining — under the 10% floor in the run order (~9.3% of the
  weekly group per 25-PNG run). No further Gemini calls were made.
  Quota resets 2026-11-30T21:41:45Z.
  **EXACT RESUME POINT for the Nov 30 fallback run** (do not redo
  completed runs — the driver skips .done runs anyway):
  - Stage 3 remaining (7 runs, 175 eps): gemini_train250-cable_clip-2,
    eye_drops_to_basket-1, eye_drops_to_basket-2, ring_insert-1,
    ring_insert-2, tool_insert-1, tool_insert-2. Prompts + PNG
    sheets are already on the DGX.
  - Stage 4 NOT started (Spa 1,200 + RC 200): prompts generated
    locally by gen_gemini_stage45.py (gemini-staging/prompts/
    gemini_spa-01..48, gemini_rc-cube-drawer-01..04,
    gemini_rc-screwdriver-01..04; run lists in gemini-staging/
    runs-gemini_*.txt). Sheets still to ship: local sheets/spa__*.jpg
    → DGX gemini/sheets-spa/ (PNG), sheets/rc-cube-drawer__*.jpg +
    rc-screwdriver__*.jpg → gemini/sheets-rc/ (PNG).
  - Stage 5 NOT started (rc2 sources, 259 eps): prompts generated
    (gemini_rc-stack-bowls-01..04, gemini_rc-stack-cubes-01..04,
    gemini_bb-push-green-01..02, gemini_bb-flip-left-01). Sheets
    still to ship → gemini/sheets-rc2/ (PNG).
  - At the observed burn rate (~9.3%/run), the full remaining leg
    (7 + 56 + 11 = 74 runs) needs ~7 weekly quota cycles — the
    Nov 30 run should expect to complete only part of stage 3/4
    unless the quota accounting changes; sequence strictly in the
    stage order above and re-check /usage between chunks.
- FLIP CLEANLINESS CLOSED (third attempt): a fresh separate-context
  grader with a hardened brief (pass/fail explicitly out of scope)
  returned grades in format: 20/20 CLEAN (verdicts/bb_clean_flip01.txt,
  written verbatim from the delivered handoff; counts recomputed
  before the register edit). Register updated in place (backup
  register.csv.bak-pre-flip-clean): flip rows now cleanliness CLEAN
  ×20, subset imitation core (derived from primary PASS per program
  rule), notes amended; all 20 remain contested on the pass/fail
  dispute. Flip is no longer the register's only ungraded source.
  This register change postdates checkpoint a7fb537 — it goes into
  the next checkpoint commit.

## GEMINI STAGE 3 COMPLETE + STAGE 4 LAUNCHED (coordinator #5, 2026-11-23 ~21:1x CT)
- AI Pro upgrade verified at source (/usage 100% both groups before the resume; plan: Google AI Pro 5 TB, $19.99/mo billed via App Store). Claim on the remaining leg recorded in coordinator6-notes.md BEFORE launching; no other driver active at launch (ps verified).
- Stage-3 remainder: 6 runs (eye/ring/tool ×2) ALL rc=0, 25 verdict lines each. Merged (backup register.csv.bak-pre-gemini-stage3b). STAGE 3 TOTAL (train-250): Gemini 249 PASS / 1 FAIL (ep604) · vs primary 250/250 (100%) · vs Grok 249/250 (99.6%; lone divergence ep573 — Gemini PASS = primary + founder adjudication; Grok FAIL stands as dissent).
- Quota burn at Pro tier: 6 runs ≈ 1% of the weekly Gemini limit (99% remaining post-merge), ~2% of the 5-hour limit. Register Gemini coverage: 251/2,059.
- STAGE 4 LAUNCHED: Spa 48 runs (rundir-gemini-spa, driver sequential, canonical #5 prompts + sheets-all symlinks). RC 8 runs (cube-drawer 4 + screwdriver 4) follow Spa; then STAGE 5: rc2 11 runs (#5 prompts) + ArmnetBench held-out 150 (6 runs, #6's gemini_arm-heldout-* prompts; sheets via gemini/sheets-all/). Usage checks between chunks.
- FOUNDER TIE-BREAK RECORDED (2026-11-23 21:31 CT): held-out cable_clip ep2020 = FAIL (ruled from the full 414-frame video; frame count verified = dataset record length). Primary FAIL stands, Grok PASS overruled, Gemini split reads noted. Recorded in heldout-audit-20261123/verdicts_master.csv (certification_status = FOUNDER ADJUDICATED: FAIL) + addendum in heldout-episode-certification-audit-20261123.md + founder-tiebreak-20261123/VERDICTS.md. Updated files pushed to DGX commit-staging/heldout/ — they fold into the NEXT tiercert commit (not committed standalone; Gemini stage 4 merge not interrupted).
- FOUNDER TIE-BREAK #2 RECORDED (2026-11-23 21:41 CT): held-out cable_clip ep2042 = FAIL (full video; frames verified 1025/1025). Primary FAIL stands, Grok PASS overruled, Gemini concurred FAIL. Recorded in verdicts_master.csv (2 adjudicated rows total now: ep2020 + ep2042), audit addendum extended, tie-break log updated (incl. the 1025-frame completeness note). Updated files re-pushed to DGX commit-staging/heldout/ — folds into the next tiercert commit with the ep2020 record.
- FOUNDER TIE-BREAK #3 RECORDED (2026-11-23 21:47 CT): held-out cable_clip ep2046 = FAIL (full video; frames verified 464/464). Primary FAIL stands, Grok PASS overruled, Gemini concurred FAIL. verdicts_master.csv now holds 3 founder-adjudicated rows (ep2020, ep2042, ep2046); audit addendum + tie-break log updated; staged files re-pushed — folds into the same next tiercert commit as #1–2.
- PROTOCOL ADDITION RECORDED (founder, 2026-11-23 21:50 CT): contested-episode judge reads must add (a) chronological trajectory walkthrough + (b) anomalies/insights. Written into criteria-frozen.md (Amendment 3), the four pending briefs (busybox-multitask, cup-nesting, df-testtube, rc-upright), and report.md. Register work unchanged; the held-out narration pass runs separately from the tie-break folder.
- FOUNDER TIE-BREAK #4 RECORDED (2026-11-23 21:52 CT): held-out cable_clip ep2092 = FAIL (full video; frames verified 902/902). Primary FAIL stands, Grok PASS overruled, Gemini concurred FAIL. verdicts_master.csv now holds 4 founder-adjudicated rows (ep2020, ep2042, ep2046, ep2092); addendum + tie-break log updated; staged files re-pushed — same next tiercert commit batch as #1–3. Remaining Grok-PASS cable contest: ep2101 only.
- GROK TRUST SUSPENDED — APPLIED (founder ruling 2026-11-23 21:53 CT): register — all 1,900 grok-read rows flagged DISTRUSTED in notes (backup register.csv.bak-pre-grok-distrust; verdict data untouched); Class A lone-dissent rows cleared of contested: 99 (rc-stack-bowls 50, spa 34, rc-cube-drawer 10, armnetbench 5 incl. ep573) → contested 123 → 24. criteria-frozen.md Amendment 4 + report.md ruling section written; held-out audit md ruling line added + staged. Grok legs for cubes/push/flip CANCELLED; their packs proceed on Muse+Gemini. STANDING DUTY: at each remaining Gemini merge, re-flag contested on any cleared row where Gemini dissents from primary. Trust re-check 2026-12-23.
- FOUNDER TIE-BREAK #5 RECORDED (2026-11-23 21:56 CT): held-out cable_clip ep2101 = FAIL (full video; frames verified 1145/1145). Primary FAIL stands, Grok PASS overruled, Gemini concurred FAIL. ALL FIVE contested cable episodes now founder-adjudicated FAIL (2020/2042/2046/2092/2101) — primary cable 0/20 stands in full. verdicts_master.csv holds 5 adjudicated rows; addendum + tie-break log updated (log carries the closure note); staged files re-pushed — same next tiercert commit batch as #1–4.
- AGY ROUND-ROBIN (founder rule 2026-11-23 21:54 CT; menu verified live ~21:58 CT): rotation per 25-ep run = gemini-4-neon-xhigh → gemini-4-carbon-high → claude-opus-5-5-high, repeat; 3.8-flash excluded (past reads stand). Full verified menu recorded in report.md. Driver: gemini/driver-gemini-rr.sh (line-number assignment, per-run .model files). Swap from the pinned driver happens at the Spa stage-4 boundary between runs; merges must read .model per run (fallback constant only for pre-rotation runs).
- FOUNDER TIE-BREAK #6 RECORDED (2026-11-23 22:03 CT): held-out block_stack ep489 = PASS (full video; frames verified 451/451). Reads stated correctly: primary PASS (borderline-flagged) vs Grok FAIL vs Gemini FAIL ("gripper remains clamped on the orange block without releasing it") — founder overruled both Grok and Gemini. verdicts_master.csv now holds 6 adjudicated rows; addendum + tie-break log updated; staged files re-pushed — same next tiercert commit batch.
- FOUNDER SPOT-CHECKS OF CONSENSUS PASSES (begun 2026-11-23 evening): random draws of unanimous-PASS episodes being reviewed by the founder — held-out: tool 902, eye 1746, ring 179, block 526, block 549, tool 630, eye 1814, ring 31; training set: cable 2159, cable 2179. His rulings will arrive one at a time and are recorded on receipt (no pre-emptive register changes).
- FOUNDER SPOT-CHECK #1 RECORDED (2026-11-23 22:03 CT): held-out tool_insert ep902 = PASS, founder concurs with the unanimous judge PASS (full video 296/296). verdicts_master.csv certification_status = "FOUNDER SPOT-CHECK: PASS — HUMAN APPROVED"; audit addendum + tie-break log spot-check section updated; staged — same next tiercert commit batch. Remaining spot-check draws awaiting rulings: eye 1746, ring 179, block 526, block 549, tool 630, eye 1814, ring 31 (held-out); cable 2159, 2179 (training set — those record in the 1h-tier audit/register on receipt).
- FOUNDER REVISION + RELEASE STANDARD APPLIED (2026-11-23 22:05 CT): ep489 PASS ruling revised to INDETERMINATE (both kept; CSV status carries the history; VERDICTS row marked revised). NEW STANDARD (criteria-frozen Amendment 5 + report.md): certified PASS requires gripper VISIBLY OPEN at end; closed-at-end = INDETERMINATE. Retroactive text scan (separate release_standard_flag column; verdicts untouched; backups register.csv.bak-pre-release-std): register 3 flags (armnetbench:573, spa 1072, spa 1133); held-out 100: 6 flags (489 founder-ruled, 453/499/514 borderline, 557/784 gripper-still). Queue file: release-standard-queue-20261123.md. Agy prompts: RELEASE STANDARD clause appended to 109 un-run prompt files; in-flight gemini_spa-14 (model confirmed gemini-4-carbon-high — RR rotation live) kept its original prompt, covered by scan. Scan limit noted in report (text-only; starting set).
- FOUNDER SPOT-CHECK #2 RECORDED (2026-11-23 22:05 CT): held-out ring_insert ep179 = PASS, founder concurs unanimous PASS (250/250). verdicts_master.csv spot-check rows now 2 (ep902, ep179); staged — same commit batch.
- FOUNDER SPOT-CHECK #3 RECORDED (2026-11-23 22:06 CT): held-out ring_insert ep31 = PASS, founder concurs unanimous PASS (226/226). Spot-check rows in verdicts_master.csv now 3 (ep902, ep179, ep31); staged — same commit batch. Spot-check draws still awaiting rulings: eye 1746, block 526, block 549, tool 630, eye 1814 (held-out); cable 2159, 2179 (training).
- SPOT-CHECK #4 + CABLE RE-SPLIT IMPLEMENTED (founder, 2026-11-23 22:11 CT): ep2159 spot-check PASS in register human_status. Re-split: eligible 49/50 (only ep2201 excluded); seed-20261123 draw = 2164, 2169, 2170, 2183, 2190, 2193, 2194, 2195, 2200, 2208 (all CLEAN) → new held-out cable set; remaining 40 = training. Manifest cable-resplit-manifest-20261123.md published in the program folder; register drawn rows noted; verdicts_master.csv: 20 cable rows eval_set_status=REMOVED (new column; others ACTIVE). Held-out audit + training audit addenda written. This batch (with the Grok-distrust flags, release-standard flags, adjudications #1-6, spot-checks #1-3, protocol amendments) goes into checkpoint commit 9.
- FOUNDER SPOT-CHECK #5 RECORDED (2026-11-23 22:13 CT): training cable ep2179 = PASS, founder concurs unanimous (927/927); register human_status updated. NOTE: commit 9's first launch attempt failed at the script-ship step (tar error; script never arrived, nothing ran) — re-shipped and verified before relaunch; ep2179 rides the same commit 9 batch.
