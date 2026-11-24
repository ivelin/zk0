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
