# Successor coordinator #4 notes — tier certification program (Wave 0)

Started 2026-11-23 (successor to coordinator #3; handoff checkpoint
commit `c7f79a2` on `study001/fl-substrate-20261121` per brief;
coordinator #3 close-out commit was `c557605`, register + notes in
this folder mirror the repo path `tier-certification-20261123/`).

## State verified at takeover (from coordinator3-notes.md + report.md)

- register.csv: 1,800 rows, ALL "PENDING HUMAN APPROVAL", full Grok
  coverage (1,800/1,800). Primary + cleanliness + seeded re-judge
  complete for: ArmnetBench training 250, ArmnetBench held-out 150,
  Spa-Bench 1,200, RoboColosseum cube-drawer 100 + remove-screwdriver
  100. Review packs built for those sources (review-pack/).
- Contested: 51 register + 22 held-out (separate audit record) — all
  unresolved for the founder; NEVER resolved by the coordinator.
- Gemini leg: stages 1–2 complete (contested + held-out 100);
  stages 3–5 resume after the 2026-11-30 quota reset — OUT OF SCOPE
  for coordinator #4 (scheduled separately). No agy work here.
- NON-REGRESSION RULE (coordinator #3 defect find, binding): verdict
  files are written ONLY from delivered judge handoffs; counts are
  recomputed from the files and checked against the handoffs BEFORE
  any register append. No exceptions.
- Downloads at #3 close (~20:10 CT): rc-stack-cubes (208M) +
  rc-upright-bottle (217M) in flight under HF rate-limit backoff;
  BusyBox multitask (132) + singles (59), Cup Nesting (200),
  DF sorting (141) unlanded; DF test-tube (160) partial, §8 criteria
  FROZEN (report.md log, 2026-11-23). DF sorting §8 task strings
  still unread; its ONE queued retry stays queued — no parallel
  retries, no retry storms, respect the backoff.
  NOTE: rc-stack-bowls (100) was crawling at #3 takeover (307M) but
  is absent from #3's close-out blocked list — verify its state on
  the DGX at takeover.
- Remaining per source (eps): stack-bowls 100 (?), stack-cubes 100,
  upright-bottle 100, BusyBox multitask 132, BusyBox singles 59,
  Cup Nesting 200, DF sorting 141, DF test-tube 160.
- Pipeline per landed source (identical to completed sources):
  confirm download → criteria freeze in writing BEFORE judging (if
  not already frozen; DF sorting strings → log freeze in report.md)
  → contact sheets (DGX scripts/make_sheets.py) → batches of 25 →
  blind primary judging (separate-context subagents, briefs in this
  folder) → cleanliness pass (CLEAN/RECOVERED/INCIDENT, one-line
  incidents) → seeded 20% blind re-judge (seed 20261123) → Grok leg
  (DGX /home/openclaw/.local/bin/grok, contact sheets, web search
  off, rc recorded per run) → register append via add_source.py →
  review pack (all FAILs, all contested, seeded PASS samples seed
  20261118 split by cleanliness subset).
- Commits: output-minimized route only (fresh worktree, output to
  on-box logs, return only exit codes/hashes/counts). If a security
  notice fires even under that route: STOP and report.
- Hard rules: no training on this data; cleanliness never changes
  pass/fail; contested never resolved by the coordinator; everything
  stays PENDING HUMAN APPROVAL; /tmp on this VM is wiped between
  exec calls — staging lives in the workspace only.

## Work log (coordinator #4)

- Takeover: read coordinator3-notes.md, report.md, README.md;
  register header + 1,800 data rows confirmed locally.
- Download state verified on the DGX (episode-referenced file
  inventory vs disk, per source): rc-stack-bowls COMPLETE (100 eps,
  0 missing), rc-stack-cubes COMPLETE (100, 0 missing), bb-push-green
  COMPLETE (39, 0 missing), bb-flip-left COMPLETE (20, 0 missing) —
  these four were missing from coordinator #3's close-out blocked
  list because the crawlers finished them late on 2026-11-23.
  df-testtube: 7 cam_front video files still missing (partial).
  rc-upright-bottle: 14 .incomplete files, hf processes alive
  (started 19:12 CT) — still in flight. BusyBox multitask +
  Cup Nesting: not started on disk; download.sh/download-small.sh
  sequences reach them after upright-bottle. DF sorting: absent;
  download-wave0.sh has exited (its list ended at bb-flip-left), so
  the ONE queued retry + the df-testtube remainder are successor
  actions for when crawler load drops — not yet run.
- Sheets rendered on the DGX under guard (rundir-sheets-rc2, exit 0):
  rc-stack-bowls 100, rc-stack-cubes 100, bb-push-green 39,
  bb-flip-left 20 (front+wrist for RC, front+top for BusyBox).
  All 259 sheets + 4 episodes CSVs pulled to this VM (verified
  counts + JPEG decode spot-checks). Pull script logs false FAILs:
  tar exits 2 on ownership-change (files extract intact) — verify
  by counts, not the pull log.
- Briefs written: brief-rc-stack.md (stack-bowls/stack-cubes,
  criteria verbatim from criteria-frozen §3), brief-busybox-singles.md
  (§4/§7 rules). Batches + Grok prompts generated (gen_batches.py):
  batches/rc_stack_bowls_batches.json (4), rc_stack_cubes_batches.json
  (4), bb_push_green_batches.json (2), bb_flip_left_batches.json (1);
  grok runs grok_rc-stack-bowls-01..04, grok_rc-stack-cubes-01..04,
  grok_bb-push-green-01..02, grok_bb-flip-left-01.
- Primary judging waves 1–3 spawned (11 judges). FORMAT DEVIATION:
  the first two returned handoffs (stack-bowls-01, stack-bowls-02)
  came back as prose/table summaries that ALSO self-graded
  cleanliness in the same context (25 PASS / 25 CLEAN each). Their
  primary verdicts are usable and will be transcribed verbatim into
  verdict files; their cleanliness self-grades are DISCARDED —
  cleanliness runs as its own separate-context pass per Amendment 2.
  (In the event, EVERY primary/re-judge judge merged a cleanliness
  self-grade into its handoff despite explicit format instructions;
  all self-grades discarded the same way.)
- PRIMARY RESULTS (all four landed sources, files in verdicts/,
  transcribed from the delivered handoffs): stack-bowls 100/100
  PASS (rc_primary_bowls01-04.txt); stack-cubes 100/100 PASS
  (rc_primary_cubes01-04.txt — cubes-04 judge noted, inside its
  primary evidence, a human hand touching the stack in ep80's final
  frame and alignment glitches in ep85/93/97; those observations are
  preserved in the primary evidence text and the episodes go through
  the independent cleanliness pass like all others); bb-push-green
  39/39 PASS (bb_primary_push01-02.txt); bb-flip-left 20/20 PASS
  (bb_primary_flip01.txt).
- RE-JUDGE (seed 20261123, batches/rc2_rejudge.json): bowls sample
  20/20 PASS = 100% agreement (rc_rejudge_bowls.txt); cubes sample
  20/20 PASS = 100% (rc_rejudge_cubes.txt); bb-push sample 8/8 PASS
  = 100% (bb_rejudge_push.txt); bb-flip sample (4 eps) pending.
- CLEANLINESS: 11 separate-context batches spawned (bowls x4,
  cubes x4, push x2, flip x1) — handoffs pending at this note.
- GROK LEG: merge_grok.py SHEET2KEY extended with the four new
  sheet prefixes (rc-stack-bowls, rc-stack-cubes, bb-push-green,
  bb-flip-left). 11 prompts shipped to DGX grok/prompts/;
  scripts/driver-rc2.sh (mirror of driver.sh, reads grok/runs-rc2.txt,
  touches grok/GROK_RC2_DONE) launched under guard rundir-grok-rc2
  on 2026-11-23 ~19:36 CT. Collect .done rc + .raw per run; re-run
  only recorded failures, once.
- Downloads: upright-bottle still crawling (13 .incomplete at last
  check, +1 mp4 over ~1 h — HF backoff; monitor only).
