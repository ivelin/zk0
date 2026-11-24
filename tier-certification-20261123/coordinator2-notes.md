# Successor coordinator notes — tier certification program (Wave 0)

Started 2026-11-23 (successor to the first coordinator; handoff checkpoint
commit `7954113` on `study001/fl-substrate-20261121`, repo path
`tier-certification-20261123/` mirrors this folder minus `sheets/`).

## State verified at takeover (2026-11-23 ~17:21 CT)

- register.csv: 1,650 rows, ALL "PENDING HUMAN APPROVAL".
  - ArmnetBench 250 carried in (249 PASS / ep604 FAIL; Grok leg done in the
    1h program; verifier sample 50 filled; contested: ep573 only).
    Cleanliness: NOT done (250 blank).
  - Spa-Bench 1,200 primary judged (1,193 PASS / 7 FAIL: ep110, 120, 159,
    497, 518, 859, 891). Cleanliness 619/1,200 (591 CLEAN / 28 INCIDENT /
    0 RECOVERED); remainder = 581 (Referential 150, Ordinal 127,
    Counting 164, Relative Size 140; indices 415–1199).
  - RoboColosseum cube-drawer 100 + remove-screwdriver 100: primary all
    PASS; cleanliness complete (179 CLEAN / 20 INCIDENT / 1 RECOVERED
    across both).
- Re-judge samples drawn (seed 20261123): Spa 240 in
  batches/spa_rejudge.json (13 batches); RC 40 in batches/rc_rejudge.json.
- Outstanding sources (~892 fresh rows): rc-stack-bowls 100,
  rc-stack-cubes 100, rc-upright-bottle 100, BusyBox multitask 132,
  BusyBox singles 59, Cup Nesting 200, DF sorting 141, DF test-tube 160,
  ArmnetBench held-out tasks 150 (eye_drops_to_shelf / tool_removal /
  cable_unclip). Total fresh = 2,542.
- DGX workdir: ~/muse-work/zk0-tier10h-cert-20261123 (data/, sheets/,
  scripts/, logs/, rundirs). Downloads running under job guard (logonly,
  JOB_LOCK=0, JOB_GUARD_ALLOW_NO_MIRROR=1 — standing mirror fault):
  download.sh, download-small.sh (on rc-stack-bowls), download-wave0.sh
  (df-sorting → df-testtube → bb singles). HF rate-limit backoff in
  effect — monitor only, no restarts, no retry storms.
- DF criteria: criteria-frozen.md §8 drafted from tier-record task
  descriptions; the freeze completes only when the downloaded
  meta/tasks.parquet strings are confirmed — log the confirmation in
  report.md BEFORE any DF judging.
- Grok method (from the 1h stage): workdir prompts, `grok
  --prompt-file <f> --output-format plain --disable-web-search
  --max-turns 80`, 25 eps/run, 3 parallel, rc in .done files, verdict
  lines parsed from .raw (narration lines ignored).

## Work log

- 2026-11-23: Spa cleanliness remainder batched (24 × 25) →
  batches/spa_clean_remainder.json. ArmnetBench cleanliness batched
  (10 × 25 by task) → batches/arm_clean.json.
- 2026-11-23: Re-judge wave 1 spawned (spa a-01..a-04, b-01, b-02,
  c-01, c-02). Returned so far: a-01 25 PASS, a-02 25 PASS, a-03 25
  PASS, a-04 7 PASS, c-02 9 PASS (all agree with primary so far —
  formal agreement computed at merge).
- 2026-11-23: Re-judge wave 2 spawned (spa d-01, e-01, e-02, f-01,
  f-02; rc cube 20, rc screw 20).
- 2026-11-23: Grok prompts for Spa (50 runs) + RC (8 runs) generated →
  grok-staging/ locally; shipping to DGX grok/prompts; driver mirrors
  the 1h-stage driver2.sh (3 parallel, .done rc files).
- /tmp on this VM is wiped between exec calls — keep staging in the
  workspace, never /tmp.
- 2026-11-23: Grok prompts confirmed on the DGX (58/58 + runs.txt 58);
  driver launched under the guard (rundir-grok, logonly).
- 2026-11-23: DF-SORTING DOWNLOAD FAILED ONCE in download-wave0.sh
  (hf error swallowed by the script's `| tail -1`; only the HF_DEBUG
  hint line survived). Diagnostic: repo + pinned revision e0546023
  verified valid (HF API HTTP 200) — failure treated as transient
  (rate limit / network). Wave0 moved on to df-testtube (4.1 GB and
  crawling). ACTION: one single df-sorting retry only after current
  crawler load drops; no parallel retry while 3 hf processes crawl.
