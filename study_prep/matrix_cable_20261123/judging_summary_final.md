# zk0 Study 001 — Closed-loop CABLE matrix: Frontier Judging Layer (final frozen set)

Date: 2026-11-24. Task: cable_clip. Criterion: PASS = at the end of the rollout the DisplayPort cable is seated/held in the cable holder (clip) on the white block — the cable end segment sits in the clip channel and is retained by the clip at settle. Anything else = FAIL. Genuinely insufficient evidence = INDETERMINATE.

## Worklist construction (seeded, reproducible)

- Sources: Phase 6 records `/home/openclaw/muse-work/zk0-study001-phase6-20261123/out/records/` (416 files) + matrix records `/home/openclaw/muse-work/zk0-study001-matrix-20261123/cable/out/records/` (701 files) = 1,117 files.
- Excluded: 1 solver-failure record (matrix, spec_eye_drops, ep1903 v1) → 1,116 counted records.
- Mechanical PASS in counted set: 2. secondary_pass=true: 0.
- Worklist = all 2 mechanical PASS + seeded 10% random sample of the 1,114 counted FAIL records (seed 20261122, stratified proportionally: matrix 700→70, phase6 414→41) = **113 items**.
- Saved: `judging_worklist_final.jsonl` (this workdir). Frames (start/mid/late/end) extracted per item to `judging_frames/` in this workdir.

## Judging composition

- Muse leg (this run): COMPLETE — 113/113 items judged blind from end-weighted frames (end + late frames; zoom crops of the block/clip region for every item where the cable was at or over the block: W001, W002, W014, W015, W046, W083, W086; W006 resolved on late+end). Verdicts recorded before any Gemini result existed for any item.
- Gemini leg (agy CLI v1.5.16 on the DGX): **BLOCKED — not run.** The first agy call (model gemini-4-neon-xhigh) returned HTTP 429 RESOURCE_EXHAUSTED ("Individual quota reached… Resets in 11m3s", error_id d86e45c8-64dd-46e4-9b1e-dfa6d747d53e-1). Under the standing rate-limit rule (a 429 is a hard stop for that provider in the task attempt), no further agy calls were made, retried, or scheduled in this run. The Gemini leg remains outstanding and can be executed after the quota reset using `run_agy_leg.sh` in this workdir (batches of ~8, round-robin gemini-4-neon-xhigh → gemini-4-carbon-high → claude-opus-5-5-high, blind codes only in prompts). All gemini fields in `judging_results_final.jsonl` are null with `gemini_status` recording the block.
- **No Grok reads were run or cited anywhere in this judging layer** (founder suspended Grok trust 2026-11-23).

## Counts

- Judged (Muse leg): N = 113. Muse verdicts: PASS 2, FAIL 110, INDETERMINATE 1.
- Mechanical vs Muse: **112 agree / 1 disagree** (of 113).
- Mechanical vs Gemini: not computable — Gemini leg blocked (0 items judged).
- Muse vs Gemini: not computable — Gemini leg blocked (0 items judged).

## Mechanical PASS items — Muse confirmation

- W001 — episode 2020, variant 2, checkpoint cexp, blind p6-0065 (phase6): mechanical PASS, Muse **PASS** — cable segment seated across the clip on the block top at settle, gripper released and away; zoom crop confirms the segment in the clip channel.
- W002 — episode 2020, variant 2, checkpoint spec_cable_clip, blind p6-0236 (phase6): mechanical PASS, Muse **PASS** — cable segment engaged in the clip on the block top at settle, gripper released and away; zoom crop confirms the strand through the clip location.

## Every disagreement (mechanical vs Muse)

1. **W086 — episode 1959, variant 4, checkpoint cexp, blind p6-0114 (phase6)**
   - Mechanical: FAIL. Muse: **INDETERMINATE**.
   - Muse reason: at the final frame the gripper is still pressing a cable segment into the clip on the block top (zoom crops, both views); whether the clip alone retains the segment after release cannot be confirmed because the rollout ends with the gripper still engaged. Not scored PASS (retention unproven) and not scored FAIL (segment is at/in the clip, not away from it).

No other disagreements. All 110 remaining sampled mechanical FAILs were confirmed FAIL by the Muse leg (cable on the mat, coiled beside/behind the block, flung outside the tray, or held only by the gripper in the air; clip channel visibly empty in the top view).

## Notes

- Two pairs of worklist items share an identical end-frame hash (W026/W027; W087/W089) — different blind codes/records with coincidentally identical final frames; each was judged on its own record.
- Results file: `judging_results_final.jsonl` (one line per item: worklist id, source, blind code, episode/variant/checkpoint unblinded at write time, mechanical verdict, Muse verdict + reason, Gemini fields null with block status, agreement flags).
