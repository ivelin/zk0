# Closed-loop cable matrix — final report (2026-11-24)

**Labels (apply to every number below):** closed-loop results in the certified twin-suite v1.1
cable_clip twins (param_hash `a4eb3f8c3611894d`); **sim bar unanchored to real-arm delta**.
Phase 6 checkpoints are **stopped by founder rule, partial**. Matrix production:
**matrix cable production stood down by founder ruling 2026-11-24; cable evaluation moves to
the eval plan (real held-out + validator sets)**.

## Headline

2 mechanical passes in 1,116 counted rollouts (1,117 record files; 1 solver-failure exclusion).
Both passes are the same unit — episode 2020, variant 2 — under two Phase 6 checkpoints
(cexp, spec_cable_clip). Every other checkpoint scored zero.

## Per-checkpoint results (cable_clip, mechanical criterion, frozen v1.1 scorer)

### Phase 6 source — stopped by founder rule, partial (416 records)

| checkpoint | passes / counted | note |
|---|---|---|
| base | 0 / 96 | partial |
| cexp | 1 / 97 | pass = ep2020 v2 |
| fl_r30_3cam | 0 / 100 | only complete Phase 6 cell |
| spec_cable_clip | 1 / 97 | pass = ep2020 v2 |
| spec_exp_cable_clip | 0 / 26 | partial |
| sequential | — | never started, 0 records |

### Matrix source — stood down by founder ruling 2026-11-24 (701 records)

| checkpoint | passes / counted | note |
|---|---|---|
| fl_r10_3cam | 0 / 100 | complete |
| fl_r20_3cam | 0 / 100 | complete |
| spec_block_stack | 0 / 100 | complete |
| spec_eye_drops | 0 / 99 | +1 excluded (ep1903 v1, solver failure) |
| spec_ring_insert | 0 / 100 | complete |
| spec_tool_insert | 0 / 100 | complete |
| spec_exp_block_stack | 0 / 50 | partial — stood down mid-cell |
| spec_exp_eye_drops | 0 / 51 | partial — stood down mid-cell |
| spec_exp_ring_insert | — | never started, 0 records |
| spec_exp_tool_insert | — | never started, 0 records |

### Separate instrument — never combined with the above

Part B L1 cable cells (v2 controls battery, different instrument): base 0/35, cexp 0/35,
fl_r30_3cam 0/35. Reported separately per the aggregator source policy; they never share a
denominator with Phase 6/matrix cable cells.

## Pass caveats (on record from the Phase 6 verification)

Both ep2020 v2 rollouts end with the gripper still attached to a *different* cable segment —
explicitly permitted by the frozen criterion (its test is retention by the clip, not release).
The state logs' `segs` array sits in a systematically translated frame vs the evaluator's
`objects_final`, so the statelog cannot independently re-derive the seated geometry; the passes
rest on the frozen evaluator's recorded checks from live sim state. Harness hygiene note for
the white paper.

## Independent verification — CERTIFY-WITH-FINDINGS (2026-11-24)

Separate-context verifier recomputed all counts from the raw records (report:
`verification-cable-final-20261124.md` in the matrix workdir). Every cable cell matches the
aggregator v2 snapshot of 2026-11-24 06:32 CT exactly. The single exclusion is
`spec_eye_drops__cable_clip__ep1903_v1.json` (matrix) — solver_failure true
(max_seg_radial_m 1.11), not a build defect. Scene identity: all 1,117 records carry
`suite_version: twin-suite-v1.1`; `param_table_hash()` recomputed live = a4eb3f8c3611894d,
matching the validation stamp and all 120 cable scene XML headers. Minor finding: records
embed no per-record param_hash field — identity rests on suite_version + harness/scene stamps.
Stand-down integrity certified: 0 production processes; latest record mtime 2026-11-24
06:31:10 CT; 0 invalid JSONs.

## Frontier judging layer (second layer, recorded)

Worklist: 113 items — both mechanical passes + seeded 10% sample of counted fails
(seed 20261122; matrix 70, phase6 41). Composition per founder rulings: Muse + Gemini only;
**Grok removed (trust suspended 2026-11-23) — no Grok reads were run or cited.**

- **Muse leg — COMPLETE (113/113, blind, end-weighted frames + zoom crops of the clip
  region):** PASS 2, FAIL 110, INDETERMINATE 1. Both mechanical passes confirmed (cable
  segment in the clip channel at settle, gripper released). Mechanical-vs-Muse agreement
  112/113.
- **Disagreement (1):** ep1959 v4, cexp (Phase 6) — mechanical FAIL vs Muse INDETERMINATE:
  at the final frame the gripper is still pressing a cable segment into the clip; whether the
  clip alone retains it after release cannot be confirmed because the rollout ends with the
  gripper engaged. Not a pass (retention unproven), not a clean fail (segment is at/in the
  clip).
- **Gemini leg — OUTSTANDING (blocked, not skipped):** the first agy call
  (gemini-4-neon-xhigh) returned HTTP 429 RESOURCE_EXHAUSTED (individual quota; reset
  ~11 min). Per the standing rate-limit rule the leg was hard-stopped with no retry. A blind
  batch runner is prepared on the DGX
  (`run_agy_leg.sh`, batches of 8, round-robin gemini-4-neon-xhigh → gemini-4-carbon-high →
  claude-opus-5-5-high, exact model recorded per item) and its verdicts merge into
  `judging_results_final.jsonl`. To be executed by a later run after quota reset.

## Stand-down record (founder ruling 2026-11-24 06:31 CT)

Terminated: retry wrapper (PID 234084), matrix_cable_driver.py (4093198), both in-flight
workers + policy servers, job_guard guard-cable2-a57 (4093174). Verified zero cable-matrix
production processes in two consecutive sweeps; retry log shows no attempt after attempt 57
(2026-11-23 14:20:43 CT). All 701 matrix records preserved unmodified. Recorded as protocol
corrections entry 6 in the matrix workdir's protocol.json.
