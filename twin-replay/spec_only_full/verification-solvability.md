# Independent Certification — Spec-Only FULL Gate Suite, Solution Trajectory Artifact Class

**Suite:** zk0 twin-suite v2, Amendment 3 — spec-only (Tier C) scenes, nominal variant k=0
**Claim under test:** Route A solvability runs (oracle-guided frozen reference controllers) produced executed passes (frozen mechanical criteria, `sim_success=true`) for exactly 28 scenes: block_stack spec02/03/08/10/13 (5), cable_clip spec00–spec19 (all 20), eye_drops_to_basket spec16 (1), tool_insert spec00/spec06 (2), ring_insert none (0).
**Verifier:** independent subagent, separate context, 2026-11-23.
**Overall verdict: CERTIFIED — 28/28 reproduced.**

## Method (independence)

- **Runner:** the builder's driver `run_reference_spec.py` was reused as permitted (frozen controller code from the desktop port: `work/reference/` + `work/code/` + `work/harness/`, frozen `v1replay.evaluate` criteria via the controllers' own record path). The driver was invoked fresh for every scene by the verifier's own batch script; no builder result file was used as an input.
- **Scenes:** staged by the verifier from the CANONICAL scene files (`work/spec_only_full/scenes/`) into the verifier's own workdir on the Pop!_OS desktop, `~/muse-work/zk0-verify-v2-20261123/staged/`. Only the arm-include path was rewritten (DGX path `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/scenes-v2/arm/so101_new_calib_padded.xml` → desktop copy `~/muse-work/zk0-spec-only-20261123/work/scenes-v2/arm/so101_new_calib_padded.xml`). Verified mechanically for every staged file (31 scenes): each staged file differs from its canonical source in **exactly 1 line** (the include line); geometry/physics otherwise identical (`stage_diff_report.json` in the verifier workdir).
- **Canonical pass selection:** read from the builder's records in `~/muse-work/zk0-spec-only-20261123/results/`, preferring rev2.1, then frozen, then rev3, among versions with a passing record. Note: for 5 scenes the most-preferred version (rev2.1) had a FAIL record and the canonical pass is the frozen controller — block_stack spec02, spec08, spec10; tool_insert spec00 (rev2.1 FAIL, frozen PASS). Conversely block_stack spec03, tool_insert spec06 and eye_drops_to_basket spec16 pass only under rev2.1. No rev3 records exist for any claimed scene (rev3 is ring_insert-only and ring has no claimed passes).
- **Environment:** `PYTHONPATH=~/muse-work/zk0-spec-only-20261123/pylibs` (MuJoCo 4.0.1, numpy 2.5.5), desktop venv python, `SPEC_MANIFEST` = full suite manifest, `SPEC_STAGED` = verifier's staged dir, `SPEC_VARIANT=0`.
- **Discipline:** single sequential batch (≤1 concurrent run, under the ≤2 cap), `nice -n 10`, run under the guard: `job_guard.sh ~/muse-work/zk0-verify-v2-20261123/rundir verify-v2-solvability active` with `JOB_GUARD_MIRROR_TARGET=~/muse-work/zk0-spec-only-20261123/telemetry-mirror`. Builder's dirs accessed read-only; builder's processes untouched.

## Per-scene results

"Key checks comparison" compares the full frozen `checks` dict of the verifier's re-execution record against the builder's passing record for the same scene/controller.

| Scene | Builder's passing controller | Re-execution verdict | sim_success | Key checks |
|---|---|---|---|---|
| block_stack__spec02 | frozen | PASS | true | match (identical) |
| block_stack__spec03 | rev2.1 | PASS | true | match (identical) |
| block_stack__spec08 | frozen | PASS | true | match (identical) |
| block_stack__spec10 | frozen | PASS | true | match (identical) |
| block_stack__spec13 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec00 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec01 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec02 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec03 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec04 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec05 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec06 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec07 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec08 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec09 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec10 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec11 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec12 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec13 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec14 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec15 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec16 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec17 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec18 | rev2.1 | PASS | true | match (identical) |
| cable_clip__spec19 | rev2.1 | PASS | true | match (identical) |
| eye_drops_to_basket__spec16 | rev2.1 | PASS | true | match (identical) |
| tool_insert__spec00 | frozen | PASS | true | match (identical) |
| tool_insert__spec06 | rev2.1 | PASS | true | match (identical) |

**Reproduced: 28/28.** Every re-execution not only reproduced `sim_success=true` but produced a `checks` dict byte-identical in content to the builder's record — the passes are deterministic replays, not borderline re-rolls.

## Failure spot-checks (false-negative check)

Three scenes the builder recorded as FAIL (rev2.1) were re-executed with the same controller:

| Scene | Builder record | Re-execution verdict | sim_success | Key checks |
|---|---|---|---|---|
| block_stack__spec00 (rev2.1) | FAIL | FAIL | false | match (identical) |
| ring_insert__spec00 (rev2.1) | FAIL | FAIL | false | match (identical) |
| eye_drops_to_basket__spec00 (rev2.1) | FAIL | FAIL | false | match (identical) |

**Spot-check outcome: 3/3 confirmed failures** — no false negatives; the FAIL side of the builder's register reproduces exactly as well.

## Discrepancies

None. No scene failed to reproduce; no checks mismatch on any of the 31 re-executions.

One observation for the register (not a discrepancy): the passing controller is version-dependent for 7 of the 28 scenes — 4 pass only under the frozen controller (block_stack spec02/spec08/spec10, tool_insert spec00), 3 pass only under rev2.1 (block_stack spec03, tool_insert spec06, eye_drops_to_basket spec16), and block_stack spec13 passes under both. Certification above is per-scene against that scene's canonical passing version, per the brief's rule.

## Evidence locations

- Verifier workdir (Pop!_OS desktop, `ivelin@100.73.62.79`): `~/muse-work/zk0-verify-v2-20261123/` — `results/` (31 re-execution records), `summary.json`, `stage_diff_report.json`, `verify_batch.py`, guard telemetry in `rundir/guard/`.
- Machine-readable certification: `certification-v2.json` (alongside this report).

## Verdict

**CERTIFIED — 28/28 reproduced** (plus 3/3 failure spot-checks confirmed). The 28 claimed solution trajectories are independently certified: each scene's canonical pass re-executes to `sim_success=true` under the frozen mechanical criteria, from canonical scene files, in a separate context.
