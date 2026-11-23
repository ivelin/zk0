# Study 001 Twin-Replay — Independent Verification V-E2E, twin-suite v1.1 FINAL

Verifier: V-E2E, fresh independent context (did not build, fix, or run any prior
v1.1 pass). All claims below were re-derived in this session on the DGX:
my own replay invocation, my own arithmetic on the raw results file, live
hash recomputation. Context reports were treated as claims, not evidence.
CPU-only, `nice -n 10`, 4 workers, DGX training undisturbed.
Date: 2026-11-22 (~19:47–20:05 CST, DGX clock).
Frozen state under audit: param_hash `a4eb3f8c3611894d`;
definitive results `validation/validation_results_v11.json` (RUN3).

## Certification rulings (summary)

| Task | Ruling | Reproduced | Bar (≥80%) |
|---|---|---|---|
| cable_clip | **CERTIFIED** | 20/20 | met (100%) |
| block_stack | **NOT CERTIFIED** | 0/20 | missed (0%) |
| eye_drops_to_basket | **NOT CERTIFIED** | 3/20 | missed (15%) |
| ring_insert | **NOT CERTIFIED** | 0/20 | missed (0%) |
| tool_insert | **NOT CERTIFIED** | 0/16 | missed (0%) |

Phase 6 may use the cable_clip certified variant set only (all 100 variants;
see Check 2). No other task's twins may be used in Phase 6.

---

## Check 1 — Independent re-run: PASS, 160/160 exact match

Method: my own driver (`validation/verify_e2e/ve2e_driver.py`) importing
`code/replay.py` and calling `run_episode()` directly on the frozen on-disk
scene XMLs — NOT `run_validation.py`, NOT the recorded outputs. 160 variant
jobs: ALL 20 cable_clip episodes × all 5 variants (full certification set,
no sampling), plus 3 episodes × all 5 variants from each other task:
block_stack ep411/453/557; eye_drops_to_basket ep1757 (a reproduced episode)
/1609/1821; ring_insert ep253/7/48; tool_insert ep784 (FP-G repaired pose),
ep697 (carries the v3 build defect), ep622.
Comparison (`validation/verify_e2e/ve2e_compare.py` vs
`validation_results_v11.json` variants_detail): primary outcome
(`sim_success`), `secondary_pass`, `solver_failure` flag, `build_defect`
string, `planar_err_m`, EVERY field of the per-task `checks` dict, and EVERY
`objects_final` coordinate — floats compared at 4-decimal rounding.

| Task | Jobs | Full exact match | Primary | Secondary | Solver flags | Final-state numbers |
|---|---|---|---|---|---|---|
| cable_clip | 100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |
| block_stack | 15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| eye_drops_to_basket | 15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| ring_insert | 15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| tool_insert | 15 | 15/15 | 15/15 | 15/15 | 15/15 | 15/15 |

Exact-match rate: **100% per task, 160/160 overall.** Zero errors in my run.
The recorded RUN3 results are exactly reproducible from the frozen scenes
by an independent invocation.

## Check 2 — Cable explosion question: RESOLVED

- My own cable re-run at the frozen timestep (`code/twin_params.py`:
  `TIMESTEP = 0.0005`, 2000 Hz, the v1.1 global fix, changelog #14):
  **0/100 solver failures.** Every variant's `solver_failure` flag is False,
  and the underlying final states are physical (max segment radial distances
  in the recorded checks are all sub-metre; my re-run reproduces those
  numbers exactly, Check 1).
- v1 obstacle, as established by the v1 V-E2E report: 10/100 cable variants
  were numerical explosions yet were counted in the certified set —
  cable was NOT CERTIFIED as issued, certifiable set at most 90.
- v1.1 disposition: the global 2000 Hz timestep fix removes the instability
  (0/480 solver failures across the whole RUN3 validation; 0/100 in my cable
  re-run), and the standing detection (`replay.py` cable branch: non-finite
  position, radial > `CABLE_MAX_RADIAL_M` = 1.0 m, or z outside
  [−0.05, 0.50] m ⇒ `solver_failure` ⇒ Amendment 1 A2 exclusion in
  `run_validation.py`) remains armed, so any recurrence would be excluded,
  not certified.
- **The certified variant set is the full 100** (valid 100, agreeing 100,
  certified 100 in the results file; all 100 independently re-verified here).
  The v1 obstacle is resolved — not by excluding the exploded variants, but
  by eliminating the explosions at the frozen timestep with detection
  standing guard.

## Check 3 — Denominator audit: PASS, exactly 96 (20/20/20/16/20)

- Results file: 96 episodes — block 20, eye 20, ring 20, tool 16, cable 20;
  exclusion list exactly ep630, ep779, ep871, ep876 (all tool_insert).
- I re-derived each exclusion from the raw pose files: all four carry
  episode confidence `low` (per-object `{tool: low}`), failing the
  registered medium-or-better inclusion rule (pre-registration honesty
  rule 3 + certification bar scope; `CONF_OK = {medium, high}` in
  `run_validation.py`). ep779's relabel to low is the FP-H evidence-based
  correction ratified by Amendment 3 / changelog MF-3 — a bookkeeping
  application of the registered rule, not a criteria change.
- Completeness: the 96 included episodes are exactly the split set minus
  those 4; no other split episode is missing, none is duplicated.
- Recorded outcome classes vs `split.json` (the ground-truth record the
  runner reads): **zero mismatches** across all 96 episodes. Split classes:
  block success ×20, eye success ×20, ring success ×20, tool success ×20
  (×16 in the validated denominator), cable FAILURE ×20 — the cable
  references are failed human executions and reproducing the failure is
  agreement (pre-registration, cable_clip criterion). The results file's
  `recorded_success` fields match these classes in every case.

## Check 4 — Bar arithmetic: PASS, headline table confirmed

I recomputed reproduction from the per-variant records in
`validation_results_v11.json` alone (valid = no build defect, no solver
failure, no error; episode reproduced iff ≥60% of valid variants agree —
Amendment 1; task bar ≥80% — pre-registration). My recomputation asserts
against every episode's recorded `reproduced` flag and
`variant_agreement_frac`: **0 discrepancies over 96 episodes.**

| Task | My recomputation | Headline (RUN3) | Match |
|---|---|---|---|
| block_stack | 0/20 | 0/20 | ✔ |
| eye_drops_to_basket | 3/20 | 3/20 | ✔ |
| ring_insert | 0/20 | 0/20 | ✔ |
| tool_insert | 0/16 | 0/16 | ✔ |
| cable_clip | 20/20 | 20/20 | ✔ |

Valid variants 478/480; the only 2 invalid are pre-settle build defects,
independently reproduced in my sample (tool ep697 v3 matched exactly):
eye ep1808 v4 (carton displaced 0.091 m), tool ep697 v3 (tool displaced
0.035 m). Total solver failures in the file: 0.

## Check 5 — Certification rulings (reason chains to the pre-registration)

Certification clause (pre-registration): a task's twins are CERTIFIED iff,
over medium-or-better-confidence episodes, outcome agreement ≥80% AND no
unexplained systematic failure mode; the independent verifier rules on
whether the pre-registration was honoured. Amendment 1 A2: the certified
variant set = variants that reproduced their demo outcome; episodes below
60% variant agreement are excluded from Phase 6 entirely. Amendment 2:
only certified scenes/variants may be published.

### cable_clip — CERTIFIED
- Bar met: 20/20 episodes reproduced (100% ≥ 80%), mean variant agreement
  1.00, all 100 variants agreeing (Checks 1, 4).
- Failure modes explained: the recorded outcome for all 20 episodes is
  FAILURE (failed human executions, split.json), and the twin reproduces
  that failure physically — the cable is manipulated toward the clip block
  but never seats (diagnosed in the validation report; in v1.1 there is no
  numerical-explosion contamination left to discount, Check 2). No
  unexplained systematic failure mode remains: the one systematic mode v1
  had (chain explosion, 10/100) is eliminated and detector-guarded.
- Certified set well-defined: valid = agreeing = certified = 100 variants,
  enumerated per episode in the results file; this is exactly the set
  Amendment 2 permits to be published and Phase 6 to use.
- Verification classes all closed: code CERTIFIED-WITH-FINDINGS
  (V-Code Delta; residuals D-01 and D-03 closed by micro-fixes MF-1/MF-2 —
  fixtures.json now hashed, ep784 provenance disambiguated — and D-02 an
  inert scope note on external vendored inputs, none affecting any scene
  or score); poses CERTIFIED-WITH-FINDINGS (findings F1/F2 concern
  tool_insert ep784/ep779 only — ep779 is now OUT of the denominator per
  Amendment 3, ep784 sits in a non-certified task; no cable pose finding);
  objects CERTIFIED-WITH-FINDINGS with cable content PROVISIONAL pending
  this check — **provisionality resolved here**: my full 100-variant cable
  re-run reproduces every cable check quantity (seated flag, nearest
  segment-to-block distance, max segment radial) and every final segment
  position to 4 decimals, exercising the built clip geometry and the
  sim-body channel scoring end-to-end (Check 1). Cable content stands
  verified, no longer provisional.

### block_stack — NOT CERTIFIED
0/20 reproduced (0% vs 80% bar): pose-σ structural limit — placement error
≈ √2 × pose σ ≈ 3.5–5 cm against the 1.1 cm inset tolerance; the recorded
closed-loop demos cannot be reproduced by open-loop replay at this pose
quality (diagnosed, validation report).

### eye_drops_to_basket — NOT CERTIFIED
3/20 reproduced (15% vs 80% bar): same pose-σ structural limit — carton
placement error of the same magnitude against basket-interior tolerances;
the 3 reproduced episodes (incl. ep1757, independently re-verified in
Check 1) are real but the rate fails the bar decisively.

### ring_insert — NOT CERTIFIED
0/20 reproduced (0% vs 80% bar): pose-σ structural limit — placement-error
budget ≈ 3.5 cm against a ≲1–2 cm peg-engagement tolerance (ep253,
reproduced in v1, no longer reproduces under the v1.1 frozen set; my
re-run of ep253 matches the recorded failure exactly).

### tool_insert — NOT CERTIFIED
0/16 reproduced (0% vs 80% bar): pose-σ structural limit compounded by
pose errors — tools land far outside the slot/3.0 cm secondary tolerance
(median nominal secondary error 18.4 cm in the validation diagnosis);
independently corroborated on ep784/ep697/ep622 (Check 1).

## Check 6 — Freeze integrity: PASS

- Live `param_table_hash()` recomputed by importing the frozen code in
  this session: **`a4eb3f8c3611894d`** — equals the frozen value.
- `validation_results_v11.json` stamp: `a4eb3f8c3611894d` — matches
  (file SHA-256 `4b526d529d5bdb5e6093e0ffbbffcabe86d697e4fce0a5b543ca5ee122df5cf9`,
  the RUN3 definitive file).
- v1 results file `validation/validation_results.json` SHA-256:
  `396378dca6e545dc4243540e188b6363fdf88084ba70c6dcbfc66e4634d7c5b5` —
  byte-identical to the registered v1 hash. Untouched.

## Verdict

Twin-suite v1.1 FINAL is verified end-to-end: the frozen suite reproduces
its definitive results exactly under independent re-execution (160/160
variants, every field), the denominator and bar arithmetic are correct,
the freeze is intact, and the v1 cable certification defect is cured.
**cable_clip is CERTIFIED (certified variant set = the full 100 variants);
block_stack, eye_drops_to_basket, ring_insert and tool_insert are NOT
CERTIFIED** — each blocked by reproduction far below the 80% bar under the
diagnosed pose-σ structural limit. Publication under Amendment 2 may
proceed for the cable_clip certified set only.
