# Certified-Solution Harvest — Definitive Inventory (2026-11-23, overnight)

Analysis only: no controllers were run and nothing new was certified in this harvest. Every already-certified sim solution (55 spec-only + 20 twins = 75) is graded for cleanliness from its canonical execution record and inventoried with its validation-leg status.

**Generator family (all solutions):** Muse (Muse Wonder). **Valid validation legs:** different-family only — Gemini (Google) via agy, GPT-OSS 120B via agy, Claude (Anthropic) via agy. **Grok reads never count (suspended).**

**Grading rule:** CLEAN = no failed grasp attempts/retries and no failed moves before task completion in the canonical record log (IK pos-only fallbacks while holding the object, and post-completion retreat IK failures, are disclosed in the evidence counts and do not dirty the grade — validator-pass precedent, block spec02/spec08); RECOVERED = failed grasp attempts/retries occurred but the task completed; INCIDENT = completed with glitches lacking a distinct recovery. Where the 2026-11-23 validator-pass report already graded a solution, its grade is kept.

## Totals

| Task | Certified | CLEAN | RECOVERED | INCIDENT | Fully validated (2/2 legs) | Legs owed |
|---|---|---|---|---|---|---|
| block_stack | 21 | 19 | 2 | 0 | 0 | 37 |
| cable_clip | 27 | 27 | 0 | 0 | 0 | 49 |
| eye_drops_to_basket | 19 | 6 | 13 | 0 | 0 | 33 |
| tool_insert | 8 | 4 | 4 | 0 | 0 | 11 |
| **TOTAL** | **75** | **56** | **19** | **0** | **0** | **130** |

**Ring: 0 certified solutions in either source** (spec-only register 20/20 UNPROVEN; twins 0/8) — no ring rows exist to inventory.

**Legs:** 20 solutions carry 1 completed leg (GPT-OSS 120B via agy, from the validator pass: 16 spec-only scenes + 4 twin anchors). Every other solution has 0 legs. **No solution is fully validated (0/75 at 2/2 legs); 130 legs are owed in total.**


## block_stack (21 certified)

| Scene | Source | Canonical controller | k | Nominal / within-envelope | Grade | Evidence (failed grasps / retries / failed moves) | Legs done | Legs owed |
|---|---|---|---|---|---|---|---|---|
| block_stack__spec01 | spec-only | frozen | 7 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['lift', 'release:retreat'] | — | 2 |
| block_stack__spec02 | spec-only | frozen | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['lift', 'release:retreat'] | GPT-OSS 120B via agy | 1 |
| block_stack__spec03 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=1, retries=0, max_disp=6.7mm | — | 2 |
| block_stack__spec07 | spec-only | frozen | 2 | within-envelope | CLEAN | grasp_events=1, retries=0 | — | 2 |
| block_stack__spec08 | spec-only | frozen | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['lift', 'release:retreat'] | GPT-OSS 120B via agy | 1 |
| block_stack__spec09 | spec-only | frozen | 3 | within-envelope | RECOVERED | grasp_events=2, retries=1, failed=['release:retreat'] | — | 2 |
| block_stack__spec10 | spec-only | frozen | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | GPT-OSS 120B via agy | 1 |
| block_stack__spec11 | spec-only | frozen | 1 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['servo:lift', 'servo:lift', 'release:retreat'] | — | 2 |
| block_stack__spec13 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'], max_disp=1.7mm | GPT-OSS 120B via agy | 1 |
| block_stack__spec14 | spec-only | frozen | 1 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |
| block_stack__spec16 | spec-only | frozen | 1 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |
| block_stack__spec17 | spec-only | frozen | 4 | within-envelope | CLEAN | grasp_events=1, retries=0 | — | 2 |
| block_stack__spec18 | spec-only | frozen | 1 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |
| block_stack__ep411 | twin | rev2.1 (§13 union register; PASS record also in results2-canonical/rev2) | 0 | nominal | RECOVERED | grasp_events=2, retries=1, failed=['grasp:above x3', 'lift:step2', 'lift:pos-only'], max_disp=28.2mm | GPT-OSS 120B via agy | 1 |
| block_stack__ep453 | twin | frozen + rev2.1 (both versions, §13) | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |
| block_stack__ep489 | twin | frozen + rev2.1 (both versions, §13) | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |
| block_stack__ep498 | twin | rev2.1 (§13 union register; PASS record also in results2-canonical/rev2) | 0 | nominal | CLEAN | grasp_events=1, retries=0 | — | 2 |
| block_stack__ep499 | twin | frozen + rev2.1 (PASS records in results-env-frozen and results-env-rev21) | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |
| block_stack__ep527 | twin | frozen ONLY (rev2.1 env record is not a PASS) | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |
| block_stack__ep546 | twin | frozen ONLY (rev2.1 env record is not a PASS) | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |
| block_stack__ep549 | twin | frozen + rev2.1 (PASS records in both env batches) | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'] | — | 2 |

- **block_stack__spec02**: validator-pass graded CLEAN (one IK lift fallback while holding cube, disclosed); GPT-OSS CLEAN/GOOD=YES

- **block_stack__spec08**: validator-pass graded CLEAN (IK lift fallback disclosed); GPT-OSS CLEAN/GOOD=YES

- **block_stack__spec10**: validator-pass graded CLEAN; GPT-OSS CLEAN/GOOD=YES

- **block_stack__spec13**: validator-pass graded CLEAN; GPT-OSS CLEAN/GOOD=YES

- **block_stack__ep411**: validator-pass anchor grade RECOVERED (three failed grasp-approach moves before successful grasp); GPT-OSS 120B graded CLEAN — DISAGREEMENT ON RECORD; GPT-OSS leg counted as done per brief (it graded the scene), grade kept is the report's RECOVERED

- **block_stack__ep527**: Single-version pass — see inconsistencies (brief expected 'both versions' for non-§13 scenes).

- **block_stack__ep546**: Single-version pass — see inconsistencies.

## eye_drops_to_basket (19 certified)

| Scene | Source | Canonical controller | k | Nominal / within-envelope | Grade | Evidence (failed grasps / retries / failed moves) | Legs done | Legs owed |
|---|---|---|---|---|---|---|---|---|
| eye_drops_to_basket__spec00 | spec-only | rev2.1 | 1 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['lift:step1', 'release:retreat'], max_disp=6.5mm | GPT-OSS 120B via agy | 1 |
| eye_drops_to_basket__spec01 | spec-only | rev2.1 | 1 | within-envelope | CLEAN | grasp_events=4, retries=3, failed=['grasp-retry:retreat', 'lift:step1', 'release:retreat'], max_disp=20.1mm | GPT-OSS 120B via agy | 1 |
| eye_drops_to_basket__spec02 | spec-only | rev2.1 | 1 | within-envelope | RECOVERED | grasp_events=3, retries=2, failed=['grasp-retry:retreat', 'lift:step1'], max_disp=6.5mm | — | 2 |
| eye_drops_to_basket__spec03 | spec-only | rev2.1 | 3 | within-envelope | RECOVERED | grasp_events=4, retries=3, failed=['grasp-retry:retreat', 'lift:step1', 'release:retreat'], max_disp=10.6mm | — | 2 |
| eye_drops_to_basket__spec06 | spec-only | rev2.1 | 3 | within-envelope | RECOVERED | grasp_events=5, retries=4, failed=['grasp-retry:retreat', 'grasp-retry:retreat', 'grasp-retry:retreat', 'release:retreat'], max_disp=17.3mm | — | 2 |
| eye_drops_to_basket__spec07 | spec-only | rev2.1 | 4 | within-envelope | RECOVERED | grasp_events=5, retries=4, failed=['grasp-retry:retreat', 'grasp-retry:retreat', 'release:retreat'], max_disp=16.1mm | — | 2 |
| eye_drops_to_basket__spec08 | spec-only | rev2.1 | 6 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['lift:step1', 'release:retreat'], max_disp=15.5mm | — | 2 |
| eye_drops_to_basket__spec09 | spec-only | rev2.1 | 4 | within-envelope | RECOVERED | grasp_events=5, retries=4, failed=['grasp-retry:retreat', 'grasp-retry:retreat', 'grasp-retry:retreat', 'lift:step3', 'release:retreat'], max_disp=10.3mm | — | 2 |
| eye_drops_to_basket__spec11 | spec-only | rev2.1 | 3 | within-envelope | RECOVERED | grasp_events=5, retries=4, failed=['grasp-retry:retreat', 'grasp-retry:retreat', 'lift:step3', 'release:retreat'], max_disp=15.6mm | — | 2 |
| eye_drops_to_basket__spec12 | spec-only | rev2.1 | 6 | within-envelope | RECOVERED | grasp_events=19, retries=19, failed=['grasp-retry:retreat', 'grasp-retry:retreat'], max_disp=36.9mm | — | 2 |
| eye_drops_to_basket__spec14 | spec-only | rev2.1 | 4 | within-envelope | RECOVERED | grasp_events=3, retries=2, failed=['grasp-retry:retreat', 'lift:step1'], max_disp=4.0mm | — | 2 |
| eye_drops_to_basket__spec15 | spec-only | rev2.1 | 3 | within-envelope | RECOVERED | grasp_events=16, retries=15, failed=['grasp-retry:retreat', 'grasp-retry:retreat', 'grasp-retry:retreat', 'grasp-retry:retreat', 'grasp-retry:retreat', 'lift:step3', 'release:retreat'], max_disp=13.3mm | — | 2 |
| eye_drops_to_basket__spec16 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=4, retries=3, failed=['grasp-retry:retreat', 'lift:step1', 'release:retreat'], max_disp=9.0mm | GPT-OSS 120B via agy | 1 |
| eye_drops_to_basket__spec17 | spec-only | rev2.1 | 1 | within-envelope | RECOVERED | grasp_events=3, retries=2, failed=['grasp-retry:retreat', 'lift:step1', 'release:retreat'], max_disp=12.1mm | GPT-OSS 120B via agy | 1 |
| eye_drops_to_basket__spec19 | spec-only | rev2.1 | 2 | within-envelope | RECOVERED | grasp_events=18, retries=17, failed=['grasp-retry:retreat', 'grasp-retry:retreat', 'grasp-retry:retreat'], max_disp=21.5mm | — | 2 |
| eye_drops_to_basket__ep1609 | twin | frozen ONLY (rev2.1 env record is not a PASS) | 0 | nominal | RECOVERED | grasp_events=5, retries=4, failed=['release:retreat'] | GPT-OSS 120B via agy | 1 |
| eye_drops_to_basket__ep1630 | twin | frozen + rev2.1 (both versions, §13) | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['lift:step1 IK fallback to pos-only'], max_disp=18.5mm | — | 2 |
| eye_drops_to_basket__ep1746 | twin | frozen + rev2.1 (both versions, §13) | 0 | nominal | CLEAN | grasp_events=1, retries=0, failed=['lift:step1 IK fallback to pos-only'], max_disp=9.1mm | — | 2 |
| eye_drops_to_basket__ep1798 | twin | frozen + rev2.1 (both versions, §13) | 0 | nominal | RECOVERED | grasp_events=2, retries=1, failed=['lift:step1 IK fallback to pos-only'], max_disp=8.8mm | — | 2 |

- **eye_drops_to_basket__spec00**: validator-pass graded CLEAN (canonical rev2.1 k1, single grasp); GPT-OSS CLEAN/GOOD=YES

- **eye_drops_to_basket__spec01**: validator-pass graded CLEAN on SUPERSEDING canonical rev2.1 k2 (single grasp), certified in validator-pass cert2/ — register canonical k1 record itself carries grasp retries (would grade RECOVERED); GPT-OSS CLEAN/GOOD=YES. See inconsistencies.

- **eye_drops_to_basket__spec16**: validator-pass graded CLEAN on SUPERSEDING canonical rev2.1 k4 (single grasp), certified in validator-pass cert2/ — register canonical k0 record itself carries grasp retries (would grade RECOVERED); GPT-OSS CLEAN/GOOD=YES. See inconsistencies.

- **eye_drops_to_basket__spec17**: validator-pass graded RECOVERED (only pass rev2.1 k1 carries grasp retries; 10 attempts, budget exhausted for a clean solution); GPT-OSS concurs GOOD=NO

- **eye_drops_to_basket__ep1609**: validator-pass anchor grade RECOVERED (report: three grasp-retry retreats); GPT-OSS concurs GOOD=NO

- **eye_drops_to_basket__ep1630**: IK lift fallback while holding carton — does not dirty grade (validator-pass precedent).

## tool_insert (8 certified)

| Scene | Source | Canonical controller | k | Nominal / within-envelope | Grade | Evidence (failed grasps / retries / failed moves) | Legs done | Legs owed |
|---|---|---|---|---|---|---|---|---|
| tool_insert__spec00 | spec-only | frozen | 0 | nominal | CLEAN | grasp_events=1, retries=0 | GPT-OSS 120B via agy | 1 |
| tool_insert__spec06 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=3, retries=2, failed=['release:retreat'], max_disp=5.5mm | GPT-OSS 120B via agy | 1 |
| tool_insert__spec08 | spec-only | rev2.1 | 8 | within-envelope | RECOVERED | grasp_events=5, retries=4, failed=['grasp-retry:retreat', 'grasp-retry:retreat', 'lift:step2'], max_disp=2.1mm | — | 2 |
| tool_insert__spec09 | spec-only | rev2.1 | 3 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'], max_disp=0.0mm | GPT-OSS 120B via agy | 1 |
| tool_insert__spec12 | spec-only | rev2.1 | 2 | within-envelope | RECOVERED | grasp_events=2, retries=1, failed=['lift:step2', 'servo:lift', 'release:retreat'], max_disp=9.4mm | — | 2 |
| tool_insert__spec16 | spec-only | rev2.1 | 7 | within-envelope | RECOVERED | grasp_events=3, retries=2, failed=['release:retreat'], max_disp=6.1mm | — | 2 |
| tool_insert__spec19 | spec-only | rev2.1 | 3 | within-envelope | CLEAN | grasp_events=1, retries=0, failed=['release:retreat'], max_disp=8.7mm | GPT-OSS 120B via agy | 1 |
| tool_insert__ep683 | twin | frozen (§13 union register: tool 683 frozen set, refcore.py + run_reference.py) | 0 | nominal | RECOVERED | grasp_events=3, retries=2, failed=['grasp:above x2', 'lift', 'servo:lift', 'release:retreat'] | GPT-OSS 120B via agy | 1 |

- **tool_insert__spec00**: validator-pass graded CLEAN (flawless record: no failed moves); GPT-OSS CLEAN/GOOD=YES

- **tool_insert__spec06**: validator-pass graded CLEAN; GPT-OSS CLEAN/GOOD=YES. NOTE: the canonical record log contains 2 grasp-retry:retreat moves — under the mechanical harvest rule this would grade RECOVERED; validator-pass grade kept per brief. See inconsistencies.

- **tool_insert__spec09**: validator-pass graded CLEAN; GPT-OSS CLEAN/GOOD=YES

- **tool_insert__spec19**: validator-pass graded CLEAN; GPT-OSS CLEAN/GOOD=YES

- **tool_insert__ep683**: validator-pass anchor grade RECOVERED (two grasp retries, failed approaches, failed lift); GPT-OSS concurs GOOD=NO

## cable_clip (27 certified)

| Scene | Source | Canonical controller | k | Nominal / within-envelope | Grade | Evidence (failed grasps / retries / failed moves) | Legs done | Legs owed |
|---|---|---|---|---|---|---|---|---|
| cable_clip__spec00 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | GPT-OSS 120B via agy | 1 |
| cable_clip__spec01 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | GPT-OSS 120B via agy | 1 |
| cable_clip__spec02 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | GPT-OSS 120B via agy | 1 |
| cable_clip__spec03 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | GPT-OSS 120B via agy | 1 |
| cable_clip__spec04 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec05 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec06 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec07 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec08 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec09 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec10 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec11 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec12 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec13 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec14 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec15 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec16 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec17 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec18 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__spec19 | spec-only | rev2.1 | 0 | nominal | CLEAN | grasp_events=0, retries=0, failed=['cable:over-channel', 'cable:retreat'] | — | 2 |
| cable_clip__ep1905 | twin | frozen ONLY (rev2.1 env record is not a PASS) | 0 | nominal | CLEAN | grasp_events=0, retries=None, failed=['cable:over-channel IK pos-only fallback', 'cable:retreat'] | — | 2 |
| cable_clip__ep1969 | twin | frozen + rev2.1 (both versions, §13) | 0 | nominal | CLEAN | grasp_events=0, retries=None, failed=['cable:over-channel IK pos-only fallback', 'cable:retreat'] | — | 2 |
| cable_clip__ep2020 | twin | frozen + rev2.1 (both versions, §13) | 0 | nominal | CLEAN | grasp_events=0, retries=None, failed=['cable:over-channel IK pos-only fallback', 'cable:seat x1 failed attempt before successful seat', 'cable:retreat'] | GPT-OSS 120B via agy | 1 |
| cable_clip__ep2046 | twin | frozen ONLY (rev2.1 env record is not a PASS) | 0 | nominal | CLEAN | grasp_events=0, retries=None, failed=['cable:over-channel IK pos-only fallback', 'cable:retreat'] | — | 2 |
| cable_clip__ep2068 | twin | frozen + rev2.1 (PASS records in both env batches) | 0 | nominal | CLEAN | grasp_events=0, retries=None, failed=['cable:over-channel IK pos-only fallback', 'cable:retreat'] | — | 2 |
| cable_clip__ep2085 | twin | frozen + rev2.1 (PASS records in both env batches) | 0 | nominal | CLEAN | grasp_events=0, retries=None, failed=['cable:over-channel IK pos-only fallback', 'cable:seat x1 failed attempt before successful seat', 'cable:retreat'] | — | 2 |
| cable_clip__ep2130 | twin | rev2.1 (§13 union register; PASS record also in results2-canonical/rev2) | 0 | nominal | CLEAN | grasp_events=0, retries=None, failed=['cable:over-channel IK pos-only fallback', 'cable:retreat'] | — | 2 |

- **cable_clip__spec00**: validator-pass graded CLEAN-by-record on the k0 distinct draw (frozen controller, validator-pass workdir); GPT-OSS CLEAN/GOOD=YES. Register canonical is rev2.1 k0 — a different execution of the identical nominal scene file. See inconsistencies.

- **cable_clip__spec01**: validator-pass graded CLEAN-by-record on the k1 distinct draw (frozen controller, validator-pass workdir); GPT-OSS CLEAN/GOOD=YES. Register canonical is rev2.1 k0 nominal. See inconsistencies.

- **cable_clip__spec02**: validator-pass graded CLEAN-by-record on the k2 distinct draw (frozen controller, validator-pass workdir); GPT-OSS CLEAN/GOOD=YES. Register canonical is rev2.1 k0 nominal. See inconsistencies.

- **cable_clip__spec03**: validator-pass graded CLEAN-by-record on the k3 distinct draw (frozen controller, validator-pass workdir); GPT-OSS CLEAN/GOOD=YES. Register canonical is rev2.1 k0 nominal. See inconsistencies.

- **cable_clip__spec04**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec05**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec06**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec07**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec08**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec09**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec10**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec11**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec12**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec13**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec14**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec15**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec16**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec17**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec18**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__spec19**: CLEAN-by-record (validator-pass cable convention): weld-device execution, one cable:over-channel IK pos-only fallback, post-completion cable:retreat IK failure — neither dirties the grade. Weld-device caveat applies (not a natural manipulation trajectory).

- **cable_clip__ep1905**: CLEAN-by-record, weld-device caveat. Single-version pass — see inconsistencies.

- **cable_clip__ep1969**: CLEAN-by-record, weld-device caveat.

- **cable_clip__ep2020**: validator-pass anchor grade CLEAN-by-record; GPT-OSS concurs GOOD=YES. Weld-device caveat.

- **cable_clip__ep2046**: CLEAN-by-record, weld-device caveat. Single-version pass — see inconsistencies.

- **cable_clip__ep2068**: CLEAN-by-record, weld-device caveat.

- **cable_clip__ep2085**: CLEAN-by-record (same cable:seat pattern as validator-graded ep2020), weld-device caveat.

- **cable_clip__ep2130**: CLEAN-by-record, weld-device caveat.

## Inconsistencies flagged (not silently fixed or dropped)

1. eye spec01: register canonical_solution = rev2.1 k1 (record eye_drops_to_basket__spec01_rev21_k1.json), whose log carries grasp retries (mechanical grade RECOVERED). The validator-pass report's canonical is rev2.1 k2 (single grasp, CLEAN), certified in ~/muse-work/zk0-validator-pass-20261123/results/cert2/. The FULL-suite register was never updated to the superseding canonical. Inventory grades the scene CLEAN per the validator-pass report and lists both records.

2. eye spec16: register canonical_solution = rev2.1 k0 nominal (record carries grasp retries; mechanical grade RECOVERED). Validator-pass canonical is rev2.1 k4 (single grasp, CLEAN), certified in the validator-pass workdir. Register not updated. Inventory grades CLEAN per validator-pass report.

3. cable spec00-spec03: register canonical for ALL 20 cable scenes is rev2.1 k0 nominal, and all 20 nominal scene files are byte-identical (validator-pass identity finding, register gap G-C1). The validator-pass GPT-OSS leg graded DISTINCT variant draws executed in the validator-pass workdir (spec00 k0 / spec01 k1 / spec02 k2 / spec03 k3, frozen controller) — not the register canonical executions. The 16 non-graded cable register canonicals inherit CLEAN-by-record grading only; their nominal scenes are the same identical file.

4. tool spec06: validator-pass grade CLEAN (kept) conflicts with the canonical record log, which contains 2 grasp-retry:retreat moves (2 failed grasp attempts before the successful grasp) plus servo final_err ok=False. Under the harvest's mechanical rule this record grades RECOVERED. Flagged, not silently resolved.

5. Twins passing under ONE controller version only (contrary to the brief's 'others both versions' for non-§13 scenes): block ep527, block ep546, eye ep1609, cable ep1905, cable ep2046 — PASS records exist only in results-env-frozen; the corresponding rev2.1 env records are not PASS. Canonical version for these five is frozen only.

6. block ep411 / ep498 and cable ep2130: §13 assigns rev2.1 as the union-register canonical version, but PASS records also exist in results2-canonical (rev2). Inventory follows §13 for the canonical version and records the rev2 PASS as an additional passing version.

7. eye spec12: canonical record phase_notes state 'grasp of carton infeasible in all yaw branches' yet the record verdict is PASS and the scene is certified in the register. Graded RECOVERED from its log (grasp retries present, task completed). The phase-note/verdict tension is flagged, not resolved.

8. No certified scene was found with a missing canonical builder record: all 55 spec-only canonical record files exist on the desktop, and all 20 twin canonical records exist on the DGX. Ring: 0 certified solutions in either source (register: 20/20 UNPROVEN) — no ring rows exist to inventory.


## Sources

- Spec-only register + certification: desktop `~/muse-work/zk0-spec-only-20261123/work/spec_only_full/register.json` + `certification-merged.json`; builder records in `~/muse-work/zk0-spec-only-20261123/results/` (paths per solution in inventory.json).
- Validator-pass grades/legs: `~/workspace/research_notes/zk0-bench003-design-20261119/validator-pass-report-20261123.md`; superseding canonical records in desktop `~/muse-work/zk0-validator-pass-20261123/results/`.
- Twins: DGX `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference/` (results/, results2-canonical/, results3-canonical/, results-env-frozen/, results-env-rev21/); union register in `twin-suite-v2-docs/solvability-v2.md` §13; archive HF `zk0bot/sim-scene-solutions` @ e57de85a.