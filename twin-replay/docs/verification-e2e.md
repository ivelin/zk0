# Study 001 Twin-Replay — Independent Verification V-E2E (demo-replay certification gate)

Verifier: V-E2E, independent context (did not build the twins). Date: 2026-11-22.
Inputs used: pre-registration + Amendment 1, scene files, `validation/validation_results.json`
(raw data), split.json, pose artifacts, param-changelog.md, calibration artifact.
NOT read: `docs/twin-validation-report.md` (builder narrative/self-assessment).
Method: re-ran the replay harness MYSELF on the DGX (CPU, nice 10, 3 workers, MuJoCo 4.0.1
venv), recorded trajectories only, no checkpoint inference. Raw outputs:
`validation/verify_e2e/rerun_summary.json`, per-variant JSONs and cable top renders there.
DEPENDENCY: re-runs use the builders harness (`code/replay.py`); this verdict verifies
reproducibility and the numbers, and is conditional on V-CODEs separate harness audit.

## 1. Independent re-runs (14 episodes x nominal + 2 sampled variants = 42 runs)

Recorded outcomes (verified against split.json): success for all 80 non-cable episodes,
failure for all 20 cable_clip episodes.

| Episode | v0 mine (builders) | v1 mine (builders) | v2 mine (builders) | Key checks (mine) |
|---|---|---|---|---|
| block_stack ep411 | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | v0 planar err 0.0086 m but face_gap -0.030, not stacked; v1 cube still welded at end (face_gap +0.149) |
| block_stack ep453 | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | planar 0.076/0.128/0.201 m |
| block_stack ep527 | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | v0 planar 0.0147 m, face_gap -0.027, within_footprint=false |
| eye_drops ep1757 (builders: reproduced) | SUCCESS (SUCCESS) | SUCCESS (SUCCESS) | SUCCESS (SUCCESS) | carton in basket, z=0.0205 all variants; nominal planar err 0.0728 m |
| eye_drops ep1778 (builders: reproduced) | SUCCESS (SUCCESS) | SUCCESS (SUCCESS) | FAIL (FAIL) | matches builders agreeing set {0,1,3} on sampled variants |
| eye_drops ep1810 (builders: reproduced) | SUCCESS (SUCCESS) | SUCCESS (SUCCESS) | SUCCESS (SUCCESS) | matches builders agreeing set {0,1,2} |
| ring_insert ep253 (builders: only reproduced ring ep) | SUCCESS (SUCCESS) | SUCCESS (SUCCESS) | FAIL (FAIL) | v0/v1 planar_to_peg 0.0024/0.0017 m engaged; v2 0.1434 m. CAVEAT below |
| ring_insert ep179 | SUCCESS (SUCCESS) | FAIL (FAIL) | FAIL (FAIL) | builders agree frac 0.2 (nominal only) - consistent |
| tool_insert ep622 | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | centre_in_slot=false, axis 43 deg (bar: <=20) |
| tool_insert ep859 | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | axis 77/28/47 deg, centre_in_slot=false |
| tool_insert ep683 | FAIL (FAIL) | FAIL (FAIL) | FAIL (FAIL) | tool ends airborne z=0.219-0.249 (still welded / never placed), axis 42 deg |
| cable_clip ep1905 (recorded FAILURE) | FAIL=agree (agree) | FAIL=agree (agree) | FAIL=agree (agree) | v0 EXPLODED: nearest seg 2.3347 m; v1/v2 sane: 0.1402/0.0836 m, not seated |
| cable_clip ep1939 (recorded FAILURE) | FAIL=agree (agree) | FAIL=agree (agree) | FAIL=agree (agree) | v0 EXPLODED: nearest seg 13.2631 m; v1/v2 sane: 0.1881/0.1307 m, not seated |
| cable_clip ep1896 (recorded FAILURE) | FAIL=agree (agree) | FAIL=agree (agree) | FAIL=agree (agree) | nearest 0.0199/0.1008/0.0133 m, cable_seated=false all |

CAVEAT on ring ep253 nominal: the ring SPAWNS essentially on the peg - initial ring pose
(0.3154,-0.0478) vs peg (0.3136,-0.0413), ~6.7 mm apart, inside the engaged tolerance
(10 mm) - and v0 records NO attach event (3x no_attach_in_excursion) yet ends engaged.
The v0 success is a spawn coincidence, not a demonstrated transport. v1/v3 do show
attach (f140/141) -> release (f174), so the episode is not wholly artefactual, but its
nominal variant proves nothing about replay fidelity.

## 2. Comparison vs validation_results.json

42/42 sampled variant runs match the builders EXACTLY: identical sim_success, identical
checks, identical planar_err_m to all 4 reported decimals (e.g. ep411 v1 0.3269 m;
ep1939 v0 nearest 13.2631 m; ep1778 v2 0.3618 m). The harness is deterministic and the
builders full-set numbers are faithfully reproducible from the scene artifacts.
Discrepancies in outcome classes: NONE in the sample.
Secondary criterion flag (not a discrepancy): even the genuinely reproduced eye_drops
episodes fail the pre-registered 3.0 cm nominal planar criterion by 2-4x (ep1757 7.3 cm,
ep1778 9.3 cm, ep1810 12.4 cm). Reproduction per Amendment 1 rests on the primary
(outcome-class) criterion; the secondary misses should be carried as a stated limitation.

## 3. Cable ruling (ep1905, ep1939)

The nominal variants are NUMERICAL EXPLOSIONS, confirmed in my own re-runs, not faithful
failure reproductions. ep1905 v0: four genuine attach/release cycles (segs 20, 9, 8, 7;
frames 66-423), then the chain leaves the workspace - subsequent excursion min-distances
grow 0.66 -> 1.06 -> 1.10 -> 1.17 m and the cable ends coiled at approx (-1.0,-2.1),
nearest segment 2.3347 m from the block. ep1939 v0: one genuine attach/release
(seg 12, f77-194), then excursion distances jump to 8.4 m; final nearest segment
13.2631 m, max |xy| 9.888 m. For these variants the outcome-class match (not seated =
recorded failure) is VACUOUS: an exploded cable is trivially not seated.
HOWEVER the failure reproduction is genuine in the non-exploded variants: ep1905 v1/v2
and ep1939 v1/v2 (and all ep1896 variants) keep the cable in the workspace, manipulate
it toward the block, and end with the nearest segment 1.3-18.8 cm from the block and
cable_seated_in_clip=false - qualitatively the recorded failed executions.
Full-set census (builders data): 10/100 cable variants have nearest-segment > 0.5 m
(8/100 > 1.0 m), across 7 episodes (1905, 1939, 1959, 1960, 1969, 2042, 2068); 6 of the
20 NOMINAL variants are exploded. The builders counted these as valid, agreeing,
certified variants. Amendment 1 A2 permits exclusion only for build defects including
solver failure - an exploded chain is a solver failure and should have been excluded,
not certified. My recomputation excluding nearest>0.5 m variants: every cable episode
still agrees on 100% of its remaining valid variants (worst case ep2068, 2 valid),
so the episodes outcome agreement survives the correction - but the certified variant
set (claimed 100) does not: at most 90 variants are certifiable.

## 4. Certification ruling (bar: >=80% of medium+ episodes reproduced, >=60% valid variants agreeing; no unexplained systematic failure mode)

- block_stack: NOT CERTIFIED. Builders 0/20 reproduced; my 9/9 sampled variants fail. Corroborated.
- eye_drops_to_basket: NOT CERTIFIED. Builders 4/20 (20%). My re-runs confirm the 4 reproduced episodes are real on the primary criterion (1757, 1778, 1810 sampled) - the rate, not the individual passes, fails the bar.
- ring_insert: NOT CERTIFIED. Builders 1/20 (5%); ep253 corroborated with the spawn caveat in S.1.
- tool_insert: NOT CERTIFIED. Builders 0/17; my 9/9 sampled variants fail (axis/slot/z all off; ep683 tool never even descends).
- cable_clip: NOT CERTIFIED as issued. Outcome agreement is corroborated and survives exclusion of exploded variants (S.3), but (a) 10 exploded variants are counted as certified, violating Amendment 1s exclusion semantics, and (b) chain explosion in 10% of variants - including 30% of nominals - is a systematic failure mode present in the certified set. Path to certification: exclude exploded variants per A2, re-issue the certified set (<=90 variants), and Phase 6 must not use any exploded variant. On the corrected set the outcome bar itself is met.

Phase 6 may not use any task twins on the current certification state.

## 5. Exclusion bookkeeping

VERIFIED. validation_results.json excludes exactly tool_insert ep630, ep871, ep876 for
initial-pose confidence=low. Loading the pose artifacts directly: all three have
confidence=low with per-object confidences {tool: low} (included ep622 is medium for
contrast), and docs/pose-estimation.md S.6 names exactly these three (tool occluded by
arm/wrist plate in top_first) as the no-pose/low-confidence set. Denominator 17 for
tool_insert is correct.

## 6. Calibration legitimacy (affine FK map; gripper-channel inversion)

Both corrections read as legitimate calibration, not result-tuning. The affine map is
fitted on data independent of validation outcomes - 93 grasp/place dwell tie points
pairing FK of the recorded gt against Builder B detection pixels - is a single global
transform stored as a versioned artifact (scenes/calibration_fk_affine.json, fk-affine-v2,
matrix + provenance), and its evidence is a tie-point residual improvement (median
12.8 cm -> 1.4 cm, 83% < 3 cm) reported on the tie points themselves, not on outcome
agreement; without it no grasp could fire at all, so it enables the test rather than
bending its result. The gripper inversion likewise rests on trajectory geometry, not
outcomes: in ep411 the arm transits cube->stack during the traces low phase (frames
~117-227), which is only a hold under the inverted reading, and my own ep411 re-run
attaches at f116 and releases at f231 at the stack under that reading; thresholds
(50%/30%) were left unchanged and the change is global. Two cautions keep this short of
unqualified endorsement: later changelog items (grasp device v3, ring peg variant
jitter #6) were iterated against validation-run failure taxonomies - global and logged
with full-set re-runs, as pre-registration rule 2 requires, but the validation set
thereby doubled as the tuning set, with no held-out confirmation; and peg jitter in
particular is outcome-adjacent (it moved ring 0/20 -> 1/20 between runs 2 and 3, per
validation_results_run1/run2.json and the final set). Neither caution changes any
ruling above: even granting every correction, four tasks fail the bar by wide margins.

## Verdict summary

Reproducibility: CONFIRMED (42/42 exact). Exclusions: CONFIRMED.
Certification: block_stack NOT CERTIFIED; eye_drops_to_basket NOT CERTIFIED;
ring_insert NOT CERTIFIED; tool_insert NOT CERTIFIED; cable_clip NOT CERTIFIED as
issued (correctable by excluding exploded variants and re-issuing the certified set).
