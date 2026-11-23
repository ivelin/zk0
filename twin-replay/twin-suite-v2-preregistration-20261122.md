# Twin-Suite v2 / Sim-Gate v1 — Pre-registration (2026-11-22)

Written BEFORE any twin-suite v2 build or gate run. **Founder approval:
2026-11-22 21:34 CT** (card: "Approve the flip — pre-register twin-suite
v2"). **Governing document:** `sim-gate-design-memo-20261122.md`
(independently verified CERTIFY-WITH-FIXES, fixes applied; committed
alongside this pre-registration). Background records: twin-suite v1.1
completion report (2026-11-22); BENCH-003 design (2026-11-19); Study 001
final eval table (independently verified 2026-11-22). Registered in the
decisions SSOT as **Decision 62**.

## What this pre-registration does

- **Certification-target flip.** For the four precision tasks
  (block_stack, eye_drops_to_basket, ring_insert, tool_insert),
  demo-replay certification is **RETIRED**. Scenes are no longer
  certified by reproducing recorded demo outcomes. Instead the
  *evaluation instrument* (scenes + harness + uncertainty envelope) is
  certified against the four Stage 1 standards in §3, and checkpoints
  are judged by closed-loop policy rollouts under the gate protocol in
  §4–§5.
- **Kept.** Demo-replay certification remains the standard for
  cable_clip (twin-suite v1.1, certified, published) and for any future
  pose-tolerant outcome class.
- **Closed.** The richer-pose-evidence path (design-memo Option E) is
  closed on the record: the recordings' FK↔image frame-agreement floor
  (~3.3 cm median) caps every estimator fitted to them above task
  tolerance. No further spend on pose accuracy from these recordings
  beyond the two cheap physical measurements in the parameter audit
  (§3.1), which are calibration items, not certification levers.
- **Scope of this build.** Phase 1 (this pre-registration) and Phase 2
  Stage 1 (instrument build + registered controls). **No gate run on
  any non-control checkpoint is authorized by this pre-registration**
  until per-task Stage 1 certification has landed and the Bounds
  Amendment (§7) is frozen. Phase 6 cable_clip evals proceed under
  their separate v1.1 authorization and are not part of this build.

## Freeze rule

This document is **FROZEN once committed** to
`ivelin/zk0 @ study001/fl-substrate-20261121` (folder `twin-replay/`).
Later changes are made only as dated amendment entries appended to the
committed copy, made **before** the results they affect are seen, never
as silent edits. The bounds values in §7 are the single registered
exception: they are set by the Bounds Amendment procedure defined
there, from control-checkpoint dispersion, before any candidate is
gated.

## 1. Suite identity

- **twin-suite v2** = the git commit that adds this pre-registration,
  the v2 parameter audit, the Drake scene ports, the envelope/gate
  harness code, and the v2 scene/variant manifests to the frozen
  twin-suite v1.1 base (param_hash `a4eb3f8c3611894d`, commit `1055f4b`
  lineage). The v2 `param_hash` is computed by the v1.1-standard hasher
  extended over all v2 pipeline code, the v2 parameter table, the
  Drake scene files, and the variant manifests for both engines.
- External inputs are pinned by SHA-256 reference, per the v1.1
  standard (HF-REFERENCE lineage): arm model, split.json, dataset
  `meta/info.json`, eval-records manifest, and the three control
  checkpoint identities (§6) are pinned in the Stage 1 controls
  manifest before any control run.
- cable_clip's Stage 0 instrument remains twin-suite v1.1 at HF pinned
  revision `579ad96b960d1c6ad756b3e08518aa0b04eb60e7`; v2 does not
  modify it.

## 2. Episode set

- The frozen held-out set: **denominator 96** — block_stack 20,
  eye_drops_to_basket 20, ring_insert 20, tool_insert 16, cable_clip
  20. Exclusions (all pre-registered under v1/v1.1, none new):
  tool_insert ep630, ep871, ep876 (low-confidence poses, v1
  pre-registration) and ep779 (Amendment 3). Full episode-id lists are
  Appendix A.
- **Disclosed:** this set is twin-side tuning-exposed (v1/v1.1 global
  iterations used it as the validation set). It is not policy-side
  contaminated. After this freeze, **no twin parameter, scene, or
  criterion may be tuned against gate outcomes on this set.**
  Instrument development and tuning use only the control checkpoints
  (§6) and the development subset (Appendix A; seed 20261124, 4
  episodes per task, drawn from the gate set). Envelope-stability
  machinery is demonstrated on the envelope subset (Appendix A; seed
  20261125, 4 episodes per task, disjoint from the development
  subset).
- BENCH-002's 3 held-out tasks stay sealed and untouched.

## 3. Stage 1 — instrument certification standards (per task)

A precision task's instrument is **certified** when all four standards
hold for that task, each verified by an independent context (builder ≠
verifier, §9). Certification is per-task: a task failing any standard
stays research-grade and out of the gate.

### 3.1 Parameter audit

- Every physics parameter for the task is re-derived or re-sourced
  from the v1 parameter table lineage, with its label printed:
  [MEASURED-EST] / [PROP] / [DEFAULT] (v1 table: 14 / 3 / 13;
  friction and restitution entirely [DEFAULT]).
- The two cheap physical measurements identified in v1.1 — **mat
  dimensions** and **cube material** — are attempted **from records**
  (the ArmnetBench dataset's own documentation and any published setup
  specification). If a value cannot be obtained from records, it is
  recorded as an explicit **open item** in the Stage 1 report with its
  parameter left at its labeled estimate/range; nothing is guessed
  and nothing is measured by proxy and relabeled. These measurements
  move calibration marginally; per the governing memo they **do not
  block certification** and cannot change any certification outcome.
- Any verdict whose envelope stability depends on a [DEFAULT]
  parameter sitting at its corner is flagged as such in every report
  (§8).

### 3.2 Cross-engine agreement (MuJoCo ↔ Drake)

- The task's scenes are ported to **Drake** (the BENCH-003 design's
  named eval gate; MuJoCo remains the primary engine and cross-check
  per SSOT Decision 23's role assignment, inverted here only in that
  both engines' agreement is required, neither one's word alone).
- Variant definitions (sampled initial poses, friction/mass draws)
  are engine-independent inputs, generated once by the frozen v1.1
  machinery and consumed **identically** by both engines.
- On the control checkpoints (§6), policy-outcome agreement between
  engines must fall inside the registered bound (§7). Disagreement
  localizes a twin suspect, not a policy suspect.
- **Time-box:** the Drake port + agreement harness is time-boxed at
  **20 agent-hours**. Exceeding the box triggers a written status
  report and a decision point; it never silently changes scope,
  standards, or this pre-registration.

### 3.3 Envelope stability

- The uncertainty envelope per task: initial pose sampled within the
  measured per-task σ (block 2.5, ring 2.5, tool 3.5, eye_drops
  3.5 cm/axis — v1.1 held-out measurements); friction and mass within
  the parameter table's sourced ranges, labels attached.
- Envelope corners are enumerated in the run manifest: pose at ±σ
  extremes per axis on the manipulated object, friction and mass at
  range corners, and the engine swap (§3.2).
- The standard: gate verdicts on the control checkpoints are
  **unchanged across envelope corners** on the envelope subset. A
  verdict that flips inside the envelope is INDETERMINATE by
  construction (§5), never rounded.

### 3.4 Known-ordering controls

- The gate protocol (§4) is exercised end-to-end on the three control
  checkpoints (§6) before anything else runs. The instrument passes
  this standard for a task only if:
  - the untrained **base checkpoint FAILS every task** (its sealed
    base-floor is 0/100 judged completion), and
  - the ordering **base < FL round-30 < C-exp** holds per the
    registered noise band (§7), where the reference is the Study 001
    judged-completion table (base 0/100, FL 26/100, C-exp 70/100 —
    independently verified).
- **Cross-modality assumption, stated:** the reference ordering comes
  from open-loop offline replay; the gate measures closed-loop sim
  rollouts. A mismatch is evidence against the instrument **only
  after** modality effects are excluded, by this registered
  procedure: (1) confirm base-fails-all still holds; (2) compare
  per-checkpoint failure taxonomies between the offline-replay record
  and the gate rollouts; (3) a checkpoint pair may be excused only if
  a specific modality mechanism is **documented with measurements**
  (e.g. closed-loop compounding divergence, measured as trajectory
  error growth across the rollout, where the open-loop signature
  shows no corresponding defect); (4) if any pair is excused, the
  ordering control **downgrades** for this suite version to
  base-fails-all + cross-engine agreement, and the downgrade and its
  evidence are recorded in the certification report; (5) if no
  mechanism is evidenced, the control **FAILS** and the task is not
  instrument-certified. There is no third outcome and no silent pass.

## 4. Stage 2 — gate protocol (per checkpoint, per task)

- The policy runs **closed-loop** in the twin: it consumes renders
  from the three camera viewpoints matched to the dataset's
  front/top/wrist cameras (as established in the v1.1 scenes), its
  proprioceptive state, and the task language prompt, under the same
  input contract as the Study 001 eval harness (3-camera repaired
  configuration). Policy-visible inputs obey the telemetry contract
  (SSOT Decision 55): rendered frames and arm state only, never
  ground-truth object state.
- The arm is **actuated**, not kinematic: position control toward the
  policy's commanded joint targets with STS3215-honest torque/speed
  limits (Decision 55). Actuator parameters are instrument
  parameters: set on the development subset + controls only, frozen
  in the suite manifest, identical for every checkpoint.
- **Variants:** K=5 per episode, seed lineage 20261122 (frozen v1.1
  variant machinery; regeneration determinism standard inherited).
  Initial states are sampled from the per-task pose distribution
  (§3.3); friction/mass drawn within sourced ranges. Every sampled
  value and every parameter label is printed in the run manifest.
- **Rollout horizon:** the episode's recorded duration (20 fps,
  dataset metadata) + 1.0 s settle; final state read at settle end.
- **Scoring (primary):** the mechanical contact/support criteria as
  implemented and verifier-closed in v1.1 (`contacts_with()`; object
  positions are diagnostics only). The secondary criterion is
  reported, not gated (thresholds per v1.1: 3.0 cm; ring 2.0 cm
  unengaged; cable exempt). Solver-failure detection is on (v1.1
  standard: non-finite state, radial or z out of bounds ⇒ variant
  excluded, counted, and reported — never scored as agreement or
  success).
- **Engine execution:** full episode set in the primary engine
  (MuJoCo); the envelope subset (§2) additionally at all envelope
  corners and in Drake. If second-engine or corner results on the
  subset breach the registered bounds, the run escalates to the full
  set in Drake before any verdict is issued.
- **Second layer (recorded, not primary):** frontier-model pass/fail
  judging of the rollout recordings under the fixed rubric,
  checkpoint identity hidden. Mechanical/judging disagreements are
  surfaced per episode; a disagreement rate above the registered
  bound (§7) makes the task verdict INDETERMINATE. Physically absurd
  successes flagged by the judging layer route to escalation (BENCH-003
  Stage 3 twin replay under harsher physics), not to PASS.

## 5. Verdict semantics (three-valued)

Per checkpoint, per task, over episodes × K=5 variants pooled with
Wilson intervals:

- **PASS:** Wilson **lower** bound of pooled success ≥ **60%** (the
  founder-signed sim bar, applied to the lower bound, not the point
  estimate) **and** the verdict is stable across envelope corners
  **and** no disqualifying sim-only failure class.
- **FAIL:** Wilson **upper** bound < 60%, stable across the envelope.
- **INDETERMINATE:** anything else — the verdict flips inside the
  envelope, cross-engine disagreement on this checkpoint exceeds the
  bound, or the judging disagreement rate exceeds its bound.
  INDETERMINATE escalates (Stage 3 or more variants); it never rounds
  to PASS or FAIL, and it never sends a checkpoint to the arm
  silently.

**Promotion** requires PASS on **every** task, or a named founder
waiver recording the exception. Reporting is per-task first, pooled
second; a pooled average may not hide a dead task (BENCH-003 §10.3).

## 6. Control checkpoints (identity-pinned)

1. **Base:** `lerobot/smolvla_base` at the P1-verified pinned revision
   `5e8d12a6…` (500/500 tensors; Decision 61 base-floor rule).
2. **FL round-30:** the A5-repaired packaged checkpoint
   `outputs/2026-11-21_14-56-47/models/checkpoint_round_30_a5repair`
   (3-camera configuration; weights SHA-256
   `5b1cbf290b8b4beeeb9a77473f0d03c0e3ffce8c038b7aab56bc6795fc7a84b4`,
   identical to the checkpoint that scored 26/100 in the verified
   3-camera re-eval).
3. **C-exp:** the Study 001 centralized expert-only final checkpoint
   (`run/train/centralized` lineage); its exact artifact hash is
   pinned in the Stage 1 controls manifest before any control run.

No other checkpoint may be gated under this pre-registration until
Stage 1 certification and the Bounds Amendment are complete.

## 7. Registered bounds — set from controls, frozen before candidates

The following bounds are **procedures in this freeze**; their values
are set during Stage 1 from the control checkpoints' measured
dispersion across engines and envelope corners, and frozen in a dated
**Bounds Amendment** appended to this pre-registration **before any
non-control checkpoint is gated**. Bounds are measured from controls,
never fitted to candidates:

- cross-engine policy-outcome agreement bound (per task);
- ordering-control noise band (per task);
- mechanical/judging disagreement-rate bound;
- envelope-corner verdict-stability criterion (the subset size and
  corner list of §3.3 are already frozen; the amendment records the
  measured control dispersion they produced).

## 8. Claims discipline

Every gate output carries the label: **"funnel verdict — sim bar
unanchored to real-arm delta (no calibration pairs yet)"**, until
BENCH-003 Stage 4 calibration pairs exist and the bar is re-anchored
to a measured delta.

The gate **would support** (per task, once that task's instrument is
certified): pass/fail funnel verdicts under the registered envelope
(three-valued); relative ranking of checkpoints on the five study
tasks, stable across the envelope and across two engines; the funnel
claim that below-bar checkpoints do not earn arm time on the
bar-clearing track; sim failure-taxonomy claims; and, for cable_clip,
the v1.1 demo-replay-certified scene claims.

The gate **would NOT support**: any absolute real-world success rate;
per-episode reproduction claims for the four precision tasks;
novel-task generalization claims; contact-force fidelity claims
beyond the parameter audit; claims about other embodiments, setups,
or task families; and any claim that a PASS guarantees the real-arm
bar (≥50% over ≥10 trials/task) — PASS earns the arm session, it does
not pre-judge it.

## 9. Blinding and independent verification

- Checkpoint identity is hidden from frontier judges and from any
  human judging sample; mechanical scoring is identity-blind by
  construction. The founder's blinded spot-check set is drawn at
  random from judged rollouts, checkpoint hidden (Phase 7 pattern).
- **Builder ≠ verifier, per artifact class**, in separate contexts,
  raw artifacts only: (a) parameter audit; (b) Drake port + agreement
  harness code; (c) envelope/gate harness code; (d) end-to-end —
  re-runs of all control runs on the envelope subset plus a
  stratified sample of nominal control runs, exact-match standard
  inherited from v1.1 (160/160 precedent). Certification verdicts are
  the verifiers', never the builders'. A failed certification is a
  valid outcome and is reported as failed with evidence; **the
  standard is never tuned to pass.**

## 10. Reporting

Per task first, pooled second. Every number with its Wilson interval,
variant count, solver-exclusion count, envelope-stability result,
cross-engine agreement result, the parameter labels its stability
depends on, and the §8 unanchored-bar label.

## 11. Change control and versioning

Inherited from v1.1: **patch** = bookkeeping-only corrections;
**minor** = parameter/template/pose/criteria-implementation changes —
requires full re-validation of affected tasks + fresh verification;
**major** = episode-set or criterion changes, or a different
instrument standard. Any change to scenes, criteria, envelope, or
bars after this freeze is a new suite version under these rules; old
versions stay runnable; certification status is part of the citation
("twin-suite v2 (instrument-certified, per-task status as reported)").
The **monthly realism re-audit** (founder refinement 2026-11-22)
applies to v2 unchanged: tie points re-derived, nominal variants
re-run, outcome flips investigated and logged, unrealistic scenes
fixed in a new version or retired, never edited in place.

## 12. Compute discipline

All sim work is CPU-bound and runs under the job guard via
`launch_dgx_job.sh` (70% whole-box ceiling, explicit caps, telemetry
mirrored off-box). DGX training in progress is never disturbed; v1.1
ran CPU-only alongside training under the same rule. No physical-arm
time. No new training runs.

## Appendix A — episode sets (frozen)

Gate set (denominator 96), from the held-out split (seed 20261122),
minus the §2 exclusions:

- block_stack (20): 411, 453, 476, 479, 489, 498, 499, 513, 514, 525,
  526, 527, 530, 531, 539, 545, 546, 548, 549, 557
- eye_drops_to_basket (20): 1609, 1618, 1630, 1639, 1643, 1724, 1741,
  1746, 1747, 1757, 1760, 1778, 1780, 1783, 1798, 1808, 1810, 1811,
  1814, 1821
- ring_insert (20): 7, 11, 30, 31, 33, 48, 50, 160, 165, 167, 168,
  177, 179, 188, 195, 198, 227, 240, 242, 253
- tool_insert (16): 622, 629, 683, 684, 697, 784, 824, 830, 852, 859,
  883, 885, 888, 892, 902, 903
- cable_clip (20): 1896, 1903, 1905, 1939, 1959, 1960, 1968, 1969,
  2020, 2032, 2042, 2046, 2059, 2068, 2085, 2092, 2101, 2125, 2130,
  2146

Development subset (seed 20261124, 4/task — the only gate-set
episodes instrument tuning may touch, with the control checkpoints):

- block_stack: 411, 453, 489, 498 · eye_drops_to_basket: 1630, 1746,
  1783, 1798 · ring_insert: 7, 33, 167, 195 · tool_insert: 683, 697,
  852, 883 · cable_clip: 1969, 2020, 2125, 2130

Envelope subset (seed 20261125, 4/task, disjoint from development):

- block_stack: 499, 527, 546, 549 · eye_drops_to_basket: 1609, 1760,
  1778, 1821 · ring_insert: 30, 50, 179, 198 · tool_insert: 684, 830,
  885, 888 · cable_clip: 1905, 2046, 2068, 2085

## Amendment 1 (2026-11-22, pre-result — verification fixes)

Appended in response to the independent pre-registration verification
(CERTIFIED-WITH-FINDINGS, `twin-suite-v2-docs/verification-v2-preregistration.md`).
No frozen text above is altered; this amendment clarifies and corrects
labels only, before any build result exists.

- **A1.1 (F1 — control reference label, §3.4/§6).** The FL control
  reference of **26/100** is the independently verified **3-camera
  re-eval** figure for the pinned control checkpoint (§6, weights
  SHA-256 `5b1cbf29…`); the original judged-completion table's 25/100
  was the same weights under the defective 2-camera configuration.
  The ordering control in §3.4 reads against 0 / 26 / 70.
- **A1.2 (F2 — subset generation procedure, §2/Appendix A).** The
  Appendix A subsets were generated by CPython `random.Random(20261124)`:
  tasks iterated in split.json order (block_stack, eye_drops_to_basket,
  ring_insert, tool_insert, cable_clip); development subset =
  `sample(gate list sorted ascending, 4)`; envelope subset =
  `sample(remaining members sorted ascending, 4)` **continuing the same
  RNG stream**. The §2 label "seed 20261125" for the envelope subset is
  corrected: both subsets derive from the single 20261124 stream
  (reproduction verified 2026-11-22 — the procedure regenerates
  Appendix A exactly). The printed Appendix A lists remain the binding
  artifact regardless of procedure.
- **A1.3 (F3 — hash coverage, §1).** The v2 `param_hash` input set
  explicitly includes, per skeleton item 1: all v2 pipeline code, the
  v2 parameter table, **calibration data, the pose-file manifest, and
  fixtures**, plus the Drake scene files and both engines' variant
  manifests. The Stage 1 suite manifest must enumerate the actual
  hashed inputs; a hash whose inputs are not enumerated does not
  satisfy §1.
- **A1.4 (F4 — control scope and verifier mapping, §3.4/§9).** (a) The
  base-fails-all control applies to **all five tasks**: the four
  precision tasks under their v2 instruments, and cable_clip under its
  Stage 0 (twin-suite v1.1) instrument when the controls battery runs
  the gate protocol end-to-end. (b) §9's verifier classes map to the
  governing skeleton's item 10 as follows: (a) parameter audit ≙
  objects/assets + parameter table; (b) Drake port + agreement harness
  ≙ environments/poses and code; (c) envelope/gate harness ≙ code;
  (d) controls re-runs ≙ end-to-end. Every class is verified in a
  separate context from its builder, as §9 requires.
