# Validator-Impersonation Pass — Per-Task Report (2026-11-23)

Founder directive 2026-11-23 ~20:10–20:13 CT: act as an external zk0
validator for the five Study 001 tasks. Per task the set is EXACTLY FIVE
scenes: ONE anchor twin (digital copy of one externally sourced held-out
episode; physical-accuracy reference only, never a gate denominator) +
FOUR validator scenes generated from the task description only, blind to
the training/eval data (spec-only path, Decision 63 / Amendment 3).
A scene's solution counts only with (a) an executed pass, (b) independent
certification by re-execution in a separate context, (c) a CLEAN
cleanliness grade under the founder's good-solution spec (smooth,
failure-free, relatively optimal, minimal hesitation/corrections), and
(d) validation by at least TWO frontier models from families different
than the generator (generator family for all scenes/controllers here:
Muse (Muse Wonder)).

Working notes: `validator-pass-notes-20261123.md` (same folder).
Scene/solution records: desktop `~/muse-work/zk0-spec-only-20261123/`
(builder program) and `~/muse-work/zk0-validator-pass-20261123/` (this
pass: certification re-executions, supplementary attempts, bundles).

## Headline accounting

| Task | Validator scenes | Solvable + certified | CLEAN + certified | Cross-family GOOD (of 2 required) | Fully counted |
|---|---|---|---|---|---|
| block_stack | 4 | 4 | 4 | GPT-OSS 4/4 · second leg blocked | pending 2nd leg |
| eye_drops_to_basket | 4 | 4 | 3 | GPT-OSS 3/3 clean · second leg blocked | pending 2nd leg |
| tool_insert | 4 | 4 | 4 | GPT-OSS 4/4 · second leg blocked | pending 2nd leg |
| cable_clip | 4 | 4 | 4 | GPT-OSS 4/4 · second leg blocked | pending 2nd leg |
| ring_insert | 4 | 0 | 0 | GPT-OSS concurs: NO-SOLUTION | 0 — NO GATE |

Certification: **21/21 canonical solutions re-executed independently in
this pass reproduced EXACTLY** (verdict match; max final-object
deviation 0.000000 m — the pipeline is deterministic). Composition of
the 21 pairs (per independent verification): 17 re-execute the spec-only
program's original records; 4 (eye spec16 rev2.1 k4 and the three new
cable draws) re-execute this pass's own first executions, because those
solutions were first found inside this pass.

Second-leg blockers (reported, not silently skipped — founder rule
2026-11-23 14:29 CT): **Grok Build CLI: HTTP 402 "usage balance
exhausted"** (confirmed on a bounded retry; the balance was consumed by
the same day's 1,800-episode data-certification leg). **Gemini via agy:
weekly group quota exhausted mid-run** (resets ~2026-11-30; two calls
returned empty under a headless tool-permission denial, three returned
429 — zero usable Gemini verdicts this cycle). **Muse: not signed
in on the DGX** ("missing meta credentials" — the founder sign-in from
the Main lane is still pending). The second family leg is OWED for all
15 clean scenes; it completes when any one of {Gemini quota reset,
Grok balance restored, Muse sign-in} lands.

## Accuracy checks — common to all tasks

- **Parameter audit:** exists and stands (`twin-suite-v2-docs/
  parameter-audit-v2.md`, 40 rows, CERTIFIED-WITH-FINDINGS). Every
  spec-only parameter carries a source label; gaps are registered, never
  silently guessed. Open founder-only items carried openly: **OI-1**
  (mat dimensions unrecorded), **OI-2** (cube material/mass unrecorded),
  plus spec-only register gaps incl. **G-C4** (cable joint damping
  0.005 in the builder vs 0.05 in the v2 spec parameter table — value
  unchanged, carried openly) and **G-E5** (eye scene spec under-declares
  its staged contents).
- **Cross-engine (MuJoCo ↔ Drake):** established at instrument level —
  Stage-1 controls battery outcome agreement **15/15** on replica scenes
  (floor caveat: all outcomes fail at L1, so agreement is necessary, not
  sufficient). **OPEN GAP:** the Drake harness cannot yet ingest
  spec-only scene files; no spec-scene cross-engine sample exists.
  Spec-scene object placements are declarative inputs shared by both
  engines; closing this gap needs a Drake scene-ingestion extension.
- **Realism screen (frontier model, separate context, renders of
  spec00/spec01 × front/top cameras per task):** block REALISTIC;
  eye REALISTIC-WITH-NOTES; ring REALISTIC-WITH-NOTES; tool UNREALISTIC
  at render level; cable UNREALISTIC at render level. Per-task detail
  below; render-level verdicts are recorded as-is alongside the
  geometry-level record — neither silently overrides the other.

## block_stack

- **Anchor ep411** (certified solution on record, rev2.1 controller).
  Accuracy: recorded episode outcome = success; twin nominal replay
  FAIL — variant outcome agreement **0/5**, nominal planar error
  **8.6 mm**, secondary (final-position) check PASS, mechanical stack
  criteria FAIL. **Outcome-class MISMATCH at the mechanical level** —
  the twin places the cube at the recorded position but does not
  reproduce the recorded contact outcome (pose σ ≈ 2.5–3.5 cm vs the
  task's contact tolerances; the settled v1.1 finding). Anchor solution
  quality: my grade RECOVERED (three failed grasp-approach moves before
  the successful grasp); GPT-OSS graded it CLEAN (counted the failed
  approaches as IK internals that never lost the object) — **grader
  disagreement recorded**; the founder's own video read of this scene
  was "borderline pass". The rev1 record for ep411 is a FAIL; no cleaner
  alternative solution exists on record.
- **Validator scenes:** spec02, spec08, spec10, spec13 — all proven at
  nominal (frozen controller), all certified by exact re-execution.
  Cleanliness: spec02 CLEAN (one IK lift fallback while holding the
  cube; disclosed), spec08 CLEAN (same note), spec10 CLEAN, spec13
  CLEAN. GPT-OSS: 4/4 CLEAN, GOOD=YES.
- **Realism screen:** REALISTIC (both cubes correct, plausible poses).
- **Open defects:** none task-specific beyond the common items.

## eye_drops_to_basket

- **Anchor ep1609** (certified solution on record, rev1/frozen).
  Accuracy: recorded success; twin agreement **1/5 variants**, nominal
  planar error **11.83 cm**, nominal FAIL. **Outcome-class MISMATCH.**
  Anchor solution quality: RECOVERED (three grasp-retry retreats) —
  GPT-OSS concurs (GOOD=NO). Does not meet the good-solution spec.
- **Validator scenes:** spec16, spec00, spec01, spec17.
  - spec16: nominal pass carried grasp retries; a cleaner solution was
    found within budget — canonical **rev2.1 k4** (single grasp),
    CLEAN, certified by exact re-execution.
  - spec00: canonical rev2.1 k1, CLEAN, certified.
  - spec01: canonical moved from k1 (grasp retries) to **rev2.1 k2**
    (single grasp), CLEAN, certified.
  - spec17: **solvable + certified but NO clean solution within the
    fair-shot budget** — 10 attempts total (2 Route A controllers +
    k1–k4 + k5–k8); the only pass (rev2.1 k1) carries grasp retries →
    RECOVERED. GPT-OSS concurs (GOOD=NO). Scene counts as proven
    solvable; it does NOT count under the good-solution spec.
  - GPT-OSS on the three clean scenes: 3/3 CLEAN, GOOD=YES.
- **Realism screen:** REALISTIC-WITH-NOTES — topcam shows the basket's
  footprint crossing the tray's rear wall line while frontcam does not
  (view inconsistency); the basket renders semi-transparent with no
  visually confirmable floor; spec00 ≈ spec01 visually (their parameter
  draws differ sub-visually).
- **Open defects:** spec17's grasp-hostile nominal placement (every
  controller generation retries grasps there) — a scene-design data
  point for future generators.

## tool_insert

- **Anchor ep683** (certified solution on record, rev1/frozen — the
  task's only certified replica solution). Accuracy: recorded success;
  twin agreement **0/5**, nominal planar error **13.03 cm**, replay ends
  with the tool still attached to the gripper. **Outcome-class
  MISMATCH.** Anchor solution quality: RECOVERED (two grasp retries,
  failed approaches, failed lift) — GPT-OSS concurs (GOOD=NO).
- **Validator scenes:** spec00 (frozen, nominal — a flawless record: no
  failed moves at all), spec06 (rev2.1 nominal), spec09 (rev2.1 k3),
  spec19 (rev2.1 k3). All certified by exact re-execution; all CLEAN
  (only the universal post-release retreat IK failure appears, after
  task completion). GPT-OSS: 4/4 CLEAN, GOOD=YES.
- **Realism screen:** UNREALISTIC *at render level* — the tool is not
  identifiable in the renders (a thin white sliver at the tray edge),
  the tray slab reads as disproportionately massive and in frontcam
  protrudes through the containment wall, and the slot's recess is not
  visually established. Geometry-level counterpoint, recorded with
  equal weight: the tool is a real capsule body in the scene (it rolls;
  settle checks pass), its slot fit is mechanically checked in every
  executed pass (centre-in-slot + axis angle 9.9–17.3° vs the 20°
  criterion), and four independent scenes produced certified passes.
  Disposition: the scenes stand on their mechanical record; the
  render/presentation layer (camera framing, tool contrast) is flagged
  as a defect for any human-facing use of these renders.
- **Open defects:** render legibility (above); scene spec's slot
  position is identical across variants (only the tool start is drawn)
  — a distribution-coverage note, not a validity defect.

## cable_clip

- **Anchor ep2020** (certified solution on record, rev2.1, v2 scene
  copy). Accuracy: the recorded episode is a **failure**
  (recorded_success = 0); the twin reproduces that failure on **5/5
  variants, agreement 1.0, certified variants 5/5** — **outcome-class
  MATCH**, the strongest anchor result of the five tasks (and the
  v1.1 cable suite's fidelity certification stands behind it).
  Anchor solution quality: CLEAN-by-record (weld-device caveat below) —
  GPT-OSS concurs (GOOD=YES).
- **Validator scenes — IDENTITY FINDING:** the generator's nominal
  cable scenes spec00–spec03 are **byte-identical files** (the nominal
  draw is the single declared default coil, register gap G-C1, for
  every scene). Four identical scenes are not four validator scenes.
  This pass therefore selected four DISTINCT draws: **spec00 k0,
  spec01 k1, spec02 k2, spec03 k3** (distinct files, distinct
  jitter/mass/friction draws). All four produced executed passes
  (frozen controller) and all four were certified by exact
  re-execution in this pass. Cleanliness: CLEAN-by-record for all
  four — GPT-OSS concurs (4/4 GOOD=YES).
- **Weld-device caveat (applies to the anchor and all four scenes):**
  cable solutions are executed with the documented v1.1 weld device
  because the gripper pads cannot pinch a 5 mm cable (min face gap
  ~8 mm). The trajectories prove the scene solvable and the clip
  mechanics correct; they are not natural manipulation trajectories.
- **Realism screen:** UNREALISTIC *at render level* — the 24-capsule
  cable chain renders as disconnected fragments, the clip channel is
  not visible at render scale, and spec00 = spec01 renders (a direct
  consequence of the identical nominal files above). Geometry-level
  counterpoint: the chain is a jointed capsule continuum in the model
  (segment positions are continuous in every record), and cable is the
  one task whose twin fidelity is certified end-to-end (v1.1).
  Disposition as for tool: mechanical record stands; render layer
  flagged.
- **Open defects:** the generator produces ONE nominal cable scene, not
  a distribution — future cable scene generation needs per-scene coil
  draws at nominal, not only in k-variants.

## ring_insert — NO GATE (carve-out honored)

- **Anchor ep7** (replica dev scene): **no certified solution exists**
  for ep7 or any ring scene. Accuracy record: recorded success; twin
  agreement **0/5**, nominal planar error **3.94 cm**; the nominal
  checks themselves display the criterion defect —
  `bottom_above_board_m = −0.0003`.
- **Validator scenes spec00–spec03: 0/4 validated.** Every attempt on
  record failed: frozen and rev2.1 controllers (Route A + variants),
  and the rev3 executor's escalation — **rev3 records on file: 112 at
  this pass's first check, 120 at independent verification, 121 at
  report close (2026-11-23 20:48 CT) — still growing, ALL FAIL**, with
  the desktop program's escalation still in flight. GPT-OSS concurs
  with the disposition: GRADE=NO-SOLUTION, exclusion from the
  validated denominator is correct.
- **Criterion defect (carried openly, NOT fixed by this pass):** the
  frozen criterion's parametric ring bottom (RING_THICK 12.0 mm) sits
  0.25 mm below the modelled tube (Ø 11.5 mm), so a ring resting
  perfectly flat and fully engaged scores bottom_above ≈ −0.0003 →
  FAIL. A true level seat is unscorable under the frozen criteria.
  **A founder decision on the criterion is pending** (and a separate
  founder card on human-in-sim for ring is also unanswered). Ring
  remains NO GATE; nothing in this report claims ring validated.

## Model attribution (rotation record)

- Generator (scenes + reference controllers): Muse (Muse Wonder) agents
  (spec-only program + this pass's supplementary attempts).
- Certifier (independent re-execution): this pass, separate context —
  21/21 exact reproductions.
- Cross-family validator 1: **GPT-OSS 120B** (OpenAI family, via agy,
  `--model gpt-oss-120b`) — all 5 tasks graded; agrees with this pass's
  record grades on all 20 validator scenes, both RECOVERED anchors it
  was shown for eye/tool, spec17's RECOVERED, and the ring disposition;
  **disagrees on anchor ep411** (CLEAN vs this pass's RECOVERED).
- Cross-family validator 2: **OWED** — Grok (xAI) blocked on 402
  balance; Gemini (Google) blocked on weekly quota (resets
  ~2026-11-30); Muse (Meta) blocked on missing sign-in. Raw CLI
  outputs are preserved in the DGX bundle dir
  `~/muse-work/zk0-validator-bundles-20261123/out_*.txt`.

## Independent verification

Separate-context verifier, 2026-11-23: **CERTIFIED-WITH-FINDINGS.**
All seven assigned claim groups were reproduced at the primary sources:
the 21 certification pairs re-compared in full (21/21 verdict match,
0.0 m deviation); the cable scene hashes; the eye spec17 budget
(10 attempts, one pass); every GPT-OSS grade as attributed (including
the ep411 disagreement); all four CLI blocker files (Grok 402, agy 429
×3, agy empty-verdict ×2, Muse missing credentials); the ring rev3
counts; and all five anchors' v1.1 fields. The two findings — ring
count staleness and the 21-pair composition — are folded into the text
above; neither changes any disposition. Outside the verifier's brief
(standing on their own prior records): the realism-screen verdicts,
the parameter audit, the cross-engine 15/15, and the ring criterion
arithmetic.

## What remains owed after this report

1. Second cross-family validation leg for the 15 clean scenes (any one
   of the three blocked families; Gemini quota resets ~Nov 30).
2. Founder decisions: ring criterion parameter; ring human-in-sim.
3. Drake ingestion of spec-only scenes (cross-engine gap).
4. Cleaner anchor solutions for ep411/ep1609/ep683 if the founder
   wants anchors to meet the good-solution spec (needs a new executor
   round on replica scenes — his call, cf. the ring rev3 mandate).
5. Render-layer fixes for tool/cable human-facing samples (framing and
   contrast), and per-scene nominal coil draws in the cable generator.
