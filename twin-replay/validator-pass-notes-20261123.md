# Validator-Impersonation Pass — running notes (2026-11-23, started ~20:12 CT)

Founder directive (voice, 2026-11-23 ~20:10–20:12 CT): act as an external zk0
validator for all five Study 001 tasks. Per task the set is EXACTLY FIVE scenes:
1 anchor twin (digital copy of one held-out physical-arm episode; accuracy
reference only, NEVER a denominator scene — Decision 63 stands) + 4 validator
scenes (spec-only, built blind from the task description only, frozen-envelope
draws). A validator scene counts only with: parameter audit + cross-engine
sample + an executed pass within the pre-stated fair-shot budget + cleanliness
grade (CLEAN / RECOVERED / INCIDENT). Ring carve-out: frozen-criterion defect
(RING_THICK 12.0 mm vs tube Ø 11.5 mm — a perfectly flat seat scores FAIL)
carried openly; founder decision pending; ring NOT claimed validated.

## State found (verified at the sources, 2026-11-23 ~20:15 CT)

- Desktop spec-only program (agent a3b6ab60, Pop!_OS, workdir
  ~/muse-work/zk0-spec-only-20261123) built a FULL spec-only gate suite:
  20 scenes/task (spec00–19), 1,140 scene XMLs, master seed per manifest,
  attempt budget stated BEFORE escalations (Route A nominal both controllers;
  Route B k=1..4 sampled variants; Route C further variants).
- Executed passes on record (PROVEN, not yet CERTIFIED — program rule 3
  requires independent re-execution in a separate context):
  block 13/20 scenes, cable 20/20, eye 14/20, tool 7/20, ring 0/20
  (rev3 round still running at 20:09 CT — ring spec19 live; DO NOT TOUCH).
- Escalations finished for block/cable/eye/tool (escalations_master.log).
- Tier C demo set (DGX, 4 scenes × K=5/task): denominators 0/4 per task —
  superseded as the working set by the desktop full suite + this pass's
  5-scene validator sets.
- Parameter audit: exists for the v2 instrument (parameter-audit-v2.md,
  40 rows, CERTIFIED-WITH-FINDINGS); OI-1 (mat dimensions) + OI-2 (cube
  material/mass) open, founder-only resolvable. Spec-only specs carry
  per-parameter labels ([ESTIMATE]/[DEFAULT] gaps G-* registered in
  spec-only-v2.md §5 + G-C4/G-E5 hygiene additions).
- Cross-engine: Stage-1 controls battery (replica scenes) MuJoCo↔Drake
  policy-outcome agreement 15/15 (floor caveat: all outcomes fail at L1).
  NO cross-engine sample exists yet for spec-only scenes → GAP this pass.
- Frontier realism screen on spec-only renders: renders exist
  (work/spec_only_full/renders/, spec00+spec01 × front/top per task) but no
  recorded frontier screen verdicts → GAP this pass.
- Anchor accuracy record (v1.1 completion report, param_hash a4eb3f8c…):
  cable 20/20 episodes reproduced (outcome class), agreement 1.00, solver
  failures 0/480 — twin FIDELITY certified for cable. Precision tasks:
  block 0/20, eye 3/20, ring 0/20, tool 0/16 episodes reproduced —
  pose σ ≈ 2.5–3.5 cm vs 1–2 cm tolerances is the binding limit.

## Validator sets selected (1 anchor + 4 variants per task)

Anchors prefer replica scenes holding certified reference-controller
solutions (solutions SSOT, HF zk0bot/sim-scene-solutions @ e57de85a):
- block_stack: anchor ep411 (certified solution, rev1+rev2.1).
- eye_drops_to_basket: anchor ep1609 (certified solution, rev1).
- tool_insert: anchor ep683 (certified solution, frozen set per register §R5).
- cable_clip: anchor ep2020 (certified solution rev1; v1.1 fidelity set).
- ring_insert: anchor ep7 (replica dev scene; NO certified solution exists
  for any ring scene — carried openly).

Validator variants (canonical pass = nominal k=0 preferred, then lowest k;
controller per program rule 5):
- block: spec02 (frozen k0), spec08 (frozen k0), spec10 (frozen k0),
  spec13 (frozen k0).
- eye: spec16 (rev21 k0), spec00 (rev21 k1), spec01 (rev21 k1),
  spec17 (rev21 k1).
- tool: spec00 (frozen k0), spec06 (rev21 k0), spec09 (rev21 k3),
  spec19 (rev21 k3).
- cable: spec00 (frozen k0), spec01 (frozen k0), spec02 (frozen k0),
  spec03 (frozen k0). NOTE: cable spec00–03 records carry the SAME scene
  sha256 prefix (365e88f81438) — cable scenes share the arm/fixture base;
  per-scene identity to be confirmed at certification (full sha + manifest).
- ring: spec00–03 selected; validated 0/4 (rev3 round under the desktop
  agent still in flight at selection time; this pass does not run ring).

## Findings as they landed (2026-11-23 evening)

- **Realism screen (separate context, 20 renders spec00/spec01 × 2 cams):**
  block REALISTIC; eye REALISTIC-WITH-NOTES (basket crosses tray rear wall
  line in topcam vs frontcam inconsistency; semi-transparent basket, floor
  not visually confirmable; spec00≈spec01 visually); ring
  REALISTIC-WITH-NOTES (board renders as a large circular disc; ring-hole vs
  peg clearance unverifiable from renders); tool UNREALISTIC *at render
  level* (tool not identifiable in renders; oversized tray slab protrudes
  through the containment wall in frontcam; slot recess not established
  visually); cable UNREALISTIC *at render level* (24-capsule chain renders
  as disconnected fragments; clip channel not visible; spec00=spec01
  renders). Render-level verdicts recorded as-is; geometry-level record
  (audited builders, executed passes, v1.1 cable fidelity certification)
  is reported alongside in the final report — the screen is one input,
  not a silent override in either direction.
- **CABLE IDENTITY FINDING:** nominal scene files for cable spec00–03 are
  BYTE-IDENTICAL (sha 3348c5ce…) — the cable generator's nominal draw is
  the single declared default coil (gap G-C1) for every scene; only the
  k-variant draws differ per scene. Fix applied in this pass: cable
  validator scenes = spec00 k0, spec01 k1, spec02 k2, spec03 k3 (distinct
  files, distinct draws). All three new draws produced executed PASSes
  (frozen controller) on first execution in this pass.
- **Canonical changes from cleanliness grading (record-based rubric:**
  visible events only — grasp retries / drops / re-places count;
  IK-internal fallbacks that never lose the object and the universal
  post-release retreat IK failure do not): eye spec01 canonical moves
  from rev21_k1 (4 grasp entries, grasp-retry) to **rev21_k2** (single
  grasp, no retries). eye spec00 canonical stays rev21_k1 (clean). eye
  spec16 (k0 pass has grasp retries → RECOVERED) and eye spec17 (only
  pass k1 has grasp retries → RECOVERED): cleaner-solution search running
  within budget (spec16 rev21 k1–k4; spec17 rev21 k5–k8, total ≤10).
- **Anchor accuracy (from validation_results_v11.json at the source):**
  block ep411 — recorded success; twin nominal FAIL (agreement 0/5),
  planar err 8.6 mm, secondary (position) PASS, mechanical FAIL →
  outcome-class MISMATCH (position close, contact criteria unmet).
  eye ep1609 — recorded success; agreement 1/5, nominal planar err
  11.83 cm, nominal FAIL → MISMATCH. tool ep683 — recorded success;
  agreement 0/5, planar err 13.03 cm, tool still attached at end →
  MISMATCH. ring ep7 — recorded success; agreement 0/5, planar err
  3.94 cm; nominal checks show bottom_above_board_m = −0.0003 (the
  criterion defect visible in the anchor's own numbers) → MISMATCH +
  defect evidence. cable ep2020 — recorded FAILURE; twin reproduces the
  failure on 5/5 variants (agreement 1.0, certified variants 5/5) →
  outcome-class MATCH.
- **Certification re-executions (round 1) so far:** block spec02/08/10/13
  all PASS reproduced; eye spec00 k1, spec01 k1, spec16 k0 PASS
  reproduced. Remainder in flight.

## Certification + cross-family stage (2026-11-23 evening, later)

- **Certification re-executions: 21/21 pairs reproduce EXACTLY** (verdict
  match, max final-object deviation 0.000000 m) — all 16 round-1 canonicals
  + eye spec01 k2, eye spec16 k4, cable spec01 k1 / spec02 k2 / spec03 k3.
- **eye spec16 canonical = rev21 k4** (CLEAN, certified). **eye spec17:
  budget EXHAUSTED (10 attempts: 2 Route A + k1–k4 + k5–k8; k5/k6/k7/k8
  all FAIL with grasp retries)** — canonical stays rev21 k1, RECOVERED,
  certified; scene is solvable but has NO clean solution within budget.
- **Ring rev3 escalation: 112 records on file, ALL FAIL** (still running
  under the desktop program at last check). Ring stays 0/4.
- **Anchor solution grades (record rubric):** ep411 RECOVERED (3 failed
  grasp approaches; founder's own video read: "borderline pass"), ep1609
  RECOVERED (3 grasp retries), ep683 RECOVERED (2 grasp retries + failed
  approaches), ep2020 CLEAN-by-record (weld device caveat). No cleaner
  alternative records exist (ep411's rev1 record is a FAIL). The anchors'
  certified solutions prove solvability; three of four do NOT meet the
  founder's 20:13 good-solution spec — a cleaner anchor solution needs a
  new executor round (founder decision, cf. the ring rev3 mandate).
- **Cross-family CLI stage:** first run failed on both CLIs — **Grok:
  HTTP 402 "Grok Build usage balance exhausted"** (the balance was
  consumed by today's 1,800-episode certification leg; one bounded retry
  queued to confirm); **agy: argument-order error** (`-p` consumed
  `--model`), fixed by ordering `--model` before `-p`. Runner 2: Gemini
  leg (5 tasks, gemini-3.8-flash-high) + Grok retry + a Muse probe
  (Meta family — a valid second non-Muse family under the rotation
  rule if its DGX sign-in is live).

## Work plan (gaps only)
1. Independent certification: re-execute the 16 canonical passes above from
  the raw scene files (SPEC_VARIANT env), own run dir on the desktop under
  job_guard, compare verdict + final object positions vs original records.
2. Cleanliness grading of the 16 canonical trajectories from their records
  (checks + phase notes + move log), basis labeled record-based.
3. Frontier realism screen (separate context) on the render set, per task.
4. Cross-engine sample for spec-only scenes (Drake on DGX vs MuJoCo settle
  records) — attempt; if the harness cannot ingest spec scenes in scope,
  log as open gap with the Stage-1 result as partial coverage.
5. Per-task validator report + separate-context verification + commit via
  the output-minimized route on study001/fl-substrate-20261121.
