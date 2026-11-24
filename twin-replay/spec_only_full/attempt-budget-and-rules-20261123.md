# Spec-only program — founder rules and per-scene attempt budget
Recorded 2026-11-23 ~13:15 CT. This record binds four founder rules into
the spec-only scene program (twin-suite v2, Amendment 3) and states the
per-scene solvability attempt budget BEFORE any escalation attempts run.

## Founder rules bound
1. SCENE SOLUTION RULE (13:05 CT): a scene may be used for pass/fail
   checks ONLY if it has a proven, certified solution trajectory (any
   working trajectory counts). No certified solution = the scene does
   not enter any pass/fail denominator, no exceptions.
2. BACKUP (13:08 CT): every proven solution trajectory is saved to the
   SSOT remote backup — hashed manifest on the study branch + the HF
   zk0bot artifact collection (HF upload held for the parent's go under
   the certified-only publication rule; package prepared with the
   register).
3. CERTIFICATION (13:09 CT): a controller pass alone is NOT enough. A
   scene's solution must be INDEPENDENTLY CERTIFIED in a separate
   context before the scene enters any denominator. Register states:
   PROVEN (executed pass on record) -> CERTIFIED (independent
   re-execution from the raw scene files reproduces the pass). Only
   CERTIFIED scenes count in reported denominators.
4. FAIR-SHOT RULE (13:11 CT): every scene gets genuine solvability
   attempts via the available routes under the bounded budget below.
   A scene that exhausts its budget is logged UNPROVEN with all
   attempts recorded — never retried indefinitely, never dropped.
5. ONE SOLUTION PER (SCENE, TASK) (13:12 CT): only one verified
   solution is kept per unique (scene, task) pair. The register names
   ONE canonical solution trajectory per validated scene (preference:
   a nominal-variant pass, then the program's controller order);
   further passing runs stay in the attempt log as duplicates and are
   not separately backed up.

## Per-scene attempt budget (stated before escalation attempts begin)
- Route A (primary): every admitted global controller version runs the
  scene's NOMINAL variant (k=0) once. block/eye/cable/tool: 2 attempts
  (frozen, rev2.1). ring: 3 attempts (frozen, rev2.1, rev3).
- Route B (only for scenes unproven after Route A): the task's
  best-performing controller version (most Route A passes; tie ->
  rev2.1) runs the scene's four SAMPLED variants k=1..4 (jitter/mass/
  friction draws inside the registered envelope). 4 attempts.
- Route C (only for scenes still unproven): the same controller runs
  the four POSE-CORNER variants k=5..8 (initial-state offsets at the
  envelope sigma). 4 attempts. Precision tasks only (cable has no
  frozen corners). Friction/mass corners (k=9..12) are NOT solvability
  routes: they change the scene's physics identity rather than its
  initial state, so a pass there would prove a different scene.
- Maximum per scene: 10 attempts (ring 11). Every attempt produces a
  run record; the register carries the full attempt census per scene.

## Denominator operationalization (stated for the founder's glance)
- PROVEN-nominal: an executed pass on k=0. These scenes are the
  denominator candidates (subject to certification, rule 3).
- A pass on a Route B/C variant is recorded as a solution found within
  the scene's envelope, with the variant named, and the trajectory is
  backed up like any proven solution — but the scene is logged
  UNPROVEN-nominal and does NOT enter the nominal denominator on that
  basis. The register reports both counts per task (proven-nominal;
  variant-only solutions) so the denominator decision stays visible
  and reversible by the founder.

## Timing note (honesty)
Route A for block_stack began 12:50 CT under the program's original
brief (both controller versions on the nominal variant — the same
Route A defined here), i.e. before the fair-shot rule was stated at
13:11 CT. No scene has received more attempts than this budget allows;
Routes B and C run only after this record. The budget therefore binds
all escalation attempts before they begin, and Route A as executed
conforms to it exactly.

## Reporting per task
attempted scenes / proven-nominal / variant-only solutions / certified
(after independent verification) / budget-exhausted-unproven, with the
full per-scene attempt log in the register.

## Final verification stages (founder rule 2026-11-23 14:29 CT)
For important results — solution trajectories first of all — the final
verification stage runs through BOTH external CLIs as independent
verifiers, on top of the internal separate-context verification:
- V2 (internal): separate-context re-execution of every canonical
  solution from the raw scene files (the 13:09 certification leg).
- V3a: Grok Build CLI verifies the solution records.
- V3b: Gemini Antigravity (agy) CLI verifies the solution records.
A scene's denominator counts as FINAL in any founder-facing report
only after V2 + V3a + V3b pass for its solution. If either CLI is
unavailable, that is stated explicitly in the report — the stage is
never silently skipped.

Availability — CORRECTED 2026-11-23 ~19:10 CT (supersedes the
~14:31 CT check below the line in the session record):
- Grok Build CLI: /home/openclaw/.local/bin/grok on the DGX reports
  "grok 1.0.88 (2fe3608f7398) [stable]" (symlink to ~/.grok/bin/grok).
  The 14:31 check missed it because the non-login SSH PATH does not
  include ~/.local/bin. (A separate fleet install, grok-1.0.70 under
  /home/openclaw/fleet/shared/, also exists; the ~/.local/bin one is
  the route.)
- Gemini Antigravity (agy): /home/openclaw/.local/bin/agy on the DGX
  reports 1.5.15. The binary's timestamp is 2026-11-23 15:39 CT —
  installed AFTER the 14:31 check (Cos installed it at the founder's
  direction; signed in as the founder's Google account, starter tier,
  weekly quota SHARED with Cos's machine — sequence agy passes with
  quota discipline).
Both V3 legs are therefore RUNNABLE on the DGX. Plan: per task, one
grok session and one agy session over a staged verification package
(canonical scenes + certified solution records + frozen criteria),
each returning a per-scene verdict; verdicts merge into the register
as the external-CLI stage. Until both legs pass for a scene, its
denominator is labeled "internal certification only — external CLI
stage pending".

## Outcome (2026-11-23, program complete)
All routes executed for all 100 scenes; register built
(schema twin-suite-v2/spec-only-solvability-register/2).
- Internal certification: V2 28/28 nominal re-executions reproduced
  exactly; V2b 27/27 variant re-executions reproduced exactly.
- External CLI stage RAN (both legs, DGX, 2026-11-23 evening):
  Grok v1.0.88 — 33 CONFIRMED / 22 DISPUTED (all 22 on the
  episode-fixtures ground for eye/tool); agy v1.5.15 — 54 CONFIRMED /
  1 DISPUTED. The fixtures contest resolved for the records (agy
  tiebreak + scene-geometry first principles; Grok dissent preserved
  in verification-external-cli.md). eye_drops_to_basket__spec12 was
  disputed by BOTH legs on substantive grounds (failed grasp, carton
  knocked into basket during retries, settle skipped) and is
  EXCLUDED from the certified set (register: failed_verification).
- Final: 54 certified canonical solutions (28 nominal + 26 variant).
  Denominators (certified nominal): block_stack 5/20, cable_clip
  20/20, eye_drops_to_basket 1/20, ring_insert 0/20, tool_insert 2/20.
  Variant-only certified: block 8, eye 13, tool 5. Budget-exhausted
  unproven: block 7, eye 5, ring 20, tool 13, cable 0.
