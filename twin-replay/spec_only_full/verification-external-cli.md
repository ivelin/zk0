# External CLI verification (V3) — spec-only FULL suite solution records

Founder rule 2026-11-23 14:29 CT: the final verification stage for
important results (solution trajectories first) runs through BOTH the
Grok Build CLI and the Gemini Antigravity (agy) CLI as independent
verifiers, on top of the internal separate-context verification
(V2: 28/28 nominal re-executions reproduced; V2b: 27/27 variant
re-executions reproduced).

Setup (2026-11-23, DGX): package at
/home/openclaw/muse-work/spec-v3-20261123/ — register.json, the 55
canonical solution records, the 55 canonical scene files, VERIFY.md
(audit all 55 records against the frozen criteria in v1replay.evaluate;
re-execute a spot sample of >=10, >=2 per task with solutions; strict
CONFIRMED/DISPUTED per scene). Each CLI ran one session, blind to the
other's verdict. Grok Build CLI v1.0.88 (~/.local/bin/grok, --single,
--max-turns 80); agy v1.5.15 (~/.local/bin/agy, -p).

## Grok leg — 33 CONFIRMED / 22 DISPUTED (grok-verdict.json)
All 22 disputes are eye_drops_to_basket (15) and tool_insert (7)
records. Ground: Grok re-ran the frozen evaluate/controllers in the
DGX native EPISODE configuration, where build_scenes.load_fixtures
reads the episode fixtures.json (basket after parallax; tool slot at
0.234,0.058); under that fixture set the recorded checks do not
reproduce. Grok's own notes confirm the physical end states match
(e.g. eye: "basket contact and carton_z 0.0205 match the record").
Grok also flagged eye spec12's record-internal inconsistency
(verdict FAIL vs sim_success true).

## agy leg — 54 CONFIRMED / 1 DISPUTED (agy-verdict.json)
agy re-executed with the spec-only harness semantics — the frozen
evaluate's fixture lookup overridden in-process to the spec scene's
own basket/slot_floor poses (the geometry physically present in the
scene XML) — and reproduced the eye/tool passes. Its summary
describes this mechanism explicitly. Sole dispute: eye spec12, on
substantive grounds — record/register verdict FAIL (grasp failed;
carton knocked into basket during retries) and an early return
skipped the required 1.0 s settle before evaluate, though its
re-execution reproduced sim_success=true.
Informational edge cases noted by agy (criteria are frozen; no
action): tool evaluate does not forbid residual gripper contact nor
threshold tool_z (two tool solutions rest at axis_angle 19.5 deg,
inside the 20 deg bound); block face_gap_m is diagnostic-only (five
records 0.0054–0.0061 m); five block nominal records encode
scene_variant as "nominal k=0" without a variant_kind field (cosmetic
schema inconsistency).

## Resolution
- Fixtures contest (22 records): resolved FOR the records. The
  deciding considerations: (1) the agy tiebreak — a frontier model
  not involved in building — confirmed all 22 under the spec-only
  harness; (2) first principles: a spec-only scene's basket/slot
  exist only at the manifest/XML positions, so scoring against
  episode-fixture coordinates would judge geometry that is not in
  the scene; the Tier C harness (driver-level fixtures override from
  the scene's target_specification) is the registered spec-only
  evaluation path, and V2/V2b reproduced every record through it.
  Grok's dissent and its ground are preserved here and in
  grok-verdict.json.
- eye_drops_to_basket__spec12: disputed by BOTH legs (Grok:
  verdict/sim_success inconsistency; agy: failed grasp, carton
  knocked in during retries, settle skipped). EXCLUDED from the
  certified solution set: register certification status =
  failed_verification with grounds recorded. It was a variant-level
  solution (k=6 pose corner); no nominal denominator is affected.

## Final certified counts (after V3)
- Canonical solutions certified through all stages: 54 of 55
  (28 nominal + 26 variant-level).
- Denominators (certified nominal solutions): block_stack 5/20,
  cable_clip 20/20, eye_drops_to_basket 1/20, ring_insert 0/20,
  tool_insert 2/20.
