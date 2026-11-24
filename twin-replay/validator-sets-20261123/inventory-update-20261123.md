# Inventory Update — spec-only program FINAL delivery folded in (2026-11-23 ~23:30 CT)

Source: the earlier spec-only program's completed delivery (report +
records committed `4b13f1b` on `study001/fl-substrate-20261121` at
`twin-replay/spec_only_full/`; local records
`~/workspace/spec-only-full-20261123/`). This update supersedes the
corresponding figures in `inventory.json` / `inventory.md` (which were
harvested from the program's live register before its V3 stage
closed).

## Count reconciliation

| Task | Harvest (live register) | FINAL certified | Nominal | Variant-only |
|---|---|---|---|---|
| block_stack | 13 | **13** | 5 | 8 |
| cable_clip | 20 | **20** | 20 | 0 |
| eye_drops_to_basket | 15 | **14** | 1 | 13 |
| tool_insert | 7 | **7** | 2 | 5 |
| ring_insert | 0 | **0** | 0 | 0 |
| **Spec-only total** | 55 | **54** | **28** | **26** |

- **eye spec12 EXCLUDED** (register: `failed_verification`): disputed
  on substantive grounds by BOTH V3 legs — record verdict FAIL vs
  sim_success true; the grasp failed and the carton was knocked into
  the basket during retries; the required 1.0 s settle before
  evaluate was skipped. (Tonight's harvest had independently flagged
  this record as anomalous.) Certified-solution grand total
  (spec-only + twins) is therefore **74, not 75**.
- **Ring:** 220 attempts across the suite, budget exhausted, 0
  certified — consistent with everything else on record.
- Nominal denominators (certified nominal / 20): block 5/20,
  cable 20/20, eye 1/20, tool 2/20, ring 0/20.

## Certification credit for the 54 (acceptance rules applied)

- **Internal reproduction (V1/V2/V2b):** stands — 28/28 nominal and
  27/27 variant re-executions reproduced (the 27th variant record is
  the excluded spec12).
- **Grok V3 leg: STRUCK.** It ran pre-suspension under the program's
  old brief (33 CONFIRMED / 22 DISPUTED). Under the founder's
  2026-11-23 21:53 CT suspension, Grok reads certify NOTHING. The 22
  disputes were a fixtures artifact (Grok scored eye/tool records
  against EPISODE fixtures rather than the spec scenes' own geometry;
  its own notes confirm the physical end states match the records).
  The dissent and its ground are preserved in `grok-verdict.json` and
  `verification-external-cli.md`; they do not block any scene.
- **agy V3 leg: CONFIRMED 54 / DISPUTED 1 (spec12) — but THIN as a
  family validation.** Two limits, stated plainly for the morning
  deliverable: (1) it was a record-audit + spot-re-execution leg
  (all 55 records statically audited against the frozen criteria; 12
  solutions re-executed), not a grading against the 20:11
  GOOD-solution standard (smooth / failure-free / relatively
  optimal); (2) **its model is not recorded** — `verification-
  external-cli.md` and the DGX session log name no `--model` pin, so
  the leg ran a single unidentified agy default model in one session.
  It therefore counts as a pass-record audit confirmation, NOT as one
  of the two different-family GOOD validations. All 54 scenes still
  receive tonight's GPT-OSS 120B + advanced Gemini (Neon/Carbon
  rotation) GOOD legs; the Gemini legs double as the advanced-model
  confirmation pass for this thin leg.
- The validator-pass GPT-OSS GOOD legs (15 spec scenes + 4 anchors)
  stand as one GOOD leg each for those scenes.

## Nominal vs variant (founder operationalization still open)

Variant-only solutions are certified but do NOT enter nominal
denominators. Validator-set reporting tonight therefore carries BOTH
counts per task — sets fillable from nominal solutions, and sets
fillable counting variant solutions — never blended silently. (This
bites hardest on eye: 1 nominal vs 13 variant; and tool: 2 vs 5.)

## Backup rule (updated by this delivery)

The spec-only program's HF backup package for the 54 is prepared but
**HELD for founder go** — not uploaded. Tonight's validator-set
packages follow the same rule: **prepare, hold, report** — no HF
upload without the founder's go.
