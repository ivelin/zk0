# Postmortem — the ring criterion's `bottom_above_board_m ≥ 0.0` band (`low_ok`)

**Ordered by the founder, 2026-11-25.** Blameless: this names roles,
artifacts, and process gaps, not people. All actors were AI agents
working under the twin-replay program; the founder directed the program
but never set this value.

**Evidence base**: the band-edge probe (2026-11-24);
`twin-replay-preregistration-20261122.md`; `docs/physics-parameters.md`,
`docs/twin-validation-report.md`, `docs/verification-code.md`,
`docs/builder1-v11-harness.md`, `docs/verification-v11-code.md` (all in
`twin-replay/docs/`); the frozen code in `twin-replay/code/`
(`replay.py`, `evaluate()`), byte-identical across every working copy.

## Answers in brief

1. **Who/when**: the band was created in two steps on 2026-11-22. The
   *concept* ("ring's lowest point within 1 cm of the board top
   surface") is prose in the coordinator's pre-registration. The
   *≥ 0.0 lower edge* — the form that fails real seats — was set by
   **Builder 1 in the v1.1 fixpass** the same day, recorded in its own
   report: "0 ≤ ring_bottom − board_top ≤ 1 cm (v1 credited 5 mm
   interpenetration)."
2. **Why**: fidelity to the registered text. V1's code had allowed
   `ring_bottom − board_top ≥ −0.005` (a 5 mm interpenetration credit);
   verifier finding F-03 flagged that as exceeding the pre-registration;
   the fixpass removed the credit and the code docstring glossed the
   result "no interpenetration credit". **No physical derivation was
   recorded at any step** — nobody measured what a correctly settled
   ring reads until the band-edge probe of 2026-11-24 (≈ −3 µm).
3. **Founder decision? No.** No decision record anywhere sets the band,
   the 1 cm, or any twin mechanical-criterion value. The founder's
   recorded inputs were the twin-replay directive and refinements of
   2026-11-22 — program direction, not criterion values.
4. **Root cause**: the provenance regime covered physics *parameters*
   but criterion *thresholds* lived in prose and code literals, outside
   it — and every downstream gate checked fidelity to the registered
   text, never the text's own derivation or physical attainability.
5. **Sweep**: 21 scoring/device-gating thresholds audited — **8
   derived, 4 rationale-only, 9 underived**. The ring band is not the
   only invented number; the worst other case (SLOT_SIZE) was later
   measured plain wrong.
6. **Fix**: the criteria-provenance manifest + regression test landed
   with this change (below).

## Timeline

| When (2026) | What | Record |
|---|---|---|
| Nov 19 | Study 001 success criteria drafted for the *judged* eval: ring completion = ring encircles the shaft below the peg tip; seating depth explicitly unspecified | study success-criteria draft §4 |
| Nov 22, before any run | Twin-replay pre-registration states the mechanical ring criterion in prose: "peg passes through the ring hole, ring's lowest point **within 1 cm of the board top surface**". No derivation, no lower-edge discussion. Its honesty rule 1 requires sources for *physics parameters* — criteria are not mentioned | `twin-replay-preregistration-20261122.md` |
| Nov 22 | Physics parameters: ring dims [MEASURED-EST]; nothing on a scoring band. Parameter-source audit later counts the tags clean: 14 MEASURED-EST / 3 PROP / 13 DEFAULT | `docs/physics-parameters.md` |
| Nov 22 | V1 code implements `low_ok` as **−0.005 ≤ ring_bottom − board_top ≤ 0.010**, plus engagement with silent +5 mm +2 mm extras | `docs/verification-code.md` F-03 |
| Nov 22 | V1 validation report records, as a weld-artefact aside: "**ring bottom-above-board median ≈ 0.0 cm**, so the effect is small on final states" — the empirical distribution was already sitting on the future band edge; never read as a criterion question | `docs/twin-validation-report.md` |
| Nov 22 | Independent v1 code verification: F-01 (contact criteria never implemented), F-02 (secondary never thresholded), **F-03 (MODERATE): the −5 mm interpenetration credit is not in the registered text** | `docs/verification-code.md` |
| Nov 22 | **V1.1 fixpass (Builder 1)**: criteria rewritten to the registered text verbatim; ring becomes `0.0 <= (ring_bottom - bt) <= 0.010`, docstring "no interpenetration credit" | `docs/builder1-v11-harness.md` Fix 1 |
| Nov 22 | Independent v1.1 verification confirms approvingly: "low_ok = 0 <= ring_bottom - board_top <= 0.010 — v1's 5 mm interpenetration credit is gone". Checked fidelity, not attainability | `docs/verification-v11-code.md` |
| Nov 22–23 | Suite versions run; criteria freeze. The band decides nothing: no executed run ever produces a flat seat | twin-replay completion reports |
| Nov 24 | Critique flag P1 (RING_THICK role misuse) upheld; scorer parameter corrected 0.012 → 0.0115 in the canonical criteria record. The band itself is not questioned | real-vs-sim critique, 2026-11-24 |
| Nov 24 | **Band-edge probe**: 37/37 proven-flat seats read bottom_above = −0.0 (raw −2.85…−4.42 µm) and FAIL on low_ok alone; the one criteria-passing end state on record sits +0.001 m (1 mm high) | band-edge probe memo, 2026-11-24 |
| Nov 25 | Founder rejects the band approach itself, rules that thresholds need recorded derivations, orders this postmortem | founder ruling, 2026-11-25 |

## Root cause (the process gap)

The twin build had a real provenance regime — for physics parameters.
Honesty rule 1 ("every physics parameter carries a source… no silent
inventions") produced tagged parameter tables that an independent audit
verified clean. **Criterion thresholds were never brought under that
regime.** They were authored as prose in a pre-registration and as
literals in code, with no manifest, no derivation field, and no check
that a threshold is physically attainable by the state it claims to
recognise.

Every quality gate after that point — v1 verification, the v1.1 fixpass,
v1.1 verification, the suite freeze — asked one question: *does the code
match the registered text?* That loop is closed: it catches code
deviations (F-01/F-02/F-03 were real catches), but it cannot catch an
underived text, because the text is its own standard. When F-03 forced
a decision about the −5 mm credit, the fixpass optimised for the only
value the loop could see — textual fidelity — and set the edge to
exactly 0.0, the one value a settled contact cannot reach (resting
contact sits micrometres below the geometric surface). The single
empirical signal that could have broken the loop — v1's own "median
seat ≈ 0.0 cm" — was filed under a different causal story (weld
artefacts), so it never arrived at the criterion's owners as a question.

In one line: **thresholds were reviewed for consistency, never for
provenance or attainability, because the provenance regime's scope
stopped at parameters.**

## Sweep — every gated threshold in the frozen criteria

Scope: `replay.evaluate()` primaries + secondary criterion +
`run_validation.py` bars + the replay-device thresholds that gate what
outcomes the replay can produce (all five tasks). "Derived" = computed
from measured geometry/metadata or a recorded experiment. The v2 gate's
own bars (Wilson ≥ 60%, envelope bounds measured from controls) are
founder-approved and are not criterion-code thresholds.

| # | Threshold | Gates | Derivation on record? |
|---|---|---|---|
| 1 | **ring low_ok band [0.0, 0.010]** | ring PASS | **NONE — the defect** |
| 2 | ring engagement ≤ RING_R_IN − PEG_R (3 mm) | ring PASS | YES — physical radial clearance |
| 3 | block footprint inset 25% (×0.75) | block PASS | NONE — preregistration convention |
| 4 | eye centre-inside-basket footprint | eye PASS | YES — basket interior geometry |
| 5 | tool centre-in-slot, SLOT_SIZE (0.110 × 0.032) | tool PASS | **NONE — LOCAL-CHOICE; later MEASURED WRONG (real ≈81 mm, waist ≈23 mm)** |
| 6 | tool axis angle ≤ 20° | tool PASS | NONE — convention (real seats ≲5.5°) |
| 7 | cable channel x half 0.012 | cable PASS | YES — built jaw half-length (V11-02) |
| 8 | cable channel y extra +0.002 | cable PASS | YES — jaw inner faces' outer edge (V11-02) |
| 9 | cable channel z half 0.012 | cable PASS | RATIONALE-ONLY — "twice the jaw half-height"; ×2 is a choice |
| 10 | cable channel centre offset +0.006 | cable PASS | YES — jaw half-height |
| 11 | cable solver envelope (radial 1.0 m; z ∈ [−0.05, 0.50]) | denominator exclusion | YES — workspace geometry + observed explosion bimodality |
| 12 | secondary planar ≤ 3.0 cm (block/eye/tool) | secondary PASS | NONE — convention |
| 13 | secondary ring ≤ 2.0 cm (not engaged) | secondary PASS | NONE — convention |
| 14 | episode REPRODUCED ≥ 0.6 of variants | episode verdict | NONE — Amendment 1 convention |
| 15 | task certification bar ≥ 0.8 agreement | task certification | NONE — convention (superseded by the founder-approved v2 bar) |
| 16 | RING_THICK = 0.012 as consumed by the scorer | ring bottom computation | **NONE in role** — measured *axial* thickness used as *tube diameter* (P1; corrected to 0.0115) |
| 17 | grasp attach distance 3.0 cm | replay attach | RATIONALE-ONLY — object-scale heuristic |
| 18 | attach closedness 0.50 | replay attach | RATIONALE-ONLY — stroke-midpoint heuristic; absolute form later replaced by p5/p99 normalisation |
| 19 | release closedness 0.30 | replay release | RATIONALE-ONLY — hysteresis heuristic |
| 20 | gripper min excursion 5° | replay attach | NONE — no basis stated |
| 21 | settle time 1.0 s | state read timing | NONE — convention (protocol, low stakes) |

Also recorded, non-gating: pose JITTER scales were explicit placeholders
for unmeasured pose error (flagged at the time); FRICTION_MULT_RANGE
±20% is a convention; MASS_RANGES are sourced (derived); TIMESTEP
2000 Hz is derived from a recorded stability experiment; FPS is read
from dataset metadata (derived).

**Headline: 21 gated thresholds — 8 derived, 4 rationale-only, 9 with
no derivation at all.** All 13 non-derived entries are now flagged
manifest entries with pending founder dispositions.

## Structural fix (landed with this change)

- **Manifest** — `twin-replay/code/criteria-provenance.json`: 78 entries
  classifying every numeric constant in `twin_params.py` and every
  threshold literal in `replay.py` / `run_validation.py`, each with
  kind, status, derivation + source for derived values, and flag +
  disposition for the rest. Regime: `twin-replay/code/CRITERIA-PROVENANCE.md`.
- **Regression test** — `tests/unit/test_criteria_provenance.py`
  (stdlib only, 8 tests): fails on an unclassified constant, a stale
  manifest value, a pattern that no longer matches the criteria source
  exactly once, an uncovered comparison literal, **any** unclassified
  numeric literal inside `evaluate()`, or a gated entry lacking
  derivation+source or flag+disposition. The flagged queue (16 entries)
  prints on every run.
- **CI wiring** — `.github/workflows/ci.yml` already runs
  `pytest tests/unit` on every push/PR, so the test is picked up
  automatically when Actions runs (budget-blocked until the 2026-12-01
  reset at the time of writing); locally it runs with the repo's normal
  test command.
- **Drill evidence** (local run, pytest 9.2.0, 2026-11-25):
  - GREEN on the honest state: `8 passed` — with the band recorded as
    UNDERIVED / pending disposition. The test forbids silence, not
    honesty.
  - RED-1, manifest deleted: hard fail ("criteria-provenance manifest
    missing").
  - RED-2, ring-band entry removed: fails naming the uncovered literals
    `[0.0, 0.01]` in both the comparison scan and the `evaluate()`
    census.
  - RED-3, invented constant `INVENTED_TOL_M = 0.007` added to
    `twin_params.py`: fails naming it.
  - RED-4, band silently widened to `<= 0.012` in `replay.py`: fails on
    pattern mismatch (`code.ring_band`) plus the uncovered literal
    `[0.012]`.
- **What watches the watcher**: the census checks mean the manifest
  cannot be gamed by deleting the awkward entry or the manifest itself;
  pattern-anchoring means any criteria edit breaks the test loudly
  until the manifest is reviewed in the same change; CI re-runs it on
  every push/PR, and PR review (builder ≠ verifier) sees manifest diffs
  as the provenance surface. Known limits, stated: the test guards the
  repo copies — working trees elsewhere receive the manifest when this
  change lands and they re-sync; integer literals in comparisons are
  out of the literal scan's scope by design (indices/counters; the
  params census and pattern entries cover the known items); a combined
  code+manifest edit in one PR is visible in review but not
  machine-blocked.

## Recommended disposition of `low_ok` (recommendation — NOT enacted)

The frozen criterion is not amended by this postmortem. When the
founder rules, the recommended replacement is to **delete the height
band and score the physical outcome the engine already resolves**:
engagement (peg through the hole — a geometric fact, already derived)
**plus settled contact** — the ring in contact with the board/peg at
settle with the gripper released — the same contact-based primary style
the other four tasks already use (ring is the only task whose primary
is a position band). Until the ruling, the criterion stands as frozen
and the manifest carries the band as FLAGGED-UNDERIVED.

**Impact note**: the band decided no historical verdict — every ring
execution failed at grasp/lift before the seat. Its demonstrated harm
is prospective and diagnostic: it fails every deliberately-placed
correct seat (37/37), and it would have failed the first genuinely
successful seat the program produces.

---

## Addendum — enactment of the ruling (2026-11-25, later the same day)

The founder ruled (2026-11-25 11:50 CT, zk0 chat) that the ring seat
check be replaced by a physical test, and corrected the fix's priority
the same day (12:00 CT): the load-bearing deliverable is the upstream
gate, not the exhibit cases. Both are enacted locally on branch
`criteria-provenance-20261125` (uncommitted working tree at the time of
writing; the ship decision is the founder's):

1. **Physical seat test** (`twin-replay/code/replay.py`, ring branch of
   `evaluate()`): the band `0.0 <= bottom_above_board_m <= 0.010` is
   deleted. The ring passes the seat check iff `engaged and
   seat_support and settled`, all read from the physics engine's
   resolved state — the peg passes through the ring's hole across the
   peg's span (planar bound = the parts' own measured clearance,
   hole Ø22 mm − peg Ø16 mm in the frozen build's modelled geometry),
   the ring is in contact with the board or the peg at settle end, and
   it is settled: no gripper contact, no active grasp weld, and body
   and rim speeds below one integration step of gravitational velocity
   gain (|g|·dt from the model's own gravity and timestep — the
   engine's rest resolution, not a chosen epsilon). `bottom_above_board_m`
   is still recorded, diagnostic only. Manifest: `code.ring_band` is
   FOUNDER-DECIDED (abolished by the ruling) and superseded by
   `code.ring_seat_test`. Symptom check: the postmortem's exhibits flip
   correctly — the 34 recorded flat seats and 3 corrected-washer seats
   pass; the floating ep48 v2 end state fails (no support contact at
   its recorded pose). The A2 run's own result files were not in the
   local record; the floating verdict rests on the A2 report's recorded
   pose values.
2. **The gate is strengthened** (`tests/unit/test_criteria_provenance.py`):
   every SCORING entry now declares a `quantity` class, a band on an
   engine-resolved quantity without a derivation is a hard failure, and
   the flagged queue is closed at the 15 entries this postmortem
   recorded — it may only shrink. Drilled: a fresh underived band
   injected into the block criterion PASSED the pre-strengthening test
   when its author self-flagged it (the hole), and FAILS the
   strengthened gate; the same injection into the tool criterion fails;
   an unclassified band fails the census; the honest tree is green.
   The remaining 15 flagged thresholds are untouched and remain listed
   as flagged pending their own founder rulings.
3. **Owed syncs (divergences, not edited here)**: the Drake cross-engine
   port (`twin-replay/twin-suite-v2/drake/drake_replay.py`) still
   carries the identical band and is outside the manifest's scope; the
   HF `zk0bot/sim-scene-solutions` `criteria/` folder still holds the
   band; DGX live-tree copies of `replay.py` still hold the band.
   Propagation is a maintainer decision.
