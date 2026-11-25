# Criteria provenance — twin-suite scoring criteria

The frozen twin-suite criteria (`replay.py`, `twin_params.py`,
`run_validation.py` in this folder) score pass/fail. Every number in them
that gates a score is a claim about the physical world or a convention
someone chose — and the record must say which, and on what basis.

**Rule** (founder, 2026-11-25): every numeric threshold in a criterion
needs a recorded derivation — a physical measurement or a founder
decision. An underived number is a defect. It may remain in the frozen
record only as a flagged entry with a pending disposition, never
silently. Scoring should let the physics engine resolve (drop, bounce,
settle) and score the physical outcome, not fight it with hand-made
bands.

**Manifest**: `criteria-provenance.json` in this folder classifies every
numeric constant in `twin_params.py` and every threshold literal in
`replay.py` / `run_validation.py`:

- `DERIVED` — computed from measured geometry, dataset metadata, or a
  recorded stability/attainability experiment (derivation + source
  required).
- `FOUNDER-DECIDED` — the value is a recorded founder decision.
- `RATIONALE-ONLY` / `UNDERIVED` — no derivation exists. The entry must
  be `flagged: true` and carry a `disposition` (what is pending, and
  where the decision record will live).
- `LABELED` — scene geometry/material parameters under the
  physics-parameters tagging regime ([MEASURED-EST] / [PROP] /
  [DEFAULT]), protocol constants that need no derivation by design
  (e.g. a fixed seed), and incidental numerics (indices, rounding).

**Enforcement**: `tests/unit/test_criteria_provenance.py` runs in CI
(`.github/workflows/ci.yml` → `pytest tests/unit`) and fails if

- a numeric constant appears in `twin_params.py` with no manifest entry,
- a threshold literal in `replay.evaluate()` or `run_validation.py` is
  not covered by an entry,
- any numeric literal inside `evaluate()` is unclassified,
- a manifest pattern no longer matches the criteria source exactly once
  (the code changed under a recorded threshold), or
- a gated (SCORING/DEVICE) entry lacks either a derivation + source or
  an explicit flag + disposition.

**Strengthened 2026-11-25 (founder correction, 12:00 CT)** — classifying
a number is not a gate; the gate is that an invented number cannot get
in. Two further failure modes:

- **The flagged queue is closed.** It held exactly the 15 underived
  entries the 2026-11-25 postmortem recorded (after the ring band's
  resolution) and may only shrink. A flagged entry outside that frozen
  set fails the suite: a new threshold can no longer be admitted by
  flagging it.
- **Every SCORING entry declares its `quantity` class** —
  `engine-resolved` (a height, position, gap, angle, settle or contact
  outcome the physics engine already resolves), `input-geometry` (a
  dimension of a built/measured object, which must cite the
  measurement), `agreement-statistic` (a fraction over verdicts), or
  `physics-state` (a boolean read of engine state, no numeric band). A
  band on an engine-resolved quantity with no derivation is a hard
  failure; the only exceptions are the frozen postmortem flags, each
  awaiting its own founder ruling. Undeclared quantity = unclassified =
  failure.

The flagged queue is printed in the test output on every run — it is the
honest pending list, not a failure. It shrinks only by a derivation
being measured, a conversion to a physics-state read, or the founder
ruling on a disposition; never by deleting the entry (the census checks
above catch that) and never by growing (the closed-queue check catches
that).

**Origin**: this regime is the structural fix from the ring band-edge
postmortem — see `twin-replay/ring-band-postmortem-20261125.md`. The
frozen criteria themselves are NOT amended by the manifest; dispositions
are founder decisions.

**Amendment 2026-11-25 (criteria A+B change)** — the founder ruled on
the whole flagged queue, and the criteria were amended accordingly
(each manifest entry carries its own derivation or ruling citation;
the decision record is `criteria-rulings-20261125.md`):

- *Derived from measurements on record*: `SLOT_SIZE` (81 × 29 mm) and
  the new `SLOT_DEPTH` (15 mm) from the P3 tool-slot measurement
  (confirmed second opinion); `code.tool_angle` 20° → 6° (the measured
  end-wall lean cap; observed rest angles ≤ 5.5°); `CABLE_CHAN_Z_HALF_M`
  0.012 → 0.006 (the built clip-jaw half-height, like the x bound);
  `RING_THICK` 0.012 → 0.0115 (not a measurement of the real ring —
  a flat-faced washer with no tube, axial ~12.0 mm measured; 11.5 mm
  is the built torus's tube diameter, adopted by the founder ruling of
  2026-11-24, ring-peg-evidence-20261124.md §6; the former 0.012 was
  the washer's axial thickness in the wrong role); `JITTER` → the measured
  pooled pose-error tier (xy p90 0.088 m, n=73; per-axis σ 0.0426 m now
  feeds the live sampler) with yaw recorded as an unmeasured null.
- *Converted to physics-state reads*: the fixed `SETTLE_S` wait →
  settle-until-rest (`code.settle_rest`); `GRASP_DIST_M` proximity →
  pad–object contact (`code.grasp_contact`); the block footprint inset
  → contact + sole support + settled (`code.block_seat_test`).
- *Founder-decided conventions*: `FRICTION_MULT_RANGE` (±20% stress
  envelope) and `SECONDARY_THRESH_M` (near-miss reporting label) stand
  as decided, values unchanged.
- *Aligned / struck*: `code.reproduced_frac` now applies the approved
  v2 Wilson lower-bound ≥ 60% standard; `code.bar_80` is struck as
  superseded by that standard (record retained).
- *Resolved by the gripper follow-up change (2026-11-25)*: the gripper
  cluster (`ATTACH_CLOSEDNESS` 0.50 → 0.47, `RELEASE_CLOSEDNESS`
  0.30 → 0.35, `GRIP_MIN_EXCURSION_DEG` 5.0 → 4.4) — derived from the
  arm-calibration plateau data (324 certified episodes; fit on train,
  validated on held-out; gripper-constants-derivation-20261125.md).
  The flagged queue is now empty.

Noted during the amendment, outside this regime's census (which covers
`twin_params.py`, `replay.evaluate()` and `run_validation.py` only):
`run_episode`'s grasp state machine still carries pre-existing
unclassified literals — the elongated-object centre bounds (carton
0.060 / tool 0.075 / cable 0.045) and the 0.03 m release-recency
eligibility distance. They are recorded here as known defects awaiting
their own disposition, not silently endorsed by this change.
