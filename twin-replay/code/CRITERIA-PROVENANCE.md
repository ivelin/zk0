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

The flagged queue is printed in the test output on every run — it is the
honest pending list, not a failure. It shrinks only by a derivation
being measured or the founder ruling on a disposition; never by deleting
the entry (the census checks above catch that).

**Origin**: this regime is the structural fix from the ring band-edge
postmortem — see `twin-replay/ring-band-postmortem-20261125.md`. The
frozen criteria themselves are NOT amended by the manifest; dispositions
are founder decisions.
