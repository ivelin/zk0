# Study 001 Twin-Replay — Amendment 3 (v1.1): tool_insert ep779 OUT of the validation denominator

Coordinator ruling, recorded by the v1.1 micro-fix pass (changelog MF-3).
This is a bookkeeping application of the registered rule, NOT a criteria change.

## Ruling

tool_insert ep779 is OUT of the twin-suite v1.1 validation denominator.

## Basis

- Fix-pass FP-H (docs/fixpass-v11.md; changelog FP-9) relabelled ep779 on
  evidence: its tool pose is FK-anchor-only, visually unconfirmable
  (V-Env+Poses F2: no loose tool visible in top_first, visual status
  indeterminate), so confidence was corrected medium -> low
  (sigma_xy_cm 3.5 -> 4.3). The pose VALUE was not changed.
- The preregistration (docs/twin-replay-preregistration-20261122.md)
  registers the low-confidence exclusion: episodes whose initial state
  cannot be pinned are named and excluded from the validation denominator.
  That is the same rule that excludes tool_insert ep630, ep871 and ep876.
- With its only initial object at confidence low, ep779 fails the
  run_validation inclusion test (CONF_OK = {medium, high}, episode
  confidence = minimum over initial objects) and is excluded by that
  documented path — the same mechanical path as ep630/871/876.

## Consequence

- RUN2 final validation therefore ran at denominator 96 (tool_insert 16),
  not 97 (tool_insert 17). See docs/revalidation-v11-final.md, which
  recorded this as a deviation; this amendment is the ruling that settles it.
- ep779 interim agreement was 0.0, so no headline number changes.

## Markers

- scenes/tool_insert__ep779.json now carries the additive top-level field
  v11_status: "OUT — low-confidence FK-only pose (amendment3)".
  No pose values were changed.
- Changelog entry MF-3 in validation/param-changelog.md states the same ruling.
