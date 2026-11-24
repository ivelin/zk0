# Judging brief — DecisionFacts Cup Nesting (blind primary judging)

You are judging whether recorded human-teleoperated demonstration
episodes actually completed their task. Each episode has ONE contact
sheet JPG: two rows (row 1 = front camera, row 2 = top camera), six
frames each, left→right in time (0%, 35%, 60%, 80%, 92%, ~100%).
Judge the END STATE — what the final frames show — not how elegant
the motion was.

## Verdicts (exactly one per episode)

- PASS — the final frames clearly show the completion end-state for
  the criterion below, with the gripper no longer holding the cup.
- FAIL — the final frames clearly show the end-state is NOT the
  completion state.
- INDETERMINATE — the final frames do not let you determine the end
  state (nesting cannot be confirmed from either view). Use
  sparingly; if one camera shows it clearly, judge from that.

## Dataset criterion (criteria-frozen.md §5, verbatim)

- **Cup Nesting** ("Pick up the empty solo cup and nest it inside
  the other static cup"): PASS = the picked cup rests nested inside
  the static cup (inner cup's wall inside the outer cup, rims
  nested), gripper released, at episode end. FAIL = cups still
  separate, inner cup left beside/on top without nesting, or the
  static cup was the one moved. INDETERMINATE if the nesting cannot
  be confirmed from either view.

## Output format (your final report: one line per episode, nothing else)

ep<index> | PASS|FAIL|INDETERMINATE | <one short clause: what the final frames show>

Judge only from the sheets. Do not open any other file, do not look
for other verdicts, do not speculate about frames you cannot see.

## PROTOCOL — contested episodes (founder, 2026-11-23 21:50 CT)
If any episode you judge becomes contested (your read disagrees with
another leg, or you return INDETERMINATE), your handoff for that
episode must include, beyond the verdict: (a) a chronological
trajectory walkthrough of the episode, and (b) anomalies/insights
observed. See criteria-frozen.md Amendment 3.
