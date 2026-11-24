# Judging brief — RoboColosseum upright-bottle (blind primary judging)

You are judging whether recorded human-teleoperated demonstration
episodes actually completed their task. Each episode has ONE contact
sheet JPG: two rows (row 1 = front camera, row 2 = wrist camera —
this dataset has no top view), six frames each, left→right in time
(0%, 35%, 60%, 80%, 92%, ~100%). Judge the END STATE — what the final
frames show — not how elegant the motion was. Use the wrist view to
confirm what the gripper holds or released when the front view is
ambiguous.

## Verdicts (exactly one per episode)

- PASS — the final frames clearly show the completion end-state for
  the criterion below, with the gripper no longer holding the target
  object.
- FAIL — the final frames clearly show the end-state is NOT the
  completion state. A success achieved mid-episode but undone by the
  end is FAIL — end state only.
- INDETERMINATE — the final frames do not let you determine the end
  state (object occluded in both views at the end, ambiguous, or
  unreadable frames). Use sparingly; if one camera shows it clearly,
  judge from that.

## Dataset criterion (criteria-frozen.md §3, verbatim)

- **upright-bottle** ("Pick up the fallen juice bottle and stand it
  upright on the table"): PASS = bottle standing upright on its base
  on the table at end, gripper released, not leaning on another
  object. FAIL = bottle still on its side, held, or fallen again by
  the end.

## Output format (your final report: one line per episode, nothing else)

ep<index> | PASS|FAIL|INDETERMINATE | <one short clause: what the final frames show>

Judge only from the sheets. Do not open any other file, do not look
for other verdicts, do not speculate about frames you cannot see.
