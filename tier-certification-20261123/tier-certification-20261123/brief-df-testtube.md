# Judging brief — DecisionFacts Test Tube Nesting (blind primary judging)

You are judging whether recorded human-teleoperated demonstration
episodes actually completed their task. Each episode has ONE contact
sheet JPG: two rows (row 1 = cam_front, row 2 = cam_top), six frames
each, left→right in time (0%, 35%, 60%, 80%, 92%, ~100%). Judge the
END STATE — what the final frames show — not how elegant the motion
was. Each episode is judged against ITS OWN task string (two tasks
in this dataset).

## Verdicts (exactly one per episode)

- PASS — the final frames clearly show the completion end-state for
  the episode's task rule below, with the gripper no longer holding
  the target object.
- FAIL — the final frames clearly show the end-state is NOT the
  completion state (object left outside its target, or still held).
- INDETERMINATE — the final frames do not let you determine the end
  state (object occluded in both views at the end, ambiguous, or
  unreadable frames). Use sparingly; if one camera shows it clearly,
  judge from that.

## Task rules (criteria-frozen.md §8(b); task strings confirmed from
## the downloaded meta/tasks.parquet and frozen 2026-11-23)

- **Task 0** — "Pick and lift the plate, transport it, and seat it
  flat in the nest with all four corners down." PASS = the plate
  rests seated flat in its nest (all four corners down), gripper
  released. FAIL = the plate left outside the nest, tilted/propped
  with a corner visibly up, or still held at end.
- **Task 1** — "Pick the test tube and properly place it in the tube
  tray." PASS = a test tube stands placed in the tray/holder,
  gripper released. FAIL = the tube left outside the tray or still
  held at end.

## Output format (your final report: one line per episode, nothing else)

ep<index> | PASS|FAIL|INDETERMINATE | <one short clause: what the final frames show>

Judge only from the sheets. Do not open any other file, do not look
for other verdicts, do not speculate about frames you cannot see.
