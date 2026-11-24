# Judging brief — ArmnetBench held-out tasks certification (blind primary judging)

You are judging whether recorded human-teleoperated demonstration
episodes actually completed their task. Each episode has ONE contact
sheet JPG: two rows (row 1 = front camera, row 2 = top camera), six
frames each, left→right in time (0%, 35%, 60%, 80%, 92%, ~100%).
Judge the END STATE — what the final frames show — not how elegant
the motion was.

## Verdicts (exactly one per episode)

- PASS — the final frames clearly show the completion end-state for
  the episode's task criterion below, with the gripper no longer
  holding the target object.
- FAIL — the final frames clearly show the end-state is NOT the
  completion state. A success achieved mid-episode but undone by the
  end is FAIL — end state only.
- INDETERMINATE — the final frames do not let you determine the end
  state (object occluded in both views at the end, ambiguous, or
  unreadable frames). Use sparingly; if one camera shows it clearly,
  judge from that.

## Task criteria (criteria-frozen.md §6; your batch names ONE task)

- **eye_drops_to_shelf** ("Put the eye drops on the shelf."): PASS =
  the eye-drops bottle rests on the shelf surface, gripper released,
  at episode end. FAIL = bottle anywhere else (table, basket, floor)
  or still held.
- **tool_removal** ("Remove the small middle tool (3/8 Extension Bar
  75mm) from the toolbox."): PASS = that tool (the small middle one)
  is out of the toolbox, lying on the table, gripper released, at
  end. FAIL = the tool still sits in its toolbox slot, or a different
  tool was removed instead.
- **cable_unclip** ("Remove the power cable from cable holder."):
  PASS = the cable is fully out of the holder — no longer retained
  by the clip — gripper released, at episode end. FAIL = cable still
  seated in the holder, or only partially pulled with the clip still
  holding it.

## Output format (your final report: one line per episode, nothing else)

ep<index> | PASS|FAIL|INDETERMINATE | <one short clause: what the final frames show>

Judge only from the sheets. Do not open any other file, do not look
for other verdicts, do not speculate about frames you cannot see.
