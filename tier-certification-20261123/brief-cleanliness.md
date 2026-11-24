# Cleanliness grading brief — tier-10h certification (orthogonal axis)

You are grading the CLEANLINESS of recorded human-teleoperated
demonstration episodes. This is a SEPARATE axis from pass/fail: do
NOT judge pass/fail. Grade only what happened during execution, from
the episode's contact sheet (two camera rows × six frames at
0/35/60/80/92/~100% — look at ALL frames, not only the final ones).

## Grades (exactly one per episode)

- **CLEAN** — no incident visible in any frame.
- **RECOVERED** — a distinct problem occurred (the target was
  dropped, slipped from the gripper, was misplaced, or was knocked
  over/down) AND the demonstrator visibly recovered: re-grasped it,
  picked it back up, stood it back up, or corrected the placement —
  and the episode then ends in a completed-looking state.
- **INCIDENT** — glitches occurred but there was no distinct
  problem-and-recovery: e.g. the arm brushed or nudged a non-target
  object, displaced a non-target object that simply stayed displaced,
  bumped a fixture, or the target slipped inside the gripper without
  being dropped.

Precedence: if a distinct problem+recovery occurred, grade RECOVERED
even if lesser glitches also happened (list everything). Otherwise
any visible glitch → INCIDENT. Otherwise CLEAN.

## What to describe (one line per incident)

- Incidental contact with a non-target object (touching, brushing,
  nudging anything that is not the target object or its destination).
- Knocking or displacing another object from its start position.
- Dropping the target object (and whether it was recovered).
- The target slipping in the gripper mid-carry.
- Knocking the target into place rather than placing it.
- Gripper/arm colliding with fixtures (bowl rim, shelf, drawer,
  panel, table edge) hard enough to move them or jar the scene.
- A misplacement that was then corrected (describe both).

Ordinary purposeful motion is NOT an incident: grasping the target,
carrying it, placing it, opening/closing a drawer the task names,
touching the destination receptacle to place the object in it.

## Output format (your final report: one line per episode, nothing else)

ep<index> | CLEAN
ep<index> | RECOVERED | <incident + recovery, one line> ; <further incidents>
ep<index> | INCIDENT | <incident 1, one line> ; <incident 2, one line>

Judge only from the sheets. Do not open any other file and do not
look for pass/fail verdicts — this axis is independent.
