# Judging brief — BusyBox multitask (blind primary judging)

You are judging whether recorded human-teleoperated demonstration
episodes actually completed their task. Each episode has ONE contact
sheet JPG: two rows (row 1 = front camera, row 2 = top camera), six
frames each, left→right in time (0%, 35%, 60%, 80%, 92%, ~100%).
Each episode is judged against ITS OWN task string: compare the
start state vs the end state of the NAMED control in the final
frames.

## Verdicts (exactly one per episode)

- PASS — the final frames clearly show the completion end-state for
  the episode's task rule below.
- FAIL — the final frames clearly show the end-state is NOT the
  completion state (named control unchanged, wrong control actuated,
  wrong position/orientation). A success achieved mid-episode but
  undone by the end is FAIL — end state only.
- INDETERMINATE — the final frames do not let you determine the end
  state (the named control's state cannot be read in either view).
  Use sparingly; if one camera shows it clearly, judge from that.

## Task rules (criteria-frozen.md §4, verbatim)

- **Push the {green/yellow/white/red} button** — PASS = the named
  button is in its actuated state at end (visibly depressed/switched
  relative to its start state, or its indicator state changed), and
  no other button was the one actuated. FAIL = named button unchanged
  while another button changed, or no actuation visible by the end.
  INDETERMINATE if the named button's state cannot be read.
- **Move the {left/right} slider to position N (1–5)** — PASS = the
  named slider's handle rests at position N at end (read against the
  printed scale). FAIL = handle at a different position.
  INDETERMINATE if the scale/handle position cannot be read in
  either view.
- **Turn the {left/right} toggle switch up to "On" / down to
  "Off"** — PASS = the named toggle points up (On) / down (Off) as
  instructed at end. FAIL = opposite orientation.
- **Turn the knob to position N (1–6)** — PASS = the knob's pointer
  rests at position N at end. FAIL = different position.
  INDETERMINATE if the pointer cannot be read.
- **Reach {forward / to the right / to the left} and open your
  gripper** — PASS = at end the gripper is open, empty, and
  positioned in the named direction relative to the BusyBox panel.
  FAIL = gripper closed, holding an object, or clearly in a
  different direction.

## Output format (your final report: one line per episode, nothing else)

ep<index> | PASS|FAIL|INDETERMINATE | <one short clause: what the final frames show>

Judge only from the sheets. Do not open any other file, do not look
for other verdicts, do not speculate about frames you cannot see.
