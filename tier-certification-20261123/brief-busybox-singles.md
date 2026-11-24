# Judging brief — BusyBox singles (blind primary judging)

You are judging whether recorded human-teleoperated demonstration
episodes actually completed their task. Each episode has ONE contact
sheet JPG: two rows (row 1 = front camera, row 2 = top camera), six
frames each, left→right in time (0%, 35%, 60%, 80%, 92%, ~100%).
Judge the END STATE of the NAMED control — compare its start state
(first frames) with its state in the final frames.

## Verdicts (exactly one per episode)

- PASS — the final frames clearly show the named control in its
  actuated / instructed end state, per the dataset criterion below.
- FAIL — the final frames clearly show the named control unchanged
  while another control changed, or no actuation happened, or the
  control ends in the opposite state.
- INDETERMINATE — the named control's state cannot be read in either
  view. Use sparingly; if one camera shows it clearly, judge from
  that.

## Dataset criteria (your batch names ONE dataset; apply its rule)

- **bb-push-green** ("push the green button"): PASS = the green
  button is in its actuated state at end (visibly depressed/switched
  relative to its start state, or its indicator state changed), and
  no other button was the one actuated. FAIL = green button unchanged
  while another button changed, or no actuation visible by the end.
  INDETERMINATE if the green button's state cannot be read.
- **bb-flip-left** ("Flip the left switch off"): PASS = the left
  toggle switch points down (Off) at end. FAIL = the left toggle
  points up (On) at end, or a different switch was the one flipped.
  INDETERMINATE if the toggle's orientation cannot be read.

## Output format (your final report: one line per episode, nothing else)

ep<index> | PASS|FAIL|INDETERMINATE | <one short clause: what the final frames show>

Judge only from the sheets. Do not open any other file, do not look
for other verdicts, do not speculate about frames you cannot see.
