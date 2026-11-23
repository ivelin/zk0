# Frozen completion criteria — tier-10h certification program

**Frozen 2026-11-23, before any judging of the fresh sources.** Each
source's section was written from that dataset's own published task
strings (read from `meta/tasks.parquet` at the pinned revision) and is
not changed after judging of that source starts. Any change requires a
dated amendment here and re-judging of the affected source.

## Amendment 2 — cleanliness axis (founder, 2026-11-23 17:00 CT)

The pass/fail criteria in this document DO NOT change. Separately and
orthogonally, every judged episode also receives a **cleanliness
grade**, because imitation learning reproduces near-miss glitches.

**Refined taxonomy (founder, 2026-11-23 17:02 CT — replaces the
two-grade scheme of 17:00 CT, before any cleanliness grading ran).**
Passing episodes are graded one of:

- **CLEAN** — no incidents. These form the **imitation core**.
- **RECOVERED** — a distinct problem occurred (drop, slip,
  misplacement, knock-over) AND the demonstrator visibly recovered
  (re-grasp, pick back up, correct the placement) and completed the
  task. These form the **recovery-training subset** — the model must
  also learn to recover, so incident episodes are not waste.
- **INCIDENT** — completed with glitches but no distinct recovery
  (e.g. brushed a non-target object, displaced a non-target that
  simply stayed displaced). These count as **pass evidence only**.

Precedence: any distinct problem+recovery → RECOVERED (all other
glitches still listed in `incidents`); else any incident → INCIDENT;
else CLEAN. Each incident is described in one line in the register's
`incidents` field. The register's `subset` column labels PASS
episodes: imitation core / recovery-training / pass evidence only.
For FAIL / INDETERMINATE episodes the cleanliness grade is recorded
as the same orthogonal observation (CLEAN / RECOVERED / INCIDENT as
seen on the sheet) but no subset label is assigned — subsets are
defined on passes.

Incident taxonomy (non-exhaustive): incidental contact with a
non-target object; brushing or knocking an object; displacing another
object from its start position; dropping the target and recovering
it; dropping a non-target object; knocking the target into place
rather than placing it; the gripper colliding with fixtures (bowl,
shelf, drawer, panel); an object ending somewhere other than where
the task intends, short of failing the end-state rule.

Grading is done in a SEPARATE pass, in separate judge contexts, from
the same contact sheets (all six timepoints, not only the end state).
Recorded limitation: the sheet samples six timepoints — incidents
occurring entirely between sampled frames cannot be seen, and the
grade certifies only "no incident visible on the contact sheet".

Applies from the first source onward: the carried-in ArmnetBench 250
and every fresh source receive the cleanliness pass before being
reported. The founder review pack separates clean passes from
incident passes.

## General rules (all sources, carried from the 2026-11-23 audits)

- Judge the END STATE in the final frames of the contact sheet
  (front/top-equivalent rows, frames at 0/35/60/80/92/~100%). A success
  achieved mid-episode but undone by the end is FAIL.
- PASS — final frames clearly show the completion end-state defined
  below, with the gripper no longer holding the target object.
- FAIL — final frames clearly show the end-state is not the completion
  state (object never moved, wrong object, wrong place, dropped,
  knocked over, still held).
- INDETERMINATE — final frames do not let you determine the end state
  (occluded in both views, unreadable). Use sparingly; if one camera
  shows it clearly, judge from that.
- Evidence: one short clause per episode stating what the final frames
  show. Judge only from the sheet; no speculation about unseen frames.

## 1. ArmnetBench (armnet/armnetbench_v01_lerobot_so101) — CARRIED IN

Not re-judged. Verdicts are the Study 001 frozen criteria (2026-11-20)
results from the 2026-11-23 training-demo certification audit + Grok leg
+ founder adjudication, carried verbatim into the register:
block_stack, eye_drops_to_basket, ring_insert, tool_insert, cable_clip —
criteria text in `training-demo-audit-20261123/JUDGING_BRIEF.md`.

## 2. Spa-Bench (spa-bench-training-teleoperation-1200) — 321 task strings, 6 families

Each episode is judged against ITS OWN task string. The string's pattern
selects the rule (classifier is part of this freeze):

- **Rule A — named object into the bowl** (Object State family; e.g.
  "Put the closed white cup in the bowl"): PASS = the named object is
  inside the bowl at episode end, gripper released. Adjectives
  (closed/open/upright/upside-down, colours) identify WHICH object when
  several of a type are present; they do not impose an end orientation
  inside the bowl. FAIL = the named object is not in the bowl (or a
  different object of the same type was put in when the string's
  description picks out a specific one and both are distinguishable).
- **Rule B — relational placement** (Relational Placement family; e.g.
  "Place the green pen to the left of the blue block"): PASS = the named
  object rests on the table in the stated relation (front/back/left/
  right/next-to/beside/near) to the named reference object at end,
  gripper released. Judge the relation in the table plane from the
  above camera; "near/beside/next to" = adjacent, not touching required.
  FAIL = object left in a clearly different relation, never moved, or
  placed in the bowl instead of on the table.
- **Rule C — referential object into the bowl** (Referential Description
  family; e.g. "Pick up the pen nearest to the crushed Coke can and
  place it in the bowl", "...positioned between A and B..."): the
  referent is identified from the FIRST frames (relative positions at
  episode start). PASS = that object is in the bowl at end, gripper
  released. FAIL = a different object is in the bowl, or the referent
  never reached the bowl. If the referent cannot be identified from the
  first frames, INDETERMINATE.
- **Rule D — ordinal object into the bowl** (Ordinal Position family;
  e.g. "Put the third object from the right in the bowl"): ordering is
  read from the FIRST frames, in the above camera's view, left/right as
  seen in that view. PASS = the object at that ordinal position is in
  the bowl at end. FAIL = a different object is in the bowl or none is.
  If the row of objects cannot be ordered from the first frames,
  INDETERMINATE.
- **Rule E — size referent into the bowl** (Relative Size family;
  "Move the smallest object into the bowl" / "largest"): referent by
  apparent size in the first frames. PASS = that object in the bowl at
  end. FAIL = a clearly different-sized object in the bowl.
- **Rule F — bowl count / removal** (Counting family): "Make/leave the
  bowl contain exactly N objects" → PASS = exactly N objects inside the
  bowl at end. "Remove/Take X out of the bowl / move X from the bowl to
  the table" → PASS = X is out of the bowl (on the table) at end and
  not held. FAIL = the count or X's location is clearly otherwise.

Family assignment per episode is derived from its task string by the
pattern order F > D > E > C > B > A and recorded in the register.

**Classifier amendment (2026-11-23, before any Spa-Bench judging):**
the Rule F pattern also covers "Adjust the bowl so that it contains
{N} objects" (no "exactly" in the string) — 26 episodes. Verified by a
full audit of every string the classifier sent to Rule A against the
dataset's 200-episode family blocks (blocks: 0 Object State,
1 Relational Placement, 2 Referential Description, 3 Ordinal Position,
4 Counting, 5 Relative Size): all other Rule A assignments in blocks
2/3/5 are plain named-object-into-bowl strings, for which Rule A is
the semantically correct rule. The register records both the judging
rule applied and the dataset family block.

## 3. RoboColosseum suite (front + wrist cameras)

- **so101-cube-drawer** — "Put the blue cube on the table into the open
  drawer, then close the drawer." PASS = blue cube inside the drawer
  AND the drawer closed at end, gripper released. FAIL = cube outside
  the drawer, or drawer left open with the cube visible outside it.
  If the drawer is closed but the cube's location cannot be confirmed
  (e.g. cube visible nowhere and drawer closed), judge PASS only if the
  sequence shows the cube entering the drawer before closing; otherwise
  INDETERMINATE.
- **so101-remove-screwdriver** — "Take the black screwdriver out of the
  box and place it on the table." PASS = black screwdriver lying on the
  table, out of the box, gripper released, at end. FAIL = screwdriver
  still in the box or still held.
- **so101-stack-bowls** — "Pick up the white plastic bowl on the right
  and stack it on top of the white plastic bowl on the left." PASS =
  the right bowl rests stacked on/nested in the left bowl as one stack
  at end, gripper released. FAIL = bowls still side by side, or the
  moved bowl left elsewhere.
- **so101-stack-cubes** — "Stack the blue cube on the red cube, then
  stack the white cube on the blue cube." PASS = one vertical stack
  red-bottom, blue-middle, white-top, standing at end, gripper
  released. FAIL = any other arrangement (wrong order, cubes unstacked,
  stack fallen).
- **so101-upright-bottle** — "Pick up the fallen juice bottle and stand
  it upright on the table." PASS = bottle standing upright on its base
  on the table at end, gripper released, not leaning on another object.
  FAIL = bottle still on its side, held, or fallen again by the end.

## 4. BusyBox multitask (armnet/busybox_multitask) — 27 task strings

Judge start-state vs end-state of the NAMED control in the final frames:

- **Push the {green/yellow/white/red} button** — PASS = the named
  button is in its actuated state at end (visibly depressed/switched
  relative to its start state, or its indicator state changed), and no
  other button was the one actuated. FAIL = named button unchanged
  while another button changed, or no actuation visible by the end.
  INDETERMINATE if the named button's state cannot be read.
- **Move the {left/right} slider to position N (1–5)** — PASS = the
  named slider's handle rests at position N at end (read against the
  printed scale). FAIL = handle at a different position. INDETERMINATE
  if the scale/handle position cannot be read in either view.
- **Turn the {left/right} toggle switch up to "On" / down to "Off"** —
  PASS = the named toggle points up (On) / down (Off) as instructed at
  end. FAIL = opposite orientation.
- **Turn the knob to position N (1–6)** — PASS = the knob's pointer
  rests at position N at end. FAIL = different position. INDETERMINATE
  if the pointer cannot be read.
- **Reach {forward / to the right / to the left} and open your
  gripper** — PASS = at end the gripper is open, empty, and positioned
  in the named direction relative to the BusyBox panel. FAIL = gripper
  closed, holding an object, or clearly in a different direction.

## 6. ArmnetBench held-out tasks (Wave 0 addition; same dataset, teleoperated)

Frozen 2026-11-23 before judging. Task strings verbatim from the
dataset's `meta/tasks.parquet`:

- **eye_drops_to_shelf** — "Put the eye drops on the shelf." PASS =
  the eye-drops bottle rests on the shelf surface, gripper released,
  at episode end. FAIL = bottle anywhere else (table, basket, floor)
  or still held.
- **tool_removal** — "Remove the small middle tool (3/8 Extension Bar
  75mm) from the toolbox." PASS = that tool (the small middle one) is
  out of the toolbox, lying on the table, gripper released, at end.
  FAIL = the tool still sits in its toolbox slot, or a different tool
  was removed instead.
- **cable_unclip** — "Remove the power cable from cable holder." PASS
  = the cable is fully out of the holder — no longer retained by the
  clip — gripper released, at episode end. FAIL = cable still seated
  in the holder, or only partially pulled with the clip still holding
  it.

## 7. BusyBox singles (Wave 0 addition)

armnet/busybox_push_green_button ("push the green button") and
armnet/busybox_flip_left_switch_off ("Flip the left switch off") are
judged under §4's button / toggle-switch rules verbatim. Standing flag
from the certified record: these sets are overlap-suspect against
busybox_multitask episodes; they are registered as their own sources
and the flag is carried in the register notes — verdicts are per
dataset entry, as assembled in Wave 0.

## 8. DecisionFacts reserve (Wave 0 addition) — freeze completed at download

Criteria below are written from the tier record's task descriptions;
the exact `meta/tasks.parquet` strings are confirmed from the
downloaded copies and this section is frozen BEFORE any judging of
these sources (no judging starts before that confirmation is logged
in the program report).

- **Pipe_And_Joint_Sorting_Task** (141 eps) — sort a mixed pile of
  plumbing/hardware pieces into three category bins. PASS = every
  piece that starts loose on the table is inside one of the bins at
  episode end, none left loose, gripper released; a piece visibly
  placed in a bin of a different category (where categories are
  visually distinguishable) is FAIL. FAIL = any piece left loose on
  the table or outside the bins at end. INDETERMINATE = bin contents
  or loose pieces cannot be seen in either view.
- **Test_Tube_Nesting_Task** (160 eps, 2 tasks) — judged per episode
  against its own task string: (a) plate seating ("seat a plate in its
  nest"): PASS = the plate rests seated flat in its nest, gripper
  released. (b) test tube ("place a test tube in the tray"): PASS = a
  test tube stands placed in the tray/holder, gripper released. FAIL
  = the object left outside its target or still held at end.

## 5. DecisionFacts Cup Nesting (Physical_AI_SO101_Cup_Nesting_Task)

"Pick up the empty solo cup and nest it inside the other static cup."
PASS = the picked cup rests nested inside the static cup (inner cup's
wall inside the outer cup, rims nested), gripper released, at episode
end. FAIL = cups still separate, inner cup left beside/on top without
nesting, or the static cup was the one moved. INDETERMINATE if the
nesting cannot be confirmed from either view.
