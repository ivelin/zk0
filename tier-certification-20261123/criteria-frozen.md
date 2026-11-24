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

## Amendment 3 — contested-episode narration fields (founder, 2026-11-23 21:50 CT)

For any episode whose reads CONTEST (primary vs re-judge vs Grok vs
Gemini disagreement, or an INDETERMINATE in any leg), the judge read
must include, beyond the verdict:
(a) a chronological trajectory walkthrough — what the arm/object do,
in order, across the episode; and
(b) anomalies/insights observed — anything unusual, ambiguous, or
diagnostic noticed during the episode.
Purpose (founder): these fields let the founder verify the judges
actually perceive the videos. Applies to all NEW contested-episode
judging from this date (Wave 1+ sources and any later Wave-0 judging);
verdicts already delivered are not re-opened. A separate narration
pass over the held-out contested set runs from
~/workspace/founder-tiebreak-20261123/narrations/ (outside this
program's register work).

## Amendment 4 — GROK TRUST SUSPENDED (founder ruling, 2026-11-23 21:53 CT)

Founder, verbatim: "We can no longer trust Grok until it gets a major
AI model upgrade. Let's check back next month." Trigger: Grok's lone
PASS reads on the contested held-out cable episodes described the
cable as seated when the full videos show it loose on the mat; the
founder overruled Grok on ep2020/2042/2046/2092 from the full videos.
Effective immediately:
1. Grok is NO LONGER a certifying leg. Existing Grok reads stay in the
   register, flagged DISTRUSTED in the row notes; the data is not
   deleted.
2. No new Grok judging runs. Certification proof rests on Muse +
   Gemini + the founder's eyes.
3. Contested-episode resolution must not cite Grok agreement as
   support. Where Grok was the lone dissent and the other legs agree,
   the episode is not treated as open on Grok's account; its note reads
   "Grok dissent — distrusted per founder ruling 2026-11-23" (applied
   2026-11-23: 99 register rows cleared of contested on this basis;
   contested 123 → 24). Founder adjudication still governs the
   held-out cable five.
4. Trust re-check scheduled 2026-12-23; restoration only on the
   founder's yes.
Consequence: the pending Grok legs for rc-stack-cubes / bb-push-green
/ bb-flip-left are CANCELLED (not merely blocked by the exhausted
balance); those sources certify on Muse + Gemini + founder, and
their review packs no longer wait on Grok.

## Amendment 5 — GRIPPER-RELEASE STANDARD (founder, 2026-11-23 22:05 CT)

A certified PASS requires the gripper VISIBLY OPEN / released at
episode end. An episode ending with the gripper still closed on the
object is INDETERMINATE, not pass, regardless of how the object sits.
Origin: the founder's revision of his held-out ep489 ruling — "Gemini
was correct. The gripper was still closed at the end. I suppose we can
say we don't know if it passed for sure... We need to see grippers
open ideally."
Application: (1) all NEW judging from this date, including the agy
round-robin leg (clause appended to every un-run agy prompt; the one
in-flight run at the change, gemini_spa-14, kept its original prompt
and is covered by the retroactive scan); (2) retroactive flagging —
original verdicts are NEVER overwritten; flagged rows carry
release_standard_flag = "INDETERMINATE-under-release-standard" in a
separate column and form a founder-review queue
(release-standard-queue-20261123.md). Scan method: PASS rows whose
judging evidence text matches release-risk patterns (gripper
still/closed/clamped/holding/touching, not/no release, borderline
flags) — a starting set, per the founder; text-silent cases are not
caught by the scan.

## Amendment 6 — UNANIMOUS-TRUST RULE (founder, 2026-11-23 22:22 CT)

The approved judges are Muse and Gemini (Grok is set aside until
the December re-check). When Muse and Gemini agree unanimously, the
PASS or FAIL is TRUSTED — certified without founder review. The
founder sees ONLY: (a) Muse≠Gemini disputes (including INDETERMINATE
on either side), and (b) gripper-release-standard flags.
Applied 2026-11-23: held-out 100 — 92 unanimous-trusted (78 marked
CERTIFIED by unanimous agreement; 8 already founder spot-check
approved; 4 already founder-adjudicated FAIL; the 2 release-flagged
unanimous passes, block 453/514, held in the founder queue under (b));
8 disputes, of which 489 (INDETERMINATE) and 2020 (FAIL, removed) are
closed — OPEN founder queue: block 499, block 557, tool 784, tool 824,
eye 1643, eye 1760 + release-flagged block 453, block 514.
Training 250 — 244 CERTIFIED by unanimous agreement (incl. ep604 as a
certified FAIL); 4 founder spot-check approvals stand; 2 pending:
ep573 (release flag) and ep2201 (verifier INDETERMINATE / qualified
PASS). Register sources get the same split as their Gemini coverage
lands (standing merge duty).

## Amendment 7 — JUDGING-INPUT STANDARD (founder, 2026-11-23 22:39–22:40 CT)

For the held-out re-judge and ALL future judging in this program:
(1) if the advanced agy models can watch full video, judge from FULL
VIDEO; (2) otherwise judge from DENSE contact sheets with this
recipe: a few frames from the START of the episode (scene setup) +
rising sampling density toward the END + the final frames always
included (~16 timepoints; no more uniform 6-frame sheets anywhere).
The input mode of every judging batch is recorded per episode
alongside the model stamp.
Application note (coordinator, 2026-11-23): the video test (ep824,
neon-xhigh) PASSED — agy accepted the full mp4 and read it correctly.
Video is therefore the standard for the dispute tier and the held-out
re-judge tier. Bulk register legs (hundreds of episodes per source)
use the dense-sheet recipe — full video at ~2.5 min/episode/call
cannot cover ~1,800 episodes in the available windows; this scoping
is stated in the morning report for the founder's countermand.
Input-mode stamps: register/audit records carry the mode per episode
("full-video" or "dense-sheet-16") next to the model.

## Amendment 8 — JUDGE NAMING CORRECTION + independence restated (founder, 2026-11-23 22:46 CT)

The primary judge leg in this program is Muse (Muse Wonder, Meta) —
the coordinator's subagent instances — NOT Anthropic Claude. Anthropic
Claude enters this system ONLY via the agy menu (Opus 5.5 / Sonnet
5.5). Judge-leg labels in the register and program files are corrected
from "Claude" to "Muse (Muse Wonder)" at the 2026-11-23 commit so the
record stays unambiguous (historical verbatim quotes are unchanged).
Independence, restated: the two certifying judges must be different
model families from MUSE WONDER (the primary). Per the founder's rule,
Gemini-family Neon/Carbon count as the second judge; Opus 5.5 reads
are recorded as third reads until the founder rules on whether Opus
counts as an independent second judge (factually it is a different
family from Muse Wonder — the question is open, not decided here).

## Amendment 9 — JUDGE NAMING TAXONOMY, FINAL FORM (founder, 2026-11-23 22:54 CT)

Three families, always distinguished in records:
(1) "Muse (Muse Wonder)" — Meta's model: the coordinator, all subagent
judges/verifiers/builders (the primary leg).
(2) "Gemini (Google)" + exact model (Neon / Carbon / Flash) — the
Google leg via agy.
(3) "Claude (Anthropic)" + exact model (Opus 5.5 / Sonnet 5.5) — ONLY
where an Anthropic model served via agy actually produced a read.
Register status strings updated to the full forms at the 2026-11-23
overnight commits; new model stamps use the taxonomy (e.g. "Gemini
(Google) gemini-4-neon-xhigh (agy 1.5.15, full-video)").
