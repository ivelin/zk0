# Study 001 Twin-Replay — Validation Report (Builder C, Phase 4)

**Status: BUILDER SELF-ASSESSMENT — not certification.** Per the workstream's
builder ≠ verifier rule, nothing in this report is certified until an
independent verifier re-checks it from the raw artifacts (code in `code/`,
scenes in `scenes/`, per-episode results in `validation/episodes/`,
aggregate in `validation/validation_results.json`, every parameter change in
`validation/param-changelog.md`).

Suite: **twin-suite v1** · variant seed 20261122 · K=5 variants/episode ·
parameter-table hash recorded in every scene file and variants log ·
physics 500 Hz, trajectory playback at the dataset rate 20 fps, settle 1.0 s.

Scope compliance: only the recorded `gt` trajectories were executed. No
checkpoint inference / policy rollouts were run (Phase 6 is forbidden and
was not touched).

## Headline result

| Task | Included | Reproduced (≥60% of valid variants agree) | Mean variant agreement | Certified variants | ≥80% bar (self-assessment) |
|---|---|---|---|---|---|
| block_stack | 20 | **0 / 20** | 0.00 | 0 / 100 | NOT MET |
| eye_drops_to_basket | 20 | **4 / 20** (ep1757, 1778, 1810, 1811) | 0.28 | 27 / 96 | NOT MET |
| ring_insert | 20 | **1 / 20** (ep253) | 0.06 | 6 / 100 | NOT MET |
| tool_insert | 17 | **0 / 17** | 0.00 | 0 / 77 | NOT MET |
| cable_clip | 20 | **20 / 20** | 1.00 | 100 / 100 | MET on outcome class — with a major caveat (§Cable) |

Excluded from the denominator (Builder B initial-pose confidence low, per
the coordinator's ruling): tool_insert ep630, ep871, ep876. eye_drops
ep1618 + ep1760 ran with the secondary criterion skipped (no final poses).
12 of 485 variant runs were excluded as logged build defects (list in
`validation_results.json`): 8 tool variants in the bad-grounding episodes
ep885/888/892/903 (tool spawns intersecting the enclosure wall — those
episodes' poses are wrong, §Data findings) and 4 eye variants
(ep1780/1783/1808/1814, one each) where the σ-jittered carton spawn fell
off the platform edge during pre-settle.

**Bottom line:** with Builder B's pose quality (σ ≈ 2.5 cm per pose) and an
open-loop kinematic replay, the twin reproduces the recorded outcomes only
for cable_clip (where agreement means reproducing a failure). The dominant
cause is structural, not a single bug: the recorded trajectories are
*closed-loop* executions — the demonstrator corrected placement errors
visually, online — while the twin replays them *open-loop* in scenes built
from pose estimates whose errors (≈2.5 cm per object, compounding to
≈3.5–5 cm at each placement) exceed the pre-registered mechanical
tolerances (block inset 1.1 cm; ring engagement ≲1–2 cm). Three full-set
validation runs were executed (plus extensive calibration work before
them); agreement plateaued, and per the brief I stopped and report the
diagnosis rather than forcing the numbers. §What would have to change
lists what certification of the remaining tasks would actually require.

## Calibration findings (all global, all logged in the param-changelog)

These were resolved BEFORE the validation runs, per the coordinator's
instruction that the FK/gt offset be resolved to <~3 cm at tie points
before validating anything. Final tie-point agreement: **median 1.4 cm,
83% < 3 cm, 89% < 5 cm (n=93 grasp/place dwell tie points)** — the bar
was met.

1. **World-frame misregistration (changelog #1).** Builder B's
   `calib_v2_mat_homography` expresses poses in a frame rotated/reflected
   vs the arm's FK frame (mat near-edge assumed parallel to world x; in
   the top camera the near–far direction runs along image-vertical and
   image-right is −y), and its placement constants (x0, v0) were assumed,
   not fitted. With B's poses as-is, no grasp ever fired in replay.
   Correction: a robust affine px→FK-world map fitted on 93 dwell tie
   points, applied to every B pose as the composite AffineFK ∘ H_B⁻¹
   (`scenes/calibration_fk_affine.json`, "fk-affine-v2"). Residuals:
   identity 12.8 cm median → 1.4 cm median.
2. **Gripper channel inverted (changelog #4).** The recorded gripper
   channel is LOW = jaws closed (holding), HIGH = jaws open — opposite
   the naive reading. Proof without image interpretation: in block ep411
   the arm site transits cube→stack during frames ~124–225, exactly the
   trace's low phase; under the high=holding reading the hold ends before
   the transit and the cube could never reach the stack (it does).
   ring ep7 confirms (transit = low phase 152–223; at frame 177, trace
   1°, the ring is visibly pinched in the jaws mid-air). Closedness
   normalisation stays per-episode (p5/p99), direction reversed.
3. **Elevated-object parallax (changelog #3/#3b).** B's mat-plane
   homography displaces points above the mat (oblique top camera). The
   eye-drops carton (centre 10.75 cm up, on the platform) carried a
   uniform (−0.014, −0.093) m error, measured on all 20 pick dwells.
   Corrected for the carton initial pose directly, and via a fitted
   per-height term k·(h − 0.016), k = (−0.153, −1.016), for B-sourced
   eye platform/basket/carton-final and the block stack finals. A global
   height-term refit of the whole frame map was tried and rejected (it
   degraded the other tasks: 2.2 cm vs 1.4 cm overall median).
4. **Attach semantics (changelog #2/#5).** Builder A's grasp spec, read
   literally (weld at the 50%-crossing instant if the object centre is
   within 3.0 cm in 3D), never fires for top grasps — at the crossing the
   gripper is still approaching and the 3D distance is dominated by the
   inherent vertical site-to-centre gap. The implemented device
   (documented interpretation, flagged for the verifier): attach at the
   closing excursion's *dwell*, gate = planar distance from the
   gripperframe site to the object's *material* ≤ 3.0 cm (Builder A's
   value; centre-distance bounds per object class for elongated
   objects), weld relpose snapshotted at the dwell frame and computed
   live (re-grasps after a drop use the dropped pose), plus
   release-recency ineligibility so the demos' final *empty* close
   cannot re-attach to the just-placed object and drag it home (run 1:
   11/20 ring episodes ended still-welded, rings dragged off the peg).
   Release: first frame closedness ≤ 30% while attached (Builder A's
   value).

## Data findings about Builder B's deliverables (for the verifier / B)

- **finalize JSON ≠ B's own detections (block_stack).** 27/80 block poses
  in the episode JSONs disagree with B's `detections_px.json` by >15 px
  (e.g. ep411 purple initial inverts to px (490,330) vs the actual
  detection (447,482) — an 11 cm spawn error). Block poses were therefore
  taken from the detection pixels (which fit FK dwells at 1.4 cm).
  Ring JSON poses agree with detections (0 mismatches); tool JSON
  (grounding) fits dwells at 1.9 cm while tool segmentation detections
  are 19.7 cm off — both kept as-is.
- **Block final detections are unreliable.** In final frames the colour
  masks catch the mat's printed lettering and arm parts (e.g. ep411's
  "green final" sits on the ROBO lettering). The stack reference uses the
  final detection nearest the base cube's initial position (the stack
  forms on the base). Even so, several demos (ep411 is the clearest) do
  not physically end in a stack at all: the manipulated cube is placed at
  a target point and the base cube is never engaged, while the recorded
  label is success. The pre-registered mechanical criterion cannot agree
  with such labels in any faithful twin.
- **Tool grounding failures (7/17 included episodes).** For ep697, 779,
  784, 885, 888, 892, 903, B's tool poses sit 14–22 cm from where the
  recorded trajectory actually grasps (FK dwell arbitration) — the
  grounding picks landed on other scene content (ep885/888/892/903 spawn
  at (0.03, 0.11), at the arm base, intersecting the enclosure wall).
  These episodes cannot reproduce regardless of the twin; B marked them
  medium confidence. Named here so the denominator discussion is
  explicit: with these 7 removed, tool would still be 0/10 (5 attach and
  place imperfectly, 5 miss the gate at 3.1–5.2 cm), so the conclusion
  for tool_insert does not hinge on them.
- **Cable block poses** are B's patched grounding picks (segmentation
  centroids rejected by B); no independent arbitration exists (the block
  is never grasped). Kept as delivered; flagged.

## Per-task results and failure taxonomy

### block_stack — 0/20 reproduced, mean variant agreement 0.00
Attaches fire in 20/20 nominal runs and the carried cube is placed close
to the recorded stack point (nominal secondary: median planar error
6.0 cm; 3/11 reportable episodes ≤3.0 cm — the criterion is only
reportable when a cube ends clearly upper). But the stack itself almost
never forms: final upper-vs-lower cube offsets median |dx| 4.8 cm,
|dy| 3.8 cm (tolerance 1.1 cm); only 4/20 end at stacked height.
Causes, in order: (1) placement error ≈ √2 × pose σ (grasped cube's
estimate error enters the weld anchor; the base cube's estimate error
moves the target) ≈ 3.5–5 cm vs a 1.1 cm inset — structural; (2) a
subset of demos do not physically stack (§Data findings); (3) weld
anchor residual (the snapshot offset is the *estimated* grasp geometry,
not the true jaw geometry). Fragile: all 20 episodes.

### ring_insert — 1/20 reproduced (ep253), mean variant agreement 0.06
The drag-off artefact is fixed (still-attached at end: 11/20 in run 1 →
1/20). Nominal planar-to-peg: median 3.4 cm; 4 episodes ≤0.2 cm (device
and poses essentially exact), a tight cluster at 2.9–5.0 cm, one outlier
17.5 cm. The engagement tolerance (peg through the ring hole, ≲1–2 cm)
vs the placement-error budget (ring σ 2.5 cm + peg σ 2.0 cm + anchor
residual ≈ 3.5 cm) explains the cluster: most rings land ON the board,
centred a few cm from the peg. Variant jitter of the peg (changelog #6)
did not rescue episodes (1/20 vs 0/20) because the landing point is set
by the trajectory + anchor, not by the peg draw. Secondary criterion
(not engaged): median 3.8 cm, 2/16 within the 2.0 cm tolerance.
Fragile: 19/20 (all but ep253).

### eye_drops_to_basket — 4/20 reproduced, mean variant agreement 0.28
Reproduced: ep1757, 1778, 1810, 1811. Nominal: attach ≥1 in 15/20,
carton centre inside the basket in 7/20. Secondary is bimodal: 8
episodes at 7–12 cm (placed near the basket / basket-edge cases) and 10
at 27–43 cm (carton never left the platform — the platform-to-basket
distance; those episodes' grasps either never attached (5/20) or
attached and released without a carry). The basket is a single global
read for all episodes; B's own carton finals sit at median (−0.3, +2.7)
cm from its centre, so the basket reference is not the dominant error.
Fragile: 16/20.

### tool_insert — 0/17 reproduced, mean variant agreement 0.00
Three disjoint groups (nominal): 5 episodes attach and carry (622, 683,
684, 859, 883) but the insert misses (final axis angle median 42° vs the
20° tolerance; the tool is released above the tray and rolls — a capsule
on a flat tray; my tray has no slot groove geometry, §Ambiguities);
5 episodes miss the grasp gate marginally (min shape distance
3.1–5.2 cm vs the 3.0 cm gate — grasp-point geometry/pose noise at the
spec boundary); 7 episodes have wrong B poses (§Data findings) and the
site never comes within 14 cm of the spawned tool. Secondary: median
18.4 cm, 0/13 within 3.0 cm. Fragile: all 17.

### cable_clip — 20/20 reproduced (failure agreement), mean agreement 1.00 — READ THE CAVEAT
The recorded outcome for all 20 episodes is FAILURE, and the twin also
fails all 20: the cable is never seated in the clip in any variant
(100/100). The coarse capsule-chain + weld approximation is therefore
consistent with the recorded outcome class. **Honesty breakdown of the
failure mode (nominal runs):** the gripper attaches to the cable in
16/20 episodes and the cable is genuinely manipulated toward the block —
nearest cable segment to the clip block at settle: median 9.7 cm;
4 episodes end with a segment 1.2–2.2 cm from the block (reaches the
block's vicinity but does not seat — the recorded failure mode,
qualitatively); most episodes end 4–13 cm away. **Artefact:** in 6/20
nominal episodes (ep1905, 1939, 1959, 1960, 1969, 2068) the ball-joint
chain numerically destabilises under the weld (segments ejected 1.2–13 m
from the workspace; a MuJoCo QACC NaN warning is emitted). The outcome
class still matches (failure), but in those episodes the twin's failure
is a numerical explosion, not a physical non-seating — they should not
be counted as faithful failure reproductions. Discounting them, the
faithful agreement is 14/20 episodes with the cable remaining in the
workspace and unseated. The clip itself is modelled as two static jaws
(no closing clip mechanism), and the cable layout is B's descriptor-grade
coil (σ 8 cm) — the pre-registration declares this task outcome-class
only, and that is all the twin supports.

## Secondary criterion — distributions (nominal variant, not just binarised)

Builder B's calibration accuracy is ±2.5 cm (1σ, near/mid mat) to ±4 cm
(far edge); misses near the tolerances should be read against that.

| Task | n | Median | p25–p75 | Within tolerance |
|---|---|---|---|---|
| block_stack (tol 3.0 cm) | 11 reportable | 6.0 cm | 3.2–13.9 | 3/11 |
| eye_drops (tol 3.0 cm) | 18 (2 skipped) | 27.1 cm (bimodal: 8 at 7–12, 10 at 27–43) | 10.9–34.9 | 0/18 |
| ring_insert (tol 2.0 cm, when not engaged) | 16 | 3.8 cm | 3.3–4.3 | 2/16 |
| tool_insert (tol 3.0 cm) | 13 reportable | 18.4 cm | 13.0–22.5 | 0/13 |
| cable_clip | exempt (outcome class only) | — | — | — |

## Grasp-device artefacts (declared, per the honesty rules)

- Weld preserves the dwell relative transform; it does not centre the
  object in the jaws. Placement error therefore includes the difference
  between the estimated-pose grasp geometry and the true jaw geometry.
- Welded objects can be pushed partly into the mat during a place and
  rest slightly sunk (observed up to ~1 cm in synthetic tests; in
  validation, ring bottom-above-board median ≈ 0.0 cm, so the effect is
  small on final states).
- Attach uses a snapshot relpose: if the gripper moved between the dwell
  frame and the attach trigger, the object snaps by that gap
  (`snap_gap_m` logged per attach; typically ≤1 cm).
- A weld slot limit of 4 attaches per object (2 for cable segments)
  exists; episodes needing more grasps of one object log missed attaches
  as no_attach events.
- Cable chain numerical explosions in 6/20 cable episodes (§Cable).

## Ambiguities flagged (not silently reinterpreted)

1. **Grasp threshold scale.** Builder A's "50% closed (normalised
   range)" cannot be absolute (object-dependent jaw stops; ring/tool
   would never attach). Implemented as per-episode normalisation
   (p5/p99) — and the channel direction turned out inverted
   (§Calibration #2). Verifier to rule on the normalisation choice.
2. **Attach gate geometry.** The spec's centre-distance/crossing-instant
   reading is unimplementable (§Calibration #4); the dwell + shape
   distance + per-class centre bounds are my documented interpretation.
3. **Playback rate.** Pre-registration says 30 fps in one place,
   physics-parameters and the dataset say 20 fps; trajectory lengths
   match 20 fps. 20 fps used.
4. **block_stack secondary** is reported for the upper (placed) cube.
5. **tool slot size** (110×32 mm) is a local choice in twin_params; B
   supplied the slot pose only. The tray is modelled without a slot
   groove — if the real tray's slot is a channel that constrains the
   tool's axis, the twin's tool is freer to roll than the real one.
6. **Ring recess count** 6 (B, counted in frames) vs 5 (A's inventory) —
   noted for the verifier; immaterial to the twin (only the peg and the
   ring's start recess matter, and the start pose is B's directly).
7. **Cable weld target**: Builder A's "cable welds to nearest segment"
   is implemented via the same state machine (nearest segment by shape
   distance).

## Certification self-assessment (builder's own, NOT certification)

- cable_clip: outcome-class bar met (20/20 ≥ 80%), but 6/20 nominal runs
  are numerical explosions — I would not sign this as a faithful twin
  certification; as an outcome-class screen it passes.
- eye_drops_to_basket: 4/20 reproduced — bar not met.
- ring_insert: 1/20 — bar not met.
- block_stack: 0/20 — bar not met.
- tool_insert: 0/17 — bar not met (with 7 episodes failing on B pose
  quality before twin physics is even exercised).

## What would have to change for the other tasks to certify

(Findings, not commitments.) (a) Pose accuracy: the placement tolerances
need σ ≲ 1 cm poses — e.g. FK-anchored pose estimation using the
recorded trajectories themselves (the dwell tie points measure object
positions to ~1.4 cm) or a calibrated multi-view rig. (b) The replay is
open-loop; the demos were closed-loop. A twin that re-executes with
any form of visual servoing on the final approach is a different device
than the pre-registered one and would need its own pre-registration.
(c) B's finalize/grounding pipeline needs a consistency pass (JSON vs
detections cross-check, lettering-mask contamination in final frames,
tool grounding outliers) — the cross-check script used here is
re-runnable (`validation/` diagnostics described in the changelog).

## Phase-6 readiness notes

Per Amendment 1: certified variant sets per episode are recorded in
`validation/validation_results.json` (`certified_variants`); episodes
below 60% agreement are flagged `fragile` there and are excluded from
the Phase-6 suite by the amendment's rule — on this run that excludes
all block/tool episodes, 19/20 ring, 16/20 eye, and none of cable
(cable's certified set should be read with the explosion caveat above:
the 6 explosion episodes' variants are outcome-agreeing but not
faithful scenes for policy rollouts). Nominal (v0) top-view renders for
all 97 included episodes are in `validation/episodes/*_v0_top.png`.
The Phase-6 plan must specify the dual judging (mechanical criteria +
blinded frontier-model pass/fail on rollout recordings) per Amendment 1
§A2 — that plan is the coordinator's/Phase-6 builder's, not produced
here.

## Monthly realism re-audit — design (Amendment 1 deliverable)

- Cadence: monthly, first run one month after suite v1 freezes.
- Sample: all 93+ tie points re-derived from the frozen records +
  20 episodes (4/task) drawn at random from the included set.
- Checks: (1) tie-point residual distribution vs the suite-v1 baseline
  (median 1.4 cm) — drift >1 cm median triggers investigation;
  (2) re-run the sampled episodes' nominal variants and compare outcome
  classes and secondary errors against the frozen
  `validation_results.json` — any outcome flip is investigated and
  logged; (3) re-examine 5 sampled scenes against their source frames
  with the then-current best pose-estimation method; scenes whose
  geometry is shown unrealistic (parameter outside Builder A's sourced
  ranges, or pose error >2σ) are fixed in a new suite version (v2) or
  retired, never silently edited in v1; (4) physics-parameter spot
  checks (friction/mass) against any new sourced measurements.
- Output: a dated audit note appended to this report's folder; suite
  version increments only through the same builder + independent
  verifier path as v1.

## Reproducibility

Code: `code/build_scenes.py`, `code/replay.py`, `code/run_validation.py`,
`code/twin_params.py` (parameter table; hash in every scene header).
Scenes regenerate deterministically from Builder B's JSONs +
`scenes/calibration_fk_affine.json` + `scenes/detections_px.json` with
seed 20261122. Full-set run logs: `validation/run1.log`, `run2.log`,
`run3.log` (run 3 is the reported set; run1/run2 result files kept as
`validation_results_run1.json` / `validation_results_run2.json`).
Asset provenance and reuse decisions: `docs/asset-inventory.md`.
