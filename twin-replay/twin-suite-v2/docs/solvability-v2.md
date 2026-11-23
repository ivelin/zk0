# Twin-Suite v2 Stage 1 — Reference Controllers + Solvability Validation (MuJoCo)

**GENERATION: ORACLE-GUIDED** — every reference execution in this report
consumes ground-truth scene state by design (object poses, fixture
positions, and the recorded demo's joint trajectory were available to the
executor's planners). Per pre-registration Amendment 2 §A2.2 and SSOT
Decision 55, these executions are **scene-solvability proofs, never
policy measurements**. No gated claim rests on the reference executor's
skill.

- **Date:** 2026-11-23 (runs 2026-11-22 22:3x – 2026-11-23 01:57 CT)
- **Scope:** development subsets only (pre-reg Appendix A), nominal
  variant, 4 scenes per task × 5 tasks = 20 scenes. The full 96-scene
  validation runs later with the controls battery, controllers unchanged.
- **Code:** DGX `/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference/`
  — `refcore.py` (md5 `96aa928072e5487b1fe1f077be293132`),
  `run_reference.py` (md5 `a9d29bdb8f07c8b55181d10295f6d457`),
  `batch_dev.sh`, probe scripts `probe1–36.py` (diagnostic evidence),
  per-scene records `reference/results/<task>__ep<id>.json`.
- **Instrument:** padded v2 arm
  (`scenes-v2/arm/so101_new_calib_padded.xml`) in the v1.1 twin scenes
  (arm include swapped to the padded arm; **no v1.1 file modified** —
  v2 scenes are generated copies under `scenes-v2/`). Mechanical
  criteria are v1.1 `replay.evaluate()` verbatim. MuJoCo 4.0.1,
  CPU-only, `nice -n 10`.
- **Canonical run:** one frozen-code batch of all 20 scenes under the
  job guard (`runs/solvability-batch1/`, launch via
  `launch_dgx_job.sh`, mode logonly, `JOB_LOCK=0`). Earlier development
  iterations are described where they produced a finding; verdicts in
  this report are the batch's.

## 1. Instrument parameters (measured, recorded per scene)

**Pinch-centre frame.** All grasp/place targeting uses the pinch centre:
the midpoint of the two finger-pad geom centres (`finger_pad_fixed`,
`finger_pad_moving`), re-evaluated at the current grip angle on every IK
iteration. The `gripperframe` site is **not** usable for targeting: the
measured site→pinch-centre offset across the batch's grasp records has
components ranging over ±33 mm and magnitude typically 30–40 mm,
varying with gripper orientation and grip angle (records:
`pinch_offset_from_site_m` in every grasp log entry). This is recorded
as an instrument parameter of the v2 arm; any controller (reference or
policy-side tooling) that targets the site inherits a 2–4 cm error.

**Pad face gap vs grip angle** (kinematic measurement on the padded
model): pad-centre distance along the pinch axis 77.3 mm at +1.45 rad →
13.7 mm at −0.17 rad; face gap ≈ centre distance − 5.5 mm, so the
**minimum achievable face gap is ≈8 mm** at the gripper joint limit
(−0.1745 rad). Consequence for cable_clip in §5.

**Joint limits** are the model's `jnt_range`, clipped inside the IK at
every iteration and never relaxed for any execution in this report
(including the block_stack place, §2).

**Torque gate.** Hold/place IK solutions must satisfy a static
gravity-torque check (`qfrc_bias` at rest ≤ 3.0 N·m per arm joint;
actuator forcerange ±3.35 N·m). Transit/approach solvers deliberately
do **not** apply this gate: near-base transit poses legitimately exceed
3.0 N·m static torque on a joint (deep-fold region) and gating them
made reachable approach points unsolvable (measured regression on
block_stack ep411 stage-1, since fixed and documented in code).

**IK.** Damped least squares on the 5 arm joints (pinch position +
gripper-body orientation), step clip 0.04, damping 0.01, weak posture
anchor to the seed (prevents nullspace branch jumps). Solvers:
multi-start branch search (dense structured grid + seeded random
restarts), position-only screening then full polish, slerp
continuation paths, and a continuity-first mode for held-object moves
(branch jumps swing eccentrically held objects by centimetres —
measured on ring_insert).

## 2. The block_stack place-reachability resolution

The inherited failure: the scripted pass grasped and carried the cube
but the place IK demanded wrist_flex ≈107° against the ±95° limit and
stalled ~25 mm short. Findings, in order:

1. The place point that fails is the **near-base cube's** location
   visited *as a place target with a held cube*. A 500-seed position
   search shows the pinch point at the near-base cube +58 mm z is
   outside the position-only reachable set entirely, and no yaw-only
   orientation reaches the near-base place point at any height. The
   reachable orientation there is a ~105°-off-identity yaw+tilt
   combination — not a fingers-down place pose.
2. **Resolution: invert the cube roles.** Grasp the near-base cube
   (the near-base point is then visited only by the grasp, which the
   reachable set does cover — via exotic base-pan branches and the
   achievable tilted orientation), and place it onto the **far** cube,
   whose place point sits inside the reachable shell with ordinary
   orientations. `place_scan` tries orientation candidates in
   preference order (yaws → tilts → free-orientation from
   position-only IK with iterated offset) and takes the first feasible
   under honest limits.
3. Joint limits were never relaxed. The two block_stack scenes whose
   geometry matches this pattern pass (§6); ep411's near-base cube is
   the extreme case and fails at the *lift* (§7), not the place.

## 3. Controller design (common pattern)

All pad-based controllers share one skeleton (`refcore.Ref` +
`run_reference.py` per-task drivers):

- **Stage 1 approach:** raise straight up from the start pose, then
  position-only (orientation-free) transit above the object —
  continuation IK, else direct position-only solution. Near-basin
  branches are preferred (exotic wrist-roll-flipped solutions solve
  the point but their descend swipes the object — measured).
- **Grasp:** all-or-nothing oriented descend — the full-IK solution to
  the grasp point is computed kinematically first (strict link:
  ≤6 mm / ≤6°, torque-gated; then multi-seed link search; then the
  legacy relaxed path: slerp path → branch search → position-only
  continuation + orientation polish). Executed only when a solution
  exists, so a failed grasp leaves the object untouched. Then:
  arrival verification (near-base gravity lag leaves the arm
  centimetres short; re-solve and re-execute slower, ≤2×), oracle
  centring correction (≤2 iterations, 3 mm tolerance), and a **slow
  close (2.4 s)** — a 1.2 s close flicked the 10 mm tool capsule
  60 mm out of the pads (measured). Hold = stall − 0.03 rad. A grasp
  counts only with both-jaw contact (`gripper` + `moving_jaw`).
- **Carry/place:** held offset `o_body` (object centre in gripper-body
  frame, measured at the pinch) is re-measured after the lift and
  before each place solve; place targets are solved continuity-first.
  `final_servo` polish (orientation-held lift, fresh offset, re-solve;
  place-scan fallback) is applied for block/tool places.
- **Release:** open, retreat up with orientation held, settle 1 s,
  then v1.1 `evaluate()`.

Per-task specifics are in §4–§5 and the register notes in §7.

## 4. Per-task controller notes

**block_stack.** Role inversion (§2); grasp dz +0.008 (pinch at the
cube's upper-middle — pinch at exact centre height is physically
unreachable because the pad tips would pass 8 mm below the table);
place onto the far cube's top centre; final_servo polish (≤2 iters).

**eye_drops_to_basket.** Grasp across the carton's 35 mm width
(yaw 90° first), dz +0.006, single-link descend (the hover/rail
intermediate points proved *harder* to solve than the grasp point
itself — wrist-limit region, measured on ep1630). Place is a **low
place, not a drop**: release at carton centre z 0.04, inside the
basket walls (floor 0.003, wall top 0.07). The earlier 0.115 release
dropped the light carton 77 mm and it came to rest leaning/perched at
z 0.048 outside the criterion.

**ring_insert.** Tilted-finger grasp candidates (55–65°) with a
**test-lift verification** per candidate (a grasp that forms but slips
on lift is rejected and the next candidate tried), plus an outside-rim
fully-open descend candidate. Seat: staging at peg-top height,
small-step closed-loop **pinch servo** for planar alignment
(≤6 mm steps, current orientation, ring centre re-measured per step),
a tilt-levelling attempt, and a stepped servo descend. This is the
task that does not validate — see the finding in §7.

**tool_insert.** Grasp at the tool's **axis height** (dz +0.004; the
capsule axis sits at z≈0.005) with strong tilts in the candidate set
(≤65°) so the pad tips clear the mat sideways — a vertical grasp at
axis height would plough the tips through the mat, and a grasp at
dz +0.020 pinches 20 mm above the axis and the tool slides out on
lift. Place into the tray slot centre (fixture-derived), final_servo
polish.

**cable_clip.** See §5 (weld device).

## 5. cable_clip: why the reference executor is the v1.1 weld device

A pad pinch of the cable is **physically impossible with this
gripper**: the cable segment is Ø5 mm and the minimum pad face gap is
≈8 mm at the joint limit (§1) — the pads close past the cable without
ever gripping it. The reference execution therefore uses the v1.1
twin's own kinematic weld device (`weld_cable_seg_<i>_*` equality
slots present in every cable scene): the actuated arm carries the
gripper to the target segment (nearest the clip block), the weld is
activated with the v1.1 attach convention (relative pose computed
from gripper-body and segment frames), the arm carries the segment
to the clip channel under actuation, seats it, the weld is released,
and the scene settles before `evaluate()`. This is still an **executed
pass in the twin**: the segment's trajectory is produced by the
actuated arm moving through the scene's physics, the channel
criterion is the unmodified v1.1 geometric criterion (segment centre
in channel volume + clip_block contact, solver-failure excluded), and
no state is teleported — the weld is the twin's registered manipulation
device for this task, the same device class v1.1 itself defines for
cable handling. The two cable failures (§7) are genuine misses of
that criterion by the executed trajectory, not device artefacts.

## 6. Per-scene verdicts (canonical batch, nominal variant)

| Task | Ep | Verdict | One-line outcome |
|---|---|---|---|
| block_stack | 411 | FAIL | grasp formed after retries; lift failed, cube at z 0.017, dragged to (0.065, −0.057) |
| block_stack | 453 | **PASS** | cube_b onto cube_a, stacked at (0.278, 0.050, 0.044) |
| block_stack | 489 | **PASS** | cube_a onto cube_b, stacked at (0.259, 0.103, 0.045) |
| block_stack | 498 | FAIL | place executed; stack settled ~8 mm off — outside the 25%-inset footprint criterion |
| eye_drops_to_basket | 1630 | **PASS** | carton in basket at (0.089, 0.154, 0.020) |
| eye_drops_to_basket | 1746 | **PASS** | carton in basket at (0.091, 0.153, 0.020) |
| eye_drops_to_basket | 1783 | FAIL | repeated closes displaced the carton; it ended off-platform at (0.432, −0.145) |
| eye_drops_to_basket | 1798 | **PASS** | carton in basket at (0.087, 0.152, 0.020) |
| ring_insert | 7 | FAIL | seat blocked; ring left at (0.311, −0.039, 0.081) vs peg (0.265, −0.056) |
| ring_insert | 33 | FAIL | lift failed; ring slipped to board contact, z 0.037 |
| ring_insert | 167 | FAIL | seat blocked; ring at (0.301, 0.008, 0.147), peg (0.272, 0.006) |
| ring_insert | 195 | FAIL | seat blocked; ring at (0.291, 0.013, 0.160), peg (0.258, 0.020) |
| tool_insert | 683 | **PASS** | tool in slot at (0.234, 0.059, 0.046) |
| tool_insert | 697 | FAIL | grasp attempts displaced the tool to (−0.012, 0.130); ungraspable there |
| tool_insert | 852 | FAIL | lift ok (z→0.047); tool lost during carry, ended on mat at (0.065, 0.041) |
| tool_insert | 883 | FAIL | grasp attempts displaced the tool to (0.315, 0.000); ungraspable there |
| cable_clip | 1969 | **PASS** | segment seated in channel (weld device, §5) |
| cable_clip | 2020 | **PASS** | segment seated in channel (weld device, §5) |
| cable_clip | 2125 | FAIL | segment settled at z 0.069 vs channel 0.061, ~19 mm off in y under chain tension |
| cable_clip | 2130 | FAIL | segment settled at z 0.066 vs channel 0.061, ~17 mm off in x under chain tension |

**Totals: 8 of 20 scenes validated** (executed pass satisfying the
v1.1 mechanical criteria at settle).

## 7. Failure diagnoses (every failure, no silent exclusions)

- **block_stack ep411 — near-base grasp/lift limit.** The moving cube
  sits at (0.066, −0.023), deep in the near-base shell: the +80 mm and
  +50 mm approach points are outside the reachable set (500-seed
  search; §2), the +30 mm approach is reachable only in exotic
  branches, and the successful pinch forms on the cube's edge; the
  lift target from that pose is IK-infeasible and the cube stays at
  z 0.017. This is a controller/arm-geometry limit at this scene's
  pose, recorded as not validated by this executor. It is **not**
  evidence the scene is unsolvable (a human-in-sim pass under A2.2
  route (c) remains possible).
- **block_stack ep498 — marginal place settle.** The full sequence
  executed; after release+settle the upper cube rests ~8.1 mm off the
  lower cube's centre, just outside the 25%-inset footprint criterion
  (allowance ≈7.5 mm). A near-miss of the criterion, not a reach or
  grasp failure.
- **eye_drops_to_basket ep1783 — grasp displacement cascade.** The
  carton's pose in this scene makes every candidate orientation close
  on the carton edge (16 grasp records, all stall −0.159 = pads fully
  closed past the carton); each close nudged the carton until it left
  the platform (final (0.432, −0.145, 0.017)) and no further grasp was
  possible. Controller limit: the retry loop re-reads position but
  has no recovery once the object leaves the reachable platform area.
- **ring_insert (all 4) — the seat is beyond this scripted
  controller; this is a finding, reported per the work order, not
  forced.** Grasp and lift work (tilted grasp, test-lift verified;
  ring lifted z 0.021→0.12 in 3 of 4 scenes; ep33 slipped on lift).
  The ring is held at its rim and **rotates in the pads during every
  carry**, ending tilted **39–50° from horizontal** (measured from the
  ring body frame at staging). Consequences, all measured on ep7:
  the grasp-time held offset is stale by centimetres at staging;
  the small-step staging servo does converge planar error
  12.3 → 1.9 mm, but the tilted hole axis catches the peg tip even at
  1.9 mm (clearance is 3 mm), the descend blocks at peg-top height,
  and further servo steps drag the ring across the peg top.
  Tilt-levelling by gripper rotation is IK-infeasible at the staging
  pose (wrist_flex already at the 95° limit; even 10° steps fail).
  An outside-rim symmetric grasp variant reduced tilt only to ~40°
  and worsened staging error. **Conclusion: with honest joint limits
  and this arm/gripper, the scripted reference controller cannot seat
  the ring within the 3 mm criterion.** Per A2.2 these scenes are
  *not validated by this executor* and are excluded from the
  validated denominator with this reason logged; the exclusion is
  about the executor, not the scenes — route (c) human-in-sim
  teleoperation, or a controller that re-grasps the ring flat at
  staging, remain open routes to validate them. A task with no
  validated scenes has no gate (A2.2) — ring_insert currently has
  **no validated dev scenes** and must not be gated on this
  executor's evidence.
- **tool_insert ep697 / ep883 — grasp ejection + displacement.** The
  10 mm, 70 g capsule is ejected or rolled by early grasp attempts in
  these poses (ep883: 17 grasp records); once displaced to
  (−0.012, 0.130) / (0.315, 0.000) it rests against fixtures at the
  workspace edge where no candidate orientation forms a both-jaw
  pinch. Same retry-recovery limit as eye ep1783.
- **tool_insert ep852 — carry loss.** Grasp and lift succeeded
  (z 0.002→0.047 — a low, tilted pinch), but the tool slipped out
  during the carry and ended on the mat at (0.065, 0.041, 0.005),
  short of the tray. The pinch that passes the both-jaw contact check
  is not always a carry-stable pinch for this capsule.
- **cable_clip ep2125 / ep2130 — chain-tension settle-out.** The weld
  carry seats the target segment at the channel, but after weld
  release the capsule chain's tension from the neighbouring segments
  pulls the target segment out of the channel volume before settle
  completes: final misses are z +8 mm / y −19 mm (ep2125) and
  z +5 mm / x −17 mm (ep2130) against the channel centre. The two
  passing scenes started with the target segment 38 mm and 89 mm from
  the clip block; the failures started at 139 mm and 129 mm — the
  longer the free chain span being dragged, the larger the
  settle-out. Executor limit of the single-segment weld carry;
  a two-point (neighbour-segment) stabilisation is the obvious next
  executor revision, not attempted here.

**Attempted and abandoned during development (for the record):**
demo-anchored grasping (nearest open-gripper demo frame as IK seed +
orientation, including full demo-path playback). The twin's pose
error against the recordings is up to ~85 mm (block ep411: the demo's
FK pinch never comes within 85 mm of the twin's cube), so the anchor
pinched objects off-centre (eye: loose end-pinch, carton rotated in
carry) and its transit dragged the ring 46 mm. Removed from all
controllers; `refcore.demo_approach` remains in code, documented as
measured-unreliable at this twin's pose accuracy.

## 8. SOLVABILITY REGISTER (pre-reg §10 + Amendment 2 §A2.2 format)

Development subsets only. Nominal dev denominator = 4 scenes/task.
Executor for all rows: reference/scripted controller, MuJoCo,
**GENERATION: ORACLE-GUIDED** (A2.2 route (b)); cable_clip via the
v1.1 weld device (§5). Scenes not validated are **excluded from the
validated denominator with reason** — never silently dropped.

| Task | Attempted | Validated (executed pass) | Excluded from validated denominator, with reason |
|---|---|---|---|
| block_stack | 4 (411, 453, 489, 498) | **2** (453, 489) | 411 — near-base grasp/lift IK limit of this executor (§7). 498 — executed fully; settle ~8.1 mm vs ≈7.5 mm footprint allowance, criterion not met (§7) |
| eye_drops_to_basket | 4 (1630, 1746, 1783, 1798) | **3** (1630, 1746, 1798) | 1783 — grasp displacement cascade; carton left platform, no executed pass (§7) |
| ring_insert | 4 (7, 33, 167, 195) | **0** | 7, 33, 167, 195 — scripted seat infeasible within honest limits: rim-held ring tilts 39–50°, hole axis cannot be aligned to the 3 mm clearance by this executor (§7). **Task has no validated scenes → no gate on this evidence (A2.2)** |
| tool_insert | 4 (683, 697, 852, 883) | **1** (683) | 697, 883 — grasp ejection displaced tool out of graspable area (§7). 852 — carry loss after successful lift (§7) |
| cable_clip | 4 (1969, 2020, 2125, 2130) | **2** (1969, 2020) | 2125, 2130 — weld-carried segment settled out of channel volume under chain tension (§7) |

**Validated denominator (dev subsets): block_stack 2/4,
eye_drops_to_basket 3/4, ring_insert 0/4, tool_insert 1/4,
cable_clip 2/4 — total 8/20**, against the nominal dev denominator
of 20. For the full gate, the nominal denominator is 96 (pre-reg §2);
these dev validated counts do **not** extrapolate — the full 96-scene
validation must run with these controllers unchanged, and any scene
it cannot validate is excluded-and-logged the same way.

## 9. Execution / guard notes

- The 20-scene batch ran under `launch_dgx_job.sh` + `job_guard.sh`
  (rundir `runs/solvability-batch1/`, logonly mode, CPU-only,
  `JOB_LOCK=0`). **Mirror: `JOB_GUARD_MIRROR_TARGET` was set** to the
  precedent off-box target
  (`hatch@100.64.115.122:…/guard-mirror-solvability`); the guard log
  records **MIRROR FAIL rc=255** on every attempt (ssh connection
  refused — the Muse VM exposes no sshd), so telemetry is on-box
  only in the rundir's `guard/` folder. **No no-mirror override flag
  was used.** Flagging for the parent: the off-box mirror path is
  broken in this environment (independently confirmed earlier in
  `verification-v2-fingerpad.md` §6).
- The guard's circuit breaker (logonly, per founder decision
  2026-11-22) logged WOULD-KILL lines for a temperature ratchet
  (82 °C) during the batch; the heat is from other tenants' training
  on the shared box (preflight load1 7.65 before our first scene
  finished), the breaker is log-only, and no action was taken against
  any process. Our batch is CPU-only MuJoCo at `nice -n 10`.
- Determinism: the pipeline is deterministic per scene (fixed seeds);
  repeated same-code runs of individual scenes during development
  reproduced identical verdicts and near-identical final poses
  (e.g. ring ep7 seat outcome identical across three runs).
- No v1.1 file was modified. All work is in the DGX v2 workdir.

## 10. Status and what this does not claim

- This is a **builder's report**; under the standing builder ≠ verifier
  rule it requires independent verification in a separate context
  before it is delivered as settled — in particular the register in
  §8 and the ring_insert finding in §7.
- Validated here means exactly A2.2's standard: one executed pass in
  the twin satisfying the mechanical criteria at settle, by an
  oracle-guided reference executor. It says nothing about any
  policy's ability to pass these scenes.
- ring_insert is the open problem: 0/4 validated. The gate must not
  run for ring_insert on this executor's evidence. Recommended next
  routes, in order: (i) human-in-sim teleoperation pass (A2.2 route
  (c)) on the dev subset to establish whether the scenes are
  seatable at all in this twin; (ii) a controller revision that
  re-grasps the ring flat at staging height before the seat.

---

## 11. Revision 2 — ONE global controller revision round (2026-11-23)

**GENERATION: ORACLE-GUIDED** (same executor class as §1–§10).

- **Date:** 2026-11-23 (dev iterations ~02:10–02:55 CT; canonical
  batches ~02:3x–04:0x CT).
- **Discipline:** improvements are GLOBAL only — one controller set
  for every scene, aimed at failure CLASSES. No per-scene patches, no
  conditionals on scene identity, no relaxed joint limits, no changed
  criteria. Development iterations ran on the dev subset only and are
  **not** evidence; verdicts below are the canonical batches'.
- **Code (new versioned files; the frozen batch's code and results
  are untouched — md5s re-verified after this round):**
  `refcore2.py` (md5 `ee9ea862d6f4b3f8b133da0f0ad43c33`),
  `run_reference2.py` (md5 `24c8f7636da79e644adb9be89bda65d1`),
  `run_reference3.py` (md5 `22b4b758d67933ac05fc8ce892262485`;
  rev2.1, see below). Frozen originals: `refcore.py`
  `96aa928072e5487b1fe1f077be293132`, `run_reference.py`
  `a9d29bdb8f07c8b55181d10295f6d457` (unchanged).
- **Canonical batches:** rev2 — `runs/solvability-batch2/`, records
  `reference/results2-canonical/`; rev2.1 — `runs/solvability-batch3/`,
  records `reference/results3-canonical/`. Both launched via
  `launch_dgx_job.sh` + `job_guard.sh` (logonly, CPU-only,
  `JOB_LOCK=0`, `nice -n 10`). Mirror: `JOB_GUARD_MIRROR_TARGET` set
  to the same precedent target as batch1; the guard log records
  **MIRROR FAIL rc=255** on every attempt (9 in batch3; the Muse VM
  exposes no sshd) — telemetry is on-box only in each rundir's
  `guard/` folder. **No no-mirror override flag was used.** Batch3's
  guard log shows no circuit-breaker fires.

### 11.1 What changed globally, and why

| # | Change (applies to every scene) | Target class |
|---|---|---|
| G1 | Staged slow close: 4.8 s total (was 2.4 s), in 30% / 65% / 100% segments, object displacement logged at the checkpoints | grasp ejection / displacement cascade (tool, eye) |
| G2 | Hold = stall − 0.04 rad (was −0.03): firmer carry pinch | carry loss (tool 852) |
| G3 | Slower execution: carry/place moves at dist/0.65 capped 5 s (was dist/0.9, cap 3.5 s); grasp descend 2.2 s (was 1.6 s) | in-pads rotation/slip in transit (ring tilt growth, tool carry) |
| G4 | Stepped lift: ≤35 mm increments, orientation held, continuity-seeded, position-only fallback kept | near-base lift IK-infeasibility (block 411) |
| G5 | Slower release open: 1.2 s (was 0.8 s) | place-settle disturbance (block 498) |
| G6 | Object-axis-aligned grasp candidates added in `grasp_with_retries` (after the caller's yaws, before the tilts) | rotated-object misses (eye 1783) |
| G7 | Block place target 3 mm lower (z +0.030 vs +0.033) + one more `final_servo` iteration — a *place* change, not a criterion change | block 498 settle margin |
| G8 | Ring re-grasp-flat: after the first lift, set the ring back down flat at its start pose, re-grasp with the symmetric outside-rim grasp at tube-centre height (dz +0.006/+0.010), accept only a test lift with measured tilt ≤ 12° (≤3 attempts, varying yaw/dz), then the existing staging servo + stepped descend | ring tilt (0/4 task) |
| G9 | Cable settle order: seat with a 2 mm press, HOLD the weld 1.2 s while the chain relaxes around the seated segment, then release and settle 1.5 s (was: release immediately, settle 1.0 s) | chain-tension settle-out (cable 2125/2130) |

**Development findings that shaped the freeze (dev iterations, not
evidence):**

- A displacement **abort** on the close checkpoints (abort the attempt
  if the object moved > 5 mm mid-close) was tried first and
  **removed**: a benign squeeze-centring pinch of a 30 mm cube
  displaces it 7–20 mm during close (measured on block ep453, a
  frozen PASS, where the abort converted *every* grasp into a
  failure and shoved the cube ~190 mm across retries) —
  indistinguishable in magnitude from ejection. The checkpoints remain
  as a staged close + displacement log only.
- Axis-aligned candidates were initially placed FIRST and reordered
  to after the caller's yaws: on near-square objects an axis-first
  candidate can be a corner grasp that shoves the object before the
  caller's proven first candidate runs (also measured on ep453).

### 11.2 rev2 batch, the latent bug, and rev2.1

The rev2 canonical batch scored **10/20** but tool ep852 returned
**ERROR**: rev2's changes carried the tool past its frozen failure
(carry loss) into `final_servo`'s rescan fallback, which contains a
**latent bug inherited verbatim from the frozen code** —
`place_scan(obj_tgt, o_body, f"servo-rescan {name}")` passes a label
*string* in the `candidates` position; iterating it yields characters
and the call crashes (`UFuncTypeError` on `Rt @ o_body`). The frozen
batch never reached that path, so the bug never fired. **rev2.1**
(`run_reference3.py`) fixes exactly that one call — a real candidate
list plus the label as `label=` — and nothing else: no controller
parameter, strategy, limit or criterion changed (diff-verified).
The rev2.1 canonical batch below is this round's canonical result;
the rev2 batch is retained in full for comparison.

**Cross-batch variance note (not hidden):** the two batches agree on
18/20 scenes. The exceptions are the two marginal-pinch tool scenes:
ep697's grasp formed only in rev2.1, and ep852's tool survived to the
servo only in rev2 — despite identical grasp/carry code. These
pinches sit at a knife's edge (stall −0.039 rad vs a −0.16 full
miss) and their trajectories diverge between batches; treat both
scenes' rev2/rev2.1 differences as sensitivity measurements, not as
code effects.

### 11.3 Per-scene before/after (frozen batch → rev2.1 canonical batch)

| Task | Ep | Frozen | Rev2.1 | One-line outcome (rev2.1) |
|---|---|---|---|---|
| block_stack | 411 | FAIL | **PASS** | stepped lift works: cube_b lifted z 0.015→0.055 from the near-base edge pinch and stacked onto cube_a |
| block_stack | 453 | PASS | **PASS** | stacked at (0.260, 0.049, 0.068) |
| block_stack | 489 | PASS | **PASS** | stacked at (0.236, 0.097, 0.068) |
| block_stack | 498 | FAIL | **PASS** | lower place + extra servo iteration: stack settles inside the footprint criterion |
| eye_drops_to_basket | 1630 | PASS | **PASS** | carton in basket |
| eye_drops_to_basket | 1746 | PASS | **PASS** | carton in basket |
| eye_drops_to_basket | 1783 | FAIL | FAIL | unchanged: no candidate forms a both-jaw pinch; carton ends off-platform at (0.428, −0.179) (frozen: (0.432, −0.145)) |
| eye_drops_to_basket | 1798 | PASS | **PASS** | carton in basket |
| ring_insert | 7 | FAIL | FAIL | re-grasp-flat attempts did not produce a seatable hold; over-peg pose unreachable from the resulting state |
| ring_insert | 33 | FAIL | FAIL | re-grasp test lifts measured tilt 93.6° / 87.3° / 95.9° (see §11.4); ring left at (0.244, 0.058, 0.029), final err 28.2 mm |
| ring_insert | 167 | FAIL | FAIL | re-grasp attempts failed to hold; fallback grasp slipped in carry; ring left flat at its start pose |
| ring_insert | 195 | FAIL | FAIL | as ep7 |
| tool_insert | 683 | PASS | **FAIL — regression** | first candidate now misses full-close (stall −0.16, zero displacement); later candidates cascade the tool 13→35 mm; ends ungraspable at (0.073, −0.027) |
| tool_insert | 697 | FAIL | FAIL | **failure mode moved**: no longer ejected — grasped, lifted, placed *in the slot* (centre_in_slot ✓, tray contact ✓) but axis_angle 82.4° (in-pads rotation during carry; criterion needs axis alignment) |
| tool_insert | 852 | FAIL | FAIL | carry loss recurred in rev2.1 (tool on mat, z 0.0048; the fixed servo-rescan ran and correctly reported infeasible). In rev2 it survived to the servo, then hit the §11.2 bug |
| tool_insert | 883 | FAIL | FAIL | unchanged: grasp infeasible in all branches; tool ends at the workspace edge |
| cable_clip | 1969 | PASS | **PASS** | segment seated in channel (weld device) |
| cable_clip | 2020 | PASS | **PASS** | segment seated in channel (weld device) |
| cable_clip | 2125 | FAIL | FAIL | still settles out: segment z 0.067 vs channel 0.061, nearest-seg xy offset 11.9 mm — the longest free span (139 mm) resists the weld-hold |
| cable_clip | 2130 | FAIL | **PASS** | weld-hold settle order works: segment stays in channel at (0.204, 0.193, 0.063) vs channel (0.202, 0.197, 0.061) |

**Totals: frozen 8/20 → rev2.1 10/20** (block 2→4, eye 3→3,
ring 0→0, tool 1→0, cable 2→3).

### 11.4 Failure-class outcomes

1. **Ring tilt — class resists; mechanism sharpened.** Re-grasp-flat
   was implemented as specified, and it *falsified* the symmetric-grip
   hypothesis: a two-point outside-rim equator grip leaves the ring
   free to rotate about the pad-to-pad grip line, and it pendulums
   until its centre of mass hangs below that line — measured test-lift
   tilts of **87–96°** (ring plane near-vertical), *worse* than the
   one-point rim pinch's 39–50°, whose pad-face contact on the tube
   partially constrains rotation. A level hold needs rotation
   constraint about the grip line (pad-face friction area on the
   tube), which neither grasp provides at the required level.
   ring_insert stays **0/4, excluded, no gate on this evidence**
   (A2.2). Remaining routes unchanged: human-in-sim teleoperation
   (route (c)), or a future executor whose grasp grips the tube with
   a rotation-constraining face contact.
2. **Tool — ejection reduced, in-pads rotation is now the binding
   failure; one regression.** ep697's ejection class is genuinely
   moved (in-slot, wrong axis), and ep852's carry loss is marginal
   rather than systematic (it survived the carry in one of two
   batches). But ep683 — a frozen PASS — regressed to a grasp
   failure under the revised approach timing/candidate order, and no
   tool scene validates under rev2.1. Net for the round: the tool
   register goes **1/4 → 0/4**. Reported plainly: the global changes
   traded the tool task backwards while fixing block and cable.
3. **Block near-base lift — fixed.** G4's stepped lift (with the
   staged close) lifts the near-base cube from the edge pinch that
   the frozen one-shot lift could not move. ep411 validates.
4. **Block 498 settle — fixed by place change only** (G7 + G5), as
   required: no criterion was touched. ep498 validates.
5. **Eye 1783 — class resists.** Axis-aligned candidates (G6) and the
   staged close did not produce a single both-jaw pinch in this
   pose; the displacement cascade still ends off-platform. Stays
   excluded with the frozen diagnosis.
6. **Cable settle-out — partially fixed.** G9's press + weld-hold
   validates ep2130 (129 mm span); ep2125 (139 mm span, the longest)
   still pulls out (z +6 mm, xy 11.9 mm). Stays excluded with its
   diagnosis; a neighbour-segment stabilisation remains the obvious
   next executor revision, not attempted in this round.

### 11.5 SOLVABILITY REGISTER — Revision 2.1 (supersedes §8 for this controller version)

Executor: reference/scripted controller **rev2.1** (`refcore2.py` +
`run_reference3.py`), MuJoCo, **GENERATION: ORACLE-GUIDED** (A2.2
route (b)); cable_clip via the v1.1 weld device (§5). The §8 register
remains the record for the frozen controller version; registers are
per controller version and do not carry scenes across versions.

| Task | Attempted | Validated (executed pass) | Excluded from validated denominator, with reason |
|---|---|---|---|
| block_stack | 4 (411, 453, 489, 498) | **4** (411, 453, 489, 498) | — |
| eye_drops_to_basket | 4 (1630, 1746, 1783, 1798) | **3** (1630, 1746, 1798) | 1783 — grasp displacement cascade; no both-jaw pinch forms in this pose (§11.4.5) |
| ring_insert | 4 (7, 33, 167, 195) | **0** | 7, 33, 167, 195 — rim-held ring cannot be held level by this executor: one-point pinch tilts 39–50°, symmetric two-point re-grasp pendulums to 87–96° (§11.4.1). **Task has no validated scenes → no gate on this evidence (A2.2)** |
| tool_insert | 4 (683, 697, 852, 883) | **0** | 683 — grasp regression under rev2 timing (§11.4.2). 697 — placed in slot with axis 82.4° off (in-pads rotation). 852 — carry loss (marginal pinch). 883 — grasp ejection to workspace edge |
| cable_clip | 4 (1969, 2020, 2125, 2130) | **3** (1969, 2020, 2130) | 2125 — segment settled out of channel under chain tension despite press + weld-hold (longest free span, 139 mm) |

**Validated denominator (dev subsets, rev2.1): block_stack 4/4,
eye_drops_to_basket 3/4, ring_insert 0/4, tool_insert 0/4,
cable_clip 3/4 — total 10/20**, against the nominal dev denominator
of 20 (frozen controller: 8/20). For the full gate the nominal
denominator is 96 (pre-reg §2); whichever controller version is
carried forward must run the full validation unchanged, and any
scene it cannot validate is excluded-and-logged the same way. Note
the version choice is now a real trade: rev2.1 validates block 411/
498 and cable 2130 that the frozen controller missed, and loses
tool 683 that the frozen controller validated.

### 11.6 Status

- This is a **builder's report**; under the standing builder ≠
  verifier rule it requires independent verification in a separate
  context before it is delivered as settled — in particular the
  register in §11.5, the ring pendulum finding in §11.4.1, and the
  ep683 regression in §11.4.2.
- Validated here means exactly A2.2's standard: one executed pass in
  the twin satisfying the mechanical criteria at settle, by an
  oracle-guided reference executor. It says nothing about any
  policy's ability to pass these scenes.
- No v1.1 file was modified. All work is in the DGX v2 workdir
  (`reference/refcore2.py`, `run_reference2.py`, `run_reference3.py`,
  `results2-canonical/`, `results3-canonical/`).

---

## 12. Errata — 2026-11-23 (correcting the original batch-1 narrative, §§1–10)

**This section is appended, dated, and labeled as an errata pass. It
corrects the RECORD in §§1–10 (the original batch-1 narrative) in
response to findings F1–F4 of `verification-v2-solvability.md`. No
original text in §§1–11 has been edited, and §11 (Revision 2) is
untouched. Each correction below was re-checked against the raw
batch-1 records on the DGX
(`/home/openclaw/muse-work/zk0-study001-twin-v2-20261122/reference/results/`)
before writing. None of these corrections changes any verdict, any
count, the validated denominators, or the ring_insert no-gate
consequence; the exclusions themselves all stand.**

### F1 — block_stack ep498: the logged exclusion reason was wrong (§§6, 7, 8)

The original narrative (§6 table, §7 diagnosis, §8 register) said
ep498 failed because the stack "settled ~8.1 mm off" and was "just
outside the 25%-inset footprint criterion (allowance ≈7.5 mm)". That
reason is **wrong** and is superseded by this erratum. The batch
record (`block_stack__ep498.json`) shows the footprint criterion
**PASSED**: `within_inset_footprint: true` and
`contact_with_lower: true`. The v1.1 criterion is per-axis —
(|Δx|, |Δy| ≤ (0.030/2)·0.75 = **11.25 mm** per-axis allowance, not
≈7.5 mm). The actual failing criterion was
**`no_other_support_contact: false`**: `upper_contacts` is
`[cube_b, moving_jaw_so101_v1]` — the moving jaw was still touching
the upper cube at settle, and the release-retreat move is logged in
the record as `release:retreat … solved: false` (retreat unsolved).
The FAIL verdict and the exclusion of ep498 from the validated
denominator stand, on this corrected reason.

### F2 — cable_clip pinch-gap figures were wrong (§§1, 5)

The original §1 statement that the "minimum achievable face gap is
≈8 mm" at the gripper joint limit, and §5's consequence that the
Ø5 mm cable therefore clears it by a ~3 mm margin, are **wrong**
and are superseded by this erratum. The correct measured values
(verifier probes `probe_padgap2.py` / `probe_padgap3.py` in the
verifier's scratch copy, re-run and reproduced in this errata pass)
are: the **pinch-centre face gap is 6.15 mm at the joint limit**
(−0.1745 rad) and **6.95 mm at the controllers' commanded close**
(−0.16 rad), against the cable's **Ø5 mm** (`CABLE_R = 0.0025`) —
a margin of **1.15 mm** at the joint limit, not ~3 mm. The ≈8 mm
figure came from subtracting only the nominal pad half-thicknesses
assuming parallel faces. For completeness: the true minimum
pad-to-pad distance over the faces is effectively ~0 mm
(`mj_geomDistance` 0.21 mm at −0.16 rad, −0.86 mm at the limit —
the pads meet at one edge), so "minimum face gap ≈8 mm" must not be
cited as a minimum over the faces. The §5 weld-device justification
stands, restated on the pinch-centre gap: at the pinch centre —
the only place this controller pinches — 6.15 mm > Ø5 mm, so a
centred pad pinch of the cable cannot form.

### F3 — ring_insert §7: batch records vs development measurements were mixed

The §7 ring narrative mixes development measurements with batch
records without marking which is which. Corrected attribution:

- **Batch records** (`ring_insert__ep{7,33,167,195}.json`, batch 1):
  initial staging tilts are **39.6° (ep7), 39.0° (ep167) and
  39.8° (ep195)** — a batch range of **39.0–39.8°**, not "39–50°"
  (ep33 has no staging tilt; its lift failed). On **ep7** the
  staging servo did **not** converge: `ring_staging_err_mm` runs
  **14.5 → 15.3 mm** over 10 steps, and tilt-levelling was
  IK-infeasible at ep7's staging pose (the `place_scan` over-peg
  solution records `wrist_flex_deg: 95.0`, exactly the model limit;
  the levelling loop records the single 39.6° tilt entry and then
  exits). On **ep167** the staging servo did converge
  (**5.5 → 0.3 mm**) and levelling succeeded in steps
  (39.0° → 29.5° → 20.5° → 12.3° → 3.6° → 1.1°), yet the seat
  **still** failed (final planar error 11.3 mm, ring z 0.119 —
  descend blocked short of the seat). On **ep195** staging ran
  6.5 → 2.7 → 1.2 mm and levelling progressed 39.8° → 13.9° before
  an IK step failed.
- **Development measurements (NOT in any batch record):** the §7
  sentence "the small-step staging servo does converge planar error
  **12.3 → 1.9 mm**, but the tilted hole axis catches the peg tip
  even at 1.9 mm (measured on ep7)" appears in **no batch record**
  and must not be cited as batch evidence — in the batch records
  12.3 appears only as a **tilt value** on ep167, and no batch
  staging error of 1.9 mm exists. Likewise, the ~50° upper end of
  the "39–50°" tilt range appears only in development-era code
  comments, not in any batch record.

The register conclusion is unchanged — the scripted seat is
infeasible for this controller on all 4 scenes, ring_insert stays
**0/4 validated, excluded, no gate on this evidence (A2.2)** — but
the single-mechanism story in §7 is ep7's story; ep167 shows the
seat fails even with the ring levelled, so at least one further
blocker (descend alignment) exists.

### F4 — grasp-record count slips (§7), plus a cosmetic note-text defect

- **eye_drops_to_basket ep1783:** the batch record contains
  **13 grasp records**, not the "16 grasp records" stated in §7
  (all with stall −0.159 rad, as stated). The §7 figure is
  presumably a development-run count.
- **tool_insert ep883:** the batch record contains **16 grasp
  records**, not the "17 grasp records" stated in §7. The §7
  figure is likewise presumably a development-run count.

Both diagnoses are otherwise unchanged (displacement finals match
the records). Also noted, cosmetic only: `do_ring` appends the
phase note **"seated pose reached" unconditionally**, even when
the log records `ring_seated: false` — which is the case in all
four ring batch records. The note text is misleading; verdicts are
unaffected, because they come from `evaluate()`, not the phase
notes.
---

## 13. Micro-corrections — 2026-11-23 (Revision 2 verification findings RF1/RF2, and the verifier's register ruling)

**This section is appended, dated, and labeled as a micro-correction pass. It corrects the RECORD in §11 in response to findings RF1/RF2 of the "Revision 2 / rev2.1" verification in `verification-v2-solvability.md`, and records that verification's register ruling (§R5) so this report is self-consistent. No original text in §§1–12 has been edited. Neither correction changes any verdict, any per-version count, or the ring_insert no-gate consequence.**

### RF1 — §11.2 cross-batch agreement is 19/20, not 18/20

§11.2's cross-batch variance note is superseded: the rev2 and rev2.1 canonical batches agree on **19/20** scenes, not 18/20. ep697's records are **byte-identical** across the canonical batches (`results2-canonical/` and `results3-canonical/`) — its grasp formed in **both** (stall −0.039 rad). The **only** differing record is **ep852** (rev2 **ERROR** via the latent `final_servo` bug → rev2.1 **FAIL**). Any ep697 divergence belongs to development iterations, not the canonical batches.

### RF2 — §11.4.1 one-point ring tilt range is 39.0–39.8°

§11.4.1's "39–50°" one-point rim-pinch tilt range is superseded by the batch-record range **39.0–39.8°** (ep7 39.6°, ep167 39.0°, ep195 39.8°) — the same correction as erratum F3 in §12. The pendulum mechanism conclusion in §11.4.1 is **unaffected**: 39.0–39.8° (one-point) vs 87–96° (two-point re-grasp) still falsifies the symmetric-grip hypothesis.

### Register ruling — union register 11/20

Per the verifier's ruling (§R5 of the Revision 2 verification), the **union register is permitted under Amendment 2 §A2.2**: **block_stack 4/4, eye_drops_to_basket 3/4, ring_insert 0/4, tool_insert 1/4, cable_clip 3/4 — total 11/20**. Validating controller version per scene: **tool 683: frozen set** (`refcore.py` / `run_reference.py`); **block 411/498 and cable 2130: rev2.1** (`refcore2.py` / `run_reference3.py`); all other validated scenes validate under both versions. The per-version registers remain on record (§8: 8/20 frozen; §11.5: 10/20 rev2.1). **Caution:** future union additions must be **registered global revision rounds** (no per-scene patches, independently verified), so the union stays a proof set, not a search over executors.
